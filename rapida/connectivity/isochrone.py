import json
import asyncio
from pathlib import Path

import geopandas
from valhalla import Actor
from shapely.geometry import shape, mapping, JOIN_STYLE
from shapely import make_valid
from pyproj import Transformer
from shapely.ops import transform
import logging
from rapida.connectivity.io import read_barriers
logger = logging.getLogger(__name__)


# Map user modes to Valhalla's internal costing models
MODE_MAP = {
    "walk": "pedestrian",
    "drive": "auto",
    "bike": "bicycle"
}

#Example max distance configured for the service
ISOCHRONES_MAX_RADIUS = 1000.0

project_to_meters = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True).transform
project_to_degrees = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True).transform



def make_isochrones_disjoint(gdf, time_col="time", group_col=None):
    """Converts overlapping concentric isochrones into mutually exclusive rings."""
    # Ensure data is sorted by travel time (smallest/inner first)
    gdf["geometry"] = gdf["geometry"].make_valid()
    sort_cols = [group_col, time_col] if group_col else [time_col]
    gdf = gdf.sort_values(sort_cols).reset_index(drop=True)

    # Copy to store the result
    result_gdf = gdf.copy()

    # Iterate backwards to subtract the smaller inner polygon from the larger outer one
    for i in range(len(gdf) - 1, 0, -1):
        # If tracking multiple starting points, ensure they belong to the same group
        if group_col and gdf.loc[i, group_col] != gdf.loc[i - 1, group_col]:
            continue

        # Subtract the inner geometry from the outer geometry
        result_gdf.loc[i, "geometry"] = gdf.loc[i, "geometry"].difference(
            gdf.loc[i - 1, "geometry"]
        )

    return result_gdf




async def connectivity_areas(
        tar_path: str,
        origins: list[tuple[float, float]],
        travel_mode: str,
        intervals_minutes: list[int],
        barriers_dataset:str=None,
        barriers_layer:str=None,
        barriers_buffer:int=None,
        disjoint:bool=False,
        smooth:bool=False,
        radius:int=ISOCHRONES_MAX_RADIUS,
        progress=None
) -> dict:
    tar_file = Path(tar_path)
    build_config_file = tar_file.parent / "valhalla.json"

    #runtime_config_file = tar_file.parent / "valhalla_runtime.json"

    # # 1. Bypass Valhalla's hardcoded multi-origin security limits
    # with open(build_config_file, "r") as f:
    #     valhalla_config = json.load(f)
    #
    # if "service_limits" not in valhalla_config:
    #     valhalla_config["service_limits"] = {}
    # if "isochrone" not in valhalla_config["service_limits"]:
    #     valhalla_config["service_limits"]["isochrone"] = {}
    #
    # # Update top-level orchestration limits
    # valhalla_config["service_limits"]["isochrone"]["max_locations"] = 50000
    # valhalla_config["service_limits"]["isochrone"]["max_distance"] = 2000000
    # valhalla_config["service_limits"]["isochrone"]["max_contours"] = 20
    #
    #
    # # THE EXACT MATCHING KEY FROM YOUR CONFIG:
    # valhalla_config["service_limits"]["max_exclude_polygons_length"] = 500000  # Bump to 500km perimeter length
    # valhalla_config["service_limits"]["allow_hard_exclusions"] = True
    #
    # with open(runtime_config_file, "w") as f:
    #     json.dump(valhalla_config, f)


    contours = [{"time": int(mins)} for mins in intervals_minutes]
    barriers_coords = read_barriers(src_path=barriers_dataset, src_layer=barriers_layer, barriers_buffer=barriers_buffer)
    locations = [{"lon": float(lon), "lat": float(lat), "radius": radius} for lon, lat in origins]


    if progress:
        routing_task_id = progress.add_task(
            description=f"[cyan]Calculating unified system service areas...",
            total=1
        )

    def run_routing():
        actor = Actor(str(build_config_file))
        results = {"type": "FeatureCollection", "features": []}


        costing_name = MODE_MAP.get(travel_mode)
        # 1. Define mode-specific costing options using Valhalla's internal keys
        costing_options = {}

        if costing_name == "auto":
            costing_options["auto"] = {
                "use_tracks": 1.0,  # Allows routing on unpaved rural tracks
                "ignore_access": True,  # Bypasses minor OSM access restriction tags
                "unclassified_penalty": 0,
            }
        elif costing_name == "pedestrian":
            costing_options["pedestrian"] = {
                "use_tracks": 1.0,
                "use_hills": 0.5,
            }
        elif costing_name == "bicycle":
            costing_options["bicycle"] = {
                "use_roads": 0.5,
                "use_hills": 0.5,
            }

        # 2. Fire a single bulk request per mode
        request = {
            "locations": locations,
            "costing": costing_name,
            "costing_options": costing_options,
            "contours": contours,
            "polygons": True,
            "denoise": 0,  # Valhalla's native pre-smoothing captures all the details when 0
            "reverse":True,
            "generalize": 50


        }

        if barriers_coords:
            request['exclude_polygons'] = barriers_coords
        try:
            response_str = actor.isochrone(json.dumps(request))
            isochrone_geojson = json.loads(response_str)

            # 3. Intercept Valhalla's output and apply Shapely smoothing
            for fid, feature in enumerate(isochrone_geojson.get("features", []), start=1):
                if smooth:
                    raw_geom_wgs84 = make_valid(shape(feature["geometry"]))

                    # 1. Get the bounding box of the raw WGS84 polygon
                    minx, miny, maxx, maxy = raw_geom_wgs84.bounds

                    # 2. Replicate Valhalla's exact internal grid sizing logic from `thor/isochrone.cc`
                    dx_deg = maxx - minx
                    dy_deg = maxy - miny

                    # Valhalla targets ~300 bins but rigidly clamps the degree step between 0.001 and 0.005
                    valhalla_degree_step = max(0.001, min(0.005, max(dx_deg, dy_deg) / 300.0))

                    # 3. Convert that exact degree step to flat meters at the local latitude
                    # 1 degree latitude is ~111,320 meters.
                    real_cell_size_meters = valhalla_degree_step * 111320

                    # 4. Use this true runtime value for your smoothing radius (e.g., 1.5x to 2x the cell size)
                    smooth_radius_meters = real_cell_size_meters * 1.5

                    # 5. Apply the Morphological Opening/Closing (Buffer out, in, out)
                    geom_meters = make_valid(transform(project_to_meters, raw_geom_wgs84))


                    # 4. Morphological Closing (Now using actual physical meters)
                    smooth_geom_meters = geom_meters.buffer(
                        smooth_radius_meters,
                        join_style=JOIN_STYLE.round
                    ).buffer(
                        -(smooth_radius_meters * 2),
                        join_style=JOIN_STYLE.round
                    ).buffer(
                        smooth_radius_meters,
                        join_style=JOIN_STYLE.round
                    )

                    # 5. Metric Simplification (Drop vertices closer than 50 meters to the line)
                    smooth_geom_meters = smooth_geom_meters.simplify(20, preserve_topology=True)

                    # 6. Convert back to WGS84 degrees so the GeoJSON renders on a map properly
                    final_geom_wgs84 = transform(project_to_degrees, smooth_geom_meters)

                    feature["geometry"] = mapping(final_geom_wgs84)



                feature["properties"].update({
                    "mode": travel_mode,
                    "type": "system_catchment",
                    "id": fid
                })
                results["features"].append(feature)


        except Exception as e:
            logger.error(e)
            # Fallback to logger if present in your environment
            if progress:
                progress.console.print(f"[red]Valhalla bulk routing failed: {e}[/red]")
            raise e
        finally:
            if progress and routing_task_id is not None:
                progress.advance(routing_task_id)

        return geopandas.GeoDataFrame.from_features(results, crs='EPSG:4326')

    isos_gdf = await asyncio.to_thread(run_routing)

    if disjoint:

        # Example Usage:

        isos_gdf = make_isochrones_disjoint(isos_gdf, time_col='contour')

    return isos_gdf