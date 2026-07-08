import datetime
import json
import os.path
from rapida.util.bbox_param_type import get_best_semantic_label
import geopandas as gpd
import logging
from rich.progress import Progress
from rapida.connectivity.io import prepare_osm_pbf,extract_health_sites, extract_origins_from_geojson, extract_origins
from rapida.connectivity.graph import compile_valhalla_graph
from rapida.connectivity.isochrone import connectivity_areas
from rapida.cli.assess import assess
import click
from rapida.project.project import Project
from tempfile import TemporaryDirectory

logger = logging.getLogger(__name__)

async def run_connectivity_analysis(
        bbox:tuple[float, float, float, float]=None, travel_mode:str=None, time_intervals:list[int] =None,
        dst_dir:str=None, barriers_dataset:str=None, barriers_layer:str=None, barriers_buffer:int=None,
        sites_dataset:str=None, sites_layer:str=None,pop_vars:str|tuple[str]=None,
        progress:Progress=None, year=datetime.datetime.now().year, disjoint:bool=False
    ):
    if bbox is None:
        assert sites_dataset is not None, f'site_dataset has to be provided when bbox is not'
        try:
            slayer = int(sites_layer)
        except ValueError:
            slayer = sites_layer
        gdf = gpd.read_file(sites_dataset, layers=slayer)
        if not gdf.crs.is_geographic:
            gdf.to_crs('EPSG:4326', inplace=True)
        bbox = gdf.total_bounds
    bbox_label = get_best_semantic_label(bbox=bbox)
    dest_dir = os.path.join(dst_dir, bbox_label)
    bbox_pbf = await prepare_osm_pbf(bbox=bbox, dst_dir=dest_dir, progress=progress)
    if sites_dataset is None:
        sites = await extract_health_sites(pbf_path=bbox_pbf, dst_dir=dest_dir, progress=progress)
    else:
        sites = sites_dataset

    dag_tar_path = await compile_valhalla_graph(pbf_path=bbox_pbf,dst_dir=dest_dir, progress=progress)
    origins = extract_origins(sites_dataset=sites, src_layer=sites_layer)


    results = await connectivity_areas(
        tar_path=dag_tar_path, origins=origins, travel_mode=travel_mode, intervals_minutes=time_intervals, disjoint=disjoint)


    isochrones_path = os.path.join(dest_dir,  'isochrones.geojson')
    with open(isochrones_path, "w") as f:
        json.dump(results, f, indent=2)
    if pop_vars:
        with TemporaryDirectory(dir=dest_dir, delete=True) as project_folder:
            project = Project(path=project_folder, polygons=isochrones_path, comment='temp project for conn isochrones')
            with click.Context(assess) as ctx:
                ctx.ensure_object(dict)
                ctx.obj['progress'] = progress
                # 2. Use invoke. Do NOT pass 'ctx' manually here.
                # Click intercepts this and injects it as the first argument automatically.
                ctx.invoke(
                    assess,
                    components=('population',),
                    variables=pop_vars,
                    year=year,
                    project=project.path,
                    force=False
                )
                stat_gpkg_path = os.path.join(project_folder,'data', f'{project.name}.gpkg')
                pop_stat_gdf = gpd.read_file(stat_gpkg_path, layer='stats.population')
                if not disjoint:
                    pop_stat_gdf = pop_stat_gdf.iloc[pop_stat_gdf.geometry.area.sort_values(ascending=False).index]
                pop_stat_gdf = pop_stat_gdf.to_crs('EPSG:4326')

                pop_stat_gdf.to_file(
                    filename=isochrones_path,
                    driver="GeoJSON",
                    engine="pyogrio",
                    mode="w",
                    layer='isochrones',
                    promote_to_multi=True,
                    index=False
                )

    if barriers_dataset is not None:
        logger.info(f'Computing isochrones with barriers')
        barrier_results = await connectivity_areas(
        tar_path=dag_tar_path, origins=origins, travel_mode=travel_mode, intervals_minutes=time_intervals,
        barriers_dataset=barriers_dataset, barriers_layer=barriers_layer, barriers_buffer=barriers_buffer, disjoint=disjoint
                             )
        barrier_isochrones_path = os.path.join(dest_dir, 'isochrones_with_barriers.geojson')
        with open(barrier_isochrones_path, "w") as f:
            json.dump(barrier_results, f, indent=2)
        if pop_vars:
            logger.info(f'Computing zonal stats for barrier isochrones')
            with TemporaryDirectory(dir=dest_dir, delete=True) as project_folder:
                project = Project(path=project_folder, polygons=barrier_isochrones_path, comment='temp project for conn isochrones')
                with click.Context(assess) as ctx:
                    ctx.ensure_object(dict)
                    ctx.obj['progress'] = progress
                    # 2. Use invoke. Do NOT pass 'ctx' manually here.
                    # Click intercepts this and injects it as the first argument automatically.
                    ctx.invoke(
                        assess,
                        components=('population',),
                        variables=pop_vars,
                        year=year,
                        project=project.path,
                        force=False
                    )
                    stat_gpkg_path = os.path.join(project_folder, 'data', f'{project.name}.gpkg')
                    barrier_pop_stat_gdf = gpd.read_file(stat_gpkg_path, layer='stats.population')
                    if not disjoint:
                        barrier_pop_stat_gdf = barrier_pop_stat_gdf.iloc[pop_stat_gdf.geometry.area.sort_values(ascending=False).index]
                    barrier_pop_stat_gdf = barrier_pop_stat_gdf.to_crs('EPSG:4326')

                    pop_col_names = [f'{popv}_{year}' for popv in pop_vars]
                    new_pop_col_names = [f'{popv}_{year}_barrier' for popv in pop_vars]
                    col_name_dict = dict(zip(pop_col_names, new_pop_col_names))
                    barrier_pop_stat_gdf.rename(columns=col_name_dict, inplace=True)

                    data_cols = ['contour']+pop_col_names
                    data_frame = pop_stat_gdf[data_cols]
                    barrier_pop_stat_gdf = barrier_pop_stat_gdf.merge(data_frame, on='contour', how='left')
                    for pvar, bar_pvar in col_name_dict.items():
                        barrier_pop_stat_gdf[f'{pvar}_{bar_pvar}_difference'] = barrier_pop_stat_gdf[pvar] - barrier_pop_stat_gdf[bar_pvar]

                    barrier_pop_stat_gdf.to_file(
                        filename=barrier_isochrones_path,
                        driver="GeoJSON",
                        engine="pyogrio",
                        mode="w",
                        layer='barrier_isochrones',
                        promote_to_multi=True,
                        index=False
                    )

    return