import json
import os.path
from rapida.util.bbox_param_type import get_best_semantic_label
from rich.progress import Progress
from rapida.connectivity.io import prepare_osm_pbf,extract_health_sites, extract_origins_from_geojson
from rapida.connectivity.graph import compile_valhalla_graph
from rapida.connectivity.isochrone import connectivity_areas



async def run_connectivity_analysis(
        bbox:tuple[float, float, float, float]=None, travel_mode:str=None, time_intervals:list[int] =None,
        dst_dir:str=None, barriers_dataset:str=None, barriers_layer:str=None, barriers_buffer:int=None, progress:Progress=None
    ):
    bbox_label = get_best_semantic_label(bbox=bbox)
    dest_dir = os.path.join(dst_dir, bbox_label)
    bbox_pbf = await prepare_osm_pbf(bbox=bbox, dst_dir=dest_dir, progress=progress)
    health_sites = await extract_health_sites(pbf_path=bbox_pbf, dst_dir=dest_dir, progress=progress)
    dag_tar_path = await compile_valhalla_graph(pbf_path=bbox_pbf,dst_dir=dest_dir, progress=progress)
    origins = extract_origins_from_geojson(geojson_path=health_sites)
    results = await connectivity_areas(
        tar_path=dag_tar_path, origins=origins, travel_mode=travel_mode, intervals_minutes=time_intervals,
        barriers_dataset=barriers_dataset, barriers_layer=barriers_layer, barriers_buffer=barriers_buffer
                                       )
    with open(os.path.join(dest_dir, 'isochrones.geojson'), "w") as f:
        json.dump(results, f, indent=2)

    return