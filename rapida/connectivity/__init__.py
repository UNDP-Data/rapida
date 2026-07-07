import json
import os.path
from rapida.util.bbox_param_type import get_best_semantic_label
from rich.progress import Progress
from rapida.connectivity.io import prepare_osm_pbf,extract_health_sites, extract_origins_from_geojson, extract_origins
from rapida.connectivity.graph import compile_valhalla_graph
from rapida.connectivity.isochrone import connectivity_areas
# from rapida.cli.assess import assess
# import click
# from rapida.project.project import Project
# from tempfile import TemporaryDirectory



async def run_connectivity_analysis(
        bbox:tuple[float, float, float, float]=None, travel_mode:str=None, time_intervals:list[int] =None,
        dst_dir:str=None, barriers_dataset:str=None, barriers_layer:str=None, barriers_buffer:int=None,
        sites_dataset:str=None, sites_layer:str=None,pop_vars:str|tuple[str]=None,
        progress:Progress=None
    ):
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
        tar_path=dag_tar_path, origins=origins, travel_mode=travel_mode, intervals_minutes=time_intervals)
    isochrones_path = os.path.join(dest_dir,  'isochrones.geojson')
    with open(isochrones_path, "w") as f:
        json.dump(results, f, indent=2)

    # with TemporaryDirectory(dir=dest_dir, delete=False) as project_folder:
    #     project = Project(path=project_folder, polygons=isochrones_path, comment='temp project for conn isochrones')
    #
    #
    #     with click.Context(assess) as ctx:
    #         ctx.ensure_object(dict)
    #         ctx.obj['progress'] = progress
    #         # 2. Use invoke. Do NOT pass 'ctx' manually here.
    #         # Click intercepts this and injects it as the first argument automatically.
    #         await ctx.invoke(
    #             assess,
    #             components=('population',),
    #             variables=pop_vars,
    #             year=2026,
    #             project=project.path,
    #             force=False
    #         )


    if barriers_dataset is not None:
        barrier_results = await connectivity_areas(
        tar_path=dag_tar_path, origins=origins, travel_mode=travel_mode, intervals_minutes=time_intervals,
        barriers_dataset=barriers_dataset, barriers_layer=barriers_layer, barriers_buffer=barriers_buffer
                             )
        with open(os.path.join(dest_dir, 'isochrones_with_barriers.geojson'), "w") as f:
            json.dump(barrier_results, f, indent=2)


    return