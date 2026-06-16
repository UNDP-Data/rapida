import os.path
from rapida.ntl.utils import get_custom_bbox_label
from rich.progress import Progress
from rapida.connectivity.io import prepare_osm_pbf
from rapida.connectivity.graph import compile_valhalla_graph



async def run_connectivity_analysis(bbox:tuple[float, float, float, float]=None, dst_dir:str=None, progress:Progress=None ):

    bbox_pbf = await prepare_osm_pbf(bbox=bbox, dst_dir=dst_dir, progress=progress)
    bbox_label = get_custom_bbox_label(bbox=bbox)
    dest_dir = os.path.join(dst_dir, bbox_label)
    dag_tar_path = await compile_valhalla_graph(pbf_path=bbox_pbf,dst_dir=dest_dir, progress=progress)


    return