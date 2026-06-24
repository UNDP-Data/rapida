import os
import json
import asyncio
from rapida.connectivity.runcli import run_cli
from valhalla import get_config
from valhalla.config import _sanitize_config, default_config
from typing import Union
from pathlib import Path


def get_config_fixed(
    tile_extract: Union[str, Path] = "valhalla_tiles.tar",
    tile_dir: Union[str, Path] = "valhalla_tiles",
    verbose: bool = False,
) -> dict:
    """
    Returns a default Valhalla configuration.

    :param tile_extract: The file path (with .tar extension) of the tile extract (mjolnir.tile_extract), if present. Preferred over tile_dir.
    :param tile_dir: The directory path where the graph tiles are stored (mjolnir.tile_dir), if present.
    :param verbose: Whether you want to see Valhalla's logs on stdout (mjolnir.logging). Default False.
    """

    config = _sanitize_config(default_config.copy())

    config["mjolnir"]["tile_dir"] = (
        ""
        if isinstance(tile_dir, str) and not str(tile_dir)
        else str(Path(tile_dir).resolve(strict=True))
    )
    config["mjolnir"]["tile_extract"] = (
        ""
        if isinstance(tile_extract, str) and not str(tile_extract)
        else str(Path(tile_extract).resolve())
    )

    config["logging"]["type"] = "std_out" if verbose else ""

    return config


async def compile_valhalla_graph(pbf_path: str, dst_dir: str, progress=None) -> str:
    """
    Compiles the raw OSM PBF into a highly optimized Valhalla routing DAG.
    Offloads the C++ execution to a background thread to keep uvloop unblocked.
    """
    os.makedirs(dst_dir, exist_ok=True)
    tile_dir = os.path.join(dst_dir, "valhalla_tiles")
    os.makedirs(tile_dir, exist_ok=True)
    tar_path = os.path.join(dst_dir, "valhalla_tiles.tar")
    #os.makedirs(tar_path, exist_ok=True)
    config_path = os.path.join(dst_dir, "valhalla.json")

    # 1. Generate the Valhalla JSON configuration natively
    if progress:
        progress.console.print("[cyan]Generating Valhalla engine configuration...[/cyan]")

    try:
        valhalla_conf = get_config(
            tile_dir=tile_dir,
            tile_extract=tar_path,
            verbose=False
        )
    except Exception:
        valhalla_conf = get_config_fixed(
            tile_dir=tile_dir,
            tile_extract=tar_path,
            verbose=False
        )
        # ---------------------------------------------------------
        # INJECT CUSTOM LIMITS BEFORE SAVING THE BUILD CONFIG
        # ---------------------------------------------------------
        if "service_limits" not in valhalla_conf:
            valhalla_conf["service_limits"] = {}
        if "isochrone" not in valhalla_conf["service_limits"]:
            valhalla_conf["service_limits"]["isochrone"] = {}

        # Expand the max_locations limit to allow system-wide bulk routing
        valhalla_conf["service_limits"]["isochrone"]["max_locations"] = 5000
        # ---------------------------------------------------------

    with open(config_path, "w") as f:
        json.dump(valhalla_conf, f, indent=4)

    # 2. Define the blocking C++ execution function
    def run_compiler():
        # 1. Build the Admin Database
        # This parses country borders, timezones, and local driving rules (e.g., right vs left side of road)
        if progress:
            progress.console.print("[cyan]Building admin rules database...[/cyan]")
        run_cli([
            "valhalla_build_admins",  "-c",  config_path, pbf_path
        ])

        # 2. Build the Routing Tiles
        # This generates the actual mathematical DAG and writes it to the 'valhalla_tiles' folder
        if progress:
            progress.console.print("[cyan]Building routing graph ...[/cyan]")
        run_cli([
            "valhalla_build_tiles", "-c", config_path, pbf_path
        ])

        # 3. Compress into the Extract
        # This reads the generated folder and packs it into the high-performance memory-mapped .tar file
        if progress:
            progress.console.print("[cyan]Compressing graph into a tarball...[/cyan]")
        run_cli([
            "valhalla_build_extract", "-c", config_path, "-v", "--overwrite"
        ])

    # 3. Offload compilation to a worker thread
    if progress:
        progress.console.print("[cyan]Compiling binary DAG ...[/cyan]")

    await asyncio.to_thread(run_compiler)

    if progress:
        progress.console.print(f"[bold green]✓ Valhalla DAG compiled successfully: {tar_path}[/bold green]")

    return tar_path