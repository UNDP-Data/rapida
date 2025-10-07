from __future__ import annotations
from typing import Tuple
import numpy as np
import rasterio
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling
from rasterio.transform import Affine
from rasterio.warp import transform_bounds, transform
from pyproj import Transformer
import mgrs

# --- internal helpers ---------------------------------------------------

def _tile_id_from_center(ds) -> str:
    # Use dataset center in WGS84 to derive the 100 km MGRS tile id (e.g., "36NYG")
    (minx, miny, maxx, maxy) = ds.bounds
    cx = (minx + maxx) / 2
    cy = (miny + maxy) / 2
    # to lon/lat
    (lon,), (lat,) = transform(ds.crs, "EPSG:4326", [cx], [cy])
    return mgrs.MGRS().toMGRS(lat, lon, MGRSPrecision=0)  # just the 100 km tile

def _utm_epsg_from_band(band_letter: str, zone: int) -> int:
    # MGRS bands north of equator
    north_bands = set("NPQRSTUVWX")
    return (32600 if band_letter in north_bands else 32700) + zone

def _tile_ul_from_mgrs(tile_id: str) -> Tuple[int, float, float]:
    """
    From a tile id like 'T36NYG' or '36NYG' return:
      (utm_epsg, x_ul, y_ul) for the 100 km tile UL corner.
    """
    t = tile_id.upper().lstrip("T")
    # parse zone (2 digits typical, but be safe)
    i = 1 if len(t) >= 2 and not t[1].isdigit() else 2
    zone = int(t[:i])
    band = t[i]
    square = t[i+1:i+3]

    utm_epsg = _utm_epsg_from_band(band, zone)

    # Build SW corner MGRS with full 1m precision (5+5 digits -> 10 total)
    sw_mgrs = f"{zone}{band}{square}0000000000"
    m = mgrs.MGRS()
    lat_sw, lon_sw = m.toLatLon(sw_mgrs)  # WGS84

    tf = Transformer.from_crs("EPSG:4326", f"EPSG:{utm_epsg}", always_xy=True)
    x_sw, y_sw = tf.transform(lon_sw, lat_sw)

    # UL is 100 km north of SW
    x_ul, y_ul = x_sw, y_sw + 100_000.0
    return utm_epsg, x_ul, y_ul

def _bbox_contains(outer, inner, tol=0.0) -> bool:
    L1, B1, R1, T1 = outer
    L2, B2, R2, T2 = inner
    return (L1 - tol) <= L2 and (B1 - tol) <= B2 and (R1 + tol) >= R2 and (T1 + tol) >= T2

# --- public API ---------------------------------------------------------

def align_s2_image(
    src_path: str,
    out_res: float = 10.0,
    *,
    resampling: Resampling = Resampling.bilinear,  # use nearest for SCL/QA masks
    src_nodata=None,
    dst_nodata=np.nan,
    warp_mem_limit_mb: int = 512,
    require_full_coverage: bool = True,
) -> WarpedVRT:
    """
    Return a WarpedVRT of a Sentinel-2 band aligned to the canonical MGRS 100 km tile grid.

    - Detects tile from the image center via MGRS.
    - Locks CRS to the tile's UTM EPSG.
    - Grid is north-up with pixel size = out_res (m).
    - Tile extent uses the Sentinel-2 canonical size: 109.8 km per side at 10 m -> 10,980 px.
      (In general: tile_px = round(109800 / out_res))

    If `require_full_coverage` is True, raises ValueError when the source (in target CRS)
    does not fully cover the canonical tile bounds.

    IMPORTANT: caller must `vrt.close()` when done (this also closes the source).
    """
    with rasterio.open(src_path) as src:
        if src.crs is None:
            raise ValueError("Source has no CRS; cannot align to UTM/MGRS grid.")

        # 1) Determine 100 km tile from center
        tile_id = _tile_id_from_center(src)
        utm_epsg, x_ul, y_ul = _tile_ul_from_mgrs(tile_id)

        # 2) Canonical S2 tile side length (meters) and pixel count at requested resolution
        tile_side_m = 109_800.0  # Sentinel-2 L1C canonical 10m grid = 10,980 px
        xres = yres = float(out_res)
        tile_px = int(round(tile_side_m / xres))
        # Canonical transform anchored to UL
        transform = Affine(xres, 0.0, x_ul, 0.0, -yres, y_ul)
        target_crs = f"EPSG:{utm_epsg}"

        # 3) Coverage check (optional but recommended)
        if require_full_coverage:
            sb = transform_bounds(src.crs, target_crs, *src.bounds, densify_pts=21)
            left, top = x_ul, y_ul
            right = left + tile_px * xres
            bottom = top - tile_px * yres
            tol = xres * 1.01
            if not _bbox_contains(sb, (left, bottom, right, top), tol=tol):
                src.close()
                raise ValueError(
                    f"Source does not fully cover canonical tile {tile_id}.\n"
                    f"  Source bounds (target): {sb}\n"
                    f"  Tile bounds:            {(left, bottom, right, top)}"
                )

        # 4) Build WarpedVRT pinned to canonical grid
        vrt = WarpedVRT(
            src,
            crs=target_crs,
            transform=transform,
            width=tile_px,
            height=tile_px,
            resampling=resampling,
            src_nodata=src_nodata,
            dst_nodata=dst_nodata,
            warp_mem_limit=warp_mem_limit_mb,
        )
        return vrt
