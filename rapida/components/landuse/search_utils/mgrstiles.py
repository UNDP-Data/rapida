from __future__ import annotations
from math import exp
from dataclasses import dataclass
from typing import Optional
from datetime import datetime, timezone
import pyproj
import requests
from shapely.geometry.polygon import Polygon
from shapely.ops import transform
from pyproj import Transformer


# ---------------------------
# Data & Config
# ---------------------------

@dataclass()
class Candidate:
    id: str
    time_ts: int           # integer timestamp (e.g., Unix seconds)
    ref_ts: int
    cloud_cover: float           # 0..100
    nodata_coverage: float          # 0..100
    grid: str
    assets: Optional[dict] = None
    data_coverage: Optional[float] = None
    tile_data_geometry: Polygon = None
    tile_info: Optional[dict] = None
    tile_geometry: Polygon = None
    mgrs_geometry: Polygon = None
    mgrs_crs: pyproj.CRS = None
    stac_item: dict = None

    @property
    def quality_score(self) -> float:
        """
        Compute a balanced (0–100) quality score combining:
          • temporal proximity (33.3%)
          • low cloud cover (33.3%)
          • data coverage (33.3%)
        All three normalized to [0,1].
        """
        # --- normalize components ---
        # 1) time proximity → 1.0 if exact match, decays exponentially with days
        days_diff = abs(self.time_ts - self.ref_ts) / 86400
        time_factor = exp(-days_diff / 3.0)  # ~3-day half-decay window

        # 2) cloud → 1.0 at 0%, 0.0 at 100%
        cloud_factor = 1.0 - min(max(self.cloud_cover, 0), 100) / 100.0

        # 3) data coverage → 0.0–1.0 directly
        data_factor = min(max(self.data_coverage, 0), 100) / 100.0

        # --- equal weights (each 1/3 of total) ---
        weight = 100 / 3.0
        score = weight * (time_factor + cloud_factor + data_factor)

        return round(score, 3)

    def reproject(self, dst_crs:str=None, geom='tile_geometry'):
        transformer = Transformer.from_crs(self.mgrs_crs, dst_crs, always_xy=True)
        return transform(transformer.transform, getattr(self, geom))

    @property
    def datetime(self):
        return datetime.fromtimestamp(self.time_ts)
    @property
    def date(self):
        return self.datetime.date()
    @property
    def nodata(self):
        return self.assets['red']['raster:bands'][0]['nodata']

    @property
    def dtype(self):
        return self.assets['red']['raster:bands'][0]['data_type']


    def __str__(self):
        #return f"{self.id} {self.date.strftime('%d-%m-%Y')} {self.cloud_cover} {self.nodata_coverage}"
        return self.__repr__()
    def __repr__(self):
        return f"[{self.date.strftime('%Y-%m-%d')} cloud-cover:{self.cloud_cover:02.2f}%, data-coverage:{self.data_coverage}%]"


def _iso_to_ts(iso: str) -> int:
    dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
    return int(dt.replace(tzinfo=timezone.utc).timestamp())

def _cloud_from_props(props: dict) -> float:
    # Try common keys; normalize 0..1 to 0..100 if needed
    for k in ("eo:cloud_cover", "s2:cloud_cover", "cloud_cover"):
        if k in props and props[k] is not None:
            v = float(props[k])
            return v
    return 0.0

def _grid_from_props(props: dict, item_id: str) -> Optional[str]:
    # Prefer STAC property; fallback to ID pattern like 'S2B_21LWD_20250130_0_L1C'
    if "grid:code" in props and props["grid:code"]:
        gc = str(props["grid:code"])
        return gc.split("-")[-1].upper()  # 'MGRS-21LWD' -> '21LWD'
    parts = item_id.split("_")
    if len(parts) >= 3 and len(parts[1]) in (5, 6):  # '21LWD' or '21LWDx'
        return parts[1][:5].upper()
    return None

def get_tileinfo(item):
    try:
        tileinfo_asset = item.assets.get("tileinfo_metadata")
        item_id = item.id
        href = tileinfo_asset.href
    except AttributeError:
        tileinfo_asset = item.get("assets", {}).get("tileinfo_metadata")
        item_id = item.get("id", "")
        href = tileinfo_asset['href']
    if not tileinfo_asset:
        return None
    # print(f"getting tileinfo for {item_id}: {href}")
    try:
        r = requests.get(s3_to_https(href), timeout=30)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"Failed to fetch tileInfo for {item_id}: {e}")
        return None
def midpoint_from_range(range_str: str) -> int:
    """Accepts 'YYYY-MM-DD[/YYYY-MM-DD]' or full ISO8601 bounds; returns Unix ts (UTC)."""
    start_s, end_s = range_str.split("/")
    def parse_bound(s: str, is_end: bool) -> datetime:
        s = s.strip()
        if "T" in s:  # full ISO8601, possibly with Z
            dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
            return dt.astimezone(timezone.utc)
        # date-only -> start-of-day / end-of-day in UTC
        if is_end:
            dt = datetime.fromisoformat(s).replace(hour=23, minute=59, second=59, microsecond=999999, tzinfo=timezone.utc)
        else:
            dt = datetime.fromisoformat(s).replace(hour=0, minute=0, second=0, microsecond=0, tzinfo=timezone.utc)
        return dt
    sdt = parse_bound(start_s, is_end=False)
    edt = parse_bound(end_s, is_end=True)
    return int((sdt.timestamp() + edt.timestamp()) / 2)


def s3_to_https(s3_url: str) -> str:
    """
    Convert an S3-style URL (s3://bucket/path)
    to an HTTPS S3 endpoint (https://bucket.s3.amazonaws.com/path).
    """
    if not s3_url.startswith("s3://"):
        raise ValueError("Not a valid S3 URL")

    # Remove scheme
    path = s3_url[5:]

    # Split into bucket + key
    parts = path.split("/", 1)
    if len(parts) != 2:
        raise ValueError("S3 URL must be in format s3://bucket/key")

    bucket, key = parts
    return f"https://{bucket}.s3.amazonaws.com/{key}"

