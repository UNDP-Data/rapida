from __future__ import annotations
from math import exp
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Iterable
from datetime import datetime, timezone
import mgrs
import pyproj
import requests
from shapely import transform
from shapely.geometry.base import BaseGeometry
from tqdm import tqdm


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
    tile_data_geometry: BaseGeometry = None
    tile_info: Optional[dict] = None
    tile_geometry: BaseGeometry = None
    mgrs_geometry: BaseGeometry = None
    mgrs_crs: pyproj.CRS = None
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


    @property
    def datetime(self):
        return datetime.fromtimestamp(self.time_ts)
    @property
    def date(self):
        return self.datetime.date()


    def __str__(self):
        #return f"{self.id} {self.date.strftime('%d-%m-%Y')} {self.cloud_cover} {self.nodata_coverage}"
        return self.__repr__()
    def __repr__(self):
        return f"[{self.date.strftime('%d-%m-%Y')} cloud-cover:{self.cloud_cover:02.2f}%, data-coverage:{self.data_coverage}%]"

@dataclass
class Config:
    t0_ts: int
    max_cloud: float = 7.0
    max_nodata: float = 100.0
    time_window_days: int = 75
    trunc_tau_days: int = 60
    k_per_tile: int = 8
    lambda_T: float = 1.0
    lambda_C: float = 1.2
    lambda_N: float = 1.5
    lambda_S: float = 0.8
    max_iters: int = 10

Tiles = Dict[str, List[Candidate]]
Adj = Dict[str, List[str]]

# ---------------------------
# Helpers (≤3 args)
# ---------------------------

_SEC_PER_DAY = 86400.0
_m = mgrs.MGRS()

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

def _days_from_ts(ts: int, ref: int) -> float:
    return abs(ts - ref) / _SEC_PER_DAY

def _normalize(vals: List[float]) -> Dict[float, float]:
    vmin, vmax = min(vals), max(vals)
    if vmax == vmin: return {v: 0.0 for v in vals}
    span = vmax - vmin
    return {v: (v - vmin) / span for v in vals}

def build_queen_adjacency_ragged(grid_ids: List[List[str]]) -> Dict[str, List[str]]:
    """Queen adjacency for ragged grids (rows may have different lengths)."""
    import itertools
    adj: Dict[str, List[str]] = {}
    H = len(grid_ids)

    # register nodes
    for r in range(H):
        for c in range(len(grid_ids[r])):
            a = grid_ids[r][c]
            adj.setdefault(a, [])

    # connect neighbors
    seen = set()
    for r in range(H):
        for c in range(len(grid_ids[r])):
            a = grid_ids[r][c]
            for dr, dc in itertools.product([-1,0,1], [-1,0,1]):
                if dr == 0 and dc == 0:
                    continue
                rr, cc = r+dr, c+dc
                if 0 <= rr < H and 0 <= cc < len(grid_ids[rr]):
                    b = grid_ids[rr][cc]
                    if a == b:
                        continue
                    if (a, b) in seen or (b, a) in seen:
                        continue
                    adj[a].append(b)
                    adj[b].append(a)
                    seen.add((a, b))
    # optional: sort neighbors for determinism
    for k in adj:
        adj[k] = sorted(set(adj[k]))
    return adj
# ---------------------------
# Prune & Rank (≤3 args)
# ---------------------------

from shapely.geometry import shape


def find_corresponding_union_v1(best_cand, candidates, threshold=1):
    """
    Find a set of candidates that, when unioned with best_cand,
    cover at least a ` threshold ` fraction of the full tile.
    """
    full_tile_geom = shape(best_cand.tile_info["tileGeometry"])
    current_union = shape(best_cand.tile_info["tileDataGeometry"])

    selected = [best_cand]
    coverage_ratio = current_union.area / full_tile_geom.area
    remaining = [c for c in candidates if c != best_cand]
    while coverage_ratio < threshold and remaining:
        best_gain = 0
        best_c = None
        for c in remaining:
            c_geom = shape(c.tile_info["tileDataGeometry"])
            new_union = current_union.union(c_geom)
            new_ratio = new_union.area / full_tile_geom.area
            gain = new_ratio - coverage_ratio
            if gain > best_gain:
                best_gain = gain
                best_c = c
        if not best_c:
            break  # no improvement possible
        selected.append(best_c)
        current_union = current_union.union(shape(best_c.tile_info["tileDataGeometry"]))
        coverage_ratio = current_union.area / full_tile_geom.area
        remaining.remove(best_c)
    return selected


def find_corresponding_union(best_cand, candidates, threshold=1.0, tolerance=0.0):
    """
    Selects a subset of candidates that, when unioned with the best_cand,
    cover at least the given threshold of the full tile area.

    The algorithm:
    - Starts from best_cand and incrementally adds candidates that maximize coverage gain.
    - Breaks ties by preferring candidates with lower cloud values.
    - Drops candidates that add no new uncovered area.
    - After reaching a threshold, prunes the selection to minimize cloud while maintaining coverage.

    Parameters
    ----------
    best_cand : Candidate
        The starting candidate, typically the most preferred one.
    candidates : list of Candidate
        Other candidate tiles to consider for unioning.
    threshold : float, default=1.0
        Fraction of the full tile area that must be covered.
    tolerance : float, default=0.0
        Allowed numerical tolerance when comparing coverage ratios.

    Returns
    -------
    list of Candidate
        The chosen subset of candidates that meet the coverage requirement.
    """
    full_tile_geom = shape(best_cand.tile_info["tileGeometry"])
    full_area = full_tile_geom.area
    if full_area == 0:
        return [best_cand]

    current_union = shape(best_cand.tile_info["tileDataGeometry"])
    coverage = current_union.area / full_area

    selected = [best_cand]
    remaining = [(c, shape(c.tile_info["tileDataGeometry"])) for c in candidates if c != best_cand]

    while coverage + tolerance < threshold and remaining:
        best_idx = None
        best_gain = 0.0

        for i, (c, geom) in enumerate(remaining):
            if geom.difference(current_union).is_empty:
                continue
            new_area = current_union.union(geom).area
            gain = new_area / full_area - coverage
            if gain > best_gain:
                best_gain = gain
                best_idx = i
            elif gain == best_gain and best_idx is not None:
                if c.cloud_cover < remaining[best_idx][0].cloud_cover:
                    best_idx = i
            elif gain == best_gain and best_idx is None:
                best_idx = i

        if best_idx is None:
            break
        chosen_c, chosen_geom = remaining.pop(best_idx)
        selected.append(chosen_c)
        current_union = current_union.union(chosen_geom)
        coverage = current_union.area / full_area
        remaining = [(cc, gg) for cc, gg in remaining if not gg.difference(current_union).is_empty]

    if coverage + tolerance < threshold:
        return selected

    sorted_selected = sorted(selected, key=lambda c: c.cloud_cover)
    pruned = []
    pruned_union = None
    for c in sorted_selected:
        geom = shape(c.tile_info["tileDataGeometry"])
        if pruned_union is None:
            pruned_union = geom
            pruned.append(c)
        else:
            if not geom.difference(pruned_union).is_empty:
                pruned_union = pruned_union.union(geom)
                pruned.append(c)
        if pruned_union.area / full_area >= threshold - tolerance:
            break

    if pruned_union is not None and pruned_union.area / full_area >= threshold - tolerance:
        return pruned
    return selected




def prune_candidates(tiles: Tiles, cfg: Config) -> Tiles:
    pruned: Tiles = {}
    for tid, cands in tiles.items():
        # Filter by time window
        valid = [
            c for c in cands
            if _days_from_ts(c.time_ts, cfg.t0_ts) <= cfg.time_window_days
        ]
        if not valid:
            pruned[tid] = []
            continue

        strict = [
            c for c in cands
            if c.cloud_cover <= cfg.max_cloud and c.nodata_coverage <= cfg.max_nodata
        ]
        if strict:
            pruned[tid] = strict
        else:

            # This will select the best tile based on cloud cover and in case of equality of any 2 tiles, break the tie based on percentage of nodata
            best = min(cands, key=lambda c: (c.cloud_cover, c.nodata_coverage))

            corresponding_union = find_corresponding_union(best, valid)
            if corresponding_union:
                group_id = tid + "_group"
                for c in corresponding_union:
                    c.union_group = group_id
            pruned[tid] = []
            pruned[tid].extend(corresponding_union) if corresponding_union else pruned[tid].append(best)
    return pruned



def rank_topk(tiles: Tiles, cfg: Config) -> Tiles:

    all_dt, all_cl, all_nd = [], [], []
    for cands in tiles.values():
        for c in cands:
            all_dt.append(_days_from_ts(c.time_ts, cfg.t0_ts))
            all_cl.append(c.cloud_cover); all_nd.append(c.nodata_coverage)
    if not all_dt: return {tid: [] for tid in tiles}
    n_dt = _normalize(all_dt); n_cl = _normalize(all_cl); n_nd = _normalize(all_nd)

    def unary(c: Candidate) -> float:
        return (cfg.lambda_T * n_dt[_days_from_ts(c.time_ts, cfg.t0_ts)]
                + cfg.lambda_C * n_cl[c.cloud_cover]
                + cfg.lambda_N * n_nd[c.nodata_coverage])

    reduced: Tiles = {}
    for tid, cands in tiles.items():
        scored = sorted(cands, key=unary)
        reduced[tid] = scored[:cfg.k_per_tile] if len(scored) > cfg.k_per_tile else scored
    return reduced

def initial_labels(tiles: Tiles, cfg: Config) -> Dict[str, int]:
    all_dt, all_cl, all_nd = [], [], []
    for cands in tiles.values():
        for c in cands:
            all_dt.append(_days_from_ts(c.time_ts, cfg.t0_ts))
            all_cl.append(c.cloud_cover); all_nd.append(c.nodata_coverage)
    n_dt = _normalize(all_dt) if all_dt else {}
    n_cl = _normalize(all_cl) if all_cl else {}
    n_nd = _normalize(all_nd) if all_nd else {}

    def unary(c: Candidate) -> float:
        return (cfg.lambda_T * n_dt.get(_days_from_ts(c.time_ts, cfg.t0_ts), 0.0)
                + cfg.lambda_C * n_cl.get(c.cloud_cover, 0.0)
                + cfg.lambda_N * n_nd.get(c.nodata_coverage, 0.0))

    labels = {}
    for tid, cands in tiles.items():
        if not cands: labels[tid] = -1; continue
        labels[tid] = min(range(len(cands)), key=lambda i: unary(cands[i]))
    return labels

# ---------------------------
# Energy & Refine (≤3 args)
# ---------------------------

def _pair_penalty_days(d_days: float, tau_days: int) -> float:
    d = min(d_days, float(tau_days))
    return d / float(tau_days) if tau_days > 0 else 0.0

def _unary_cache(tiles: Tiles, cfg: Config) -> Dict[Tuple[str,int], float]:
    all_dt, all_cl, all_nd = [], [], []
    for cands in tiles.values():
        for c in cands:
            all_dt.append(_days_from_ts(c.time_ts, cfg.t0_ts))
            all_cl.append(c.cloud_cover); all_nd.append(c.nodata_coverage)
    n_dt = _normalize(all_dt) if all_dt else {}
    n_cl = _normalize(all_cl) if all_cl else {}
    n_nd = _normalize(all_nd) if all_nd else {}

    cache = {}
    for tid, cands in tiles.items():
        for i, c in enumerate(cands):
            cache[(tid, i)] = (cfg.lambda_T * n_dt.get(_days_from_ts(c.time_ts, cfg.t0_ts), 0.0)
                               + cfg.lambda_C * n_cl.get(c.cloud_cover, 0.0)
                               + cfg.lambda_N * n_nd.get(c.nodata_coverage, 0.0))
    return cache

def objective(labels: Dict[str,int], tiles: Tiles, adj: Adj, cfg: Config) -> float:
    U = 0.0; cache = _unary_cache(tiles, cfg)
    for tid, li in labels.items():
        if li >= 0: U += cache[(tid, li)]
    P = 0.0; seen = set()
    for a, nbrs in adj.items():
        la = labels.get(a, -1)
        if la < 0: continue
        for b in nbrs:
            if (b, a) in seen: continue
            lb = labels.get(b, -1)
            if lb < 0: continue
            da = _days_from_ts(tiles[a][la].time_ts, tiles[b][lb].time_ts)
            P += _pair_penalty_days(da, cfg.trunc_tau_days)
            seen.add((a, b))
    return U + cfg.lambda_S * P

def refine(labels: Dict[str,int], tiles: Tiles, adj: Adj, cfg: Config) -> Dict[str,int]:
    cache = _unary_cache(tiles, cfg)
    cur = labels.copy()

    def local_delta(tid: str, new_i: int) -> float:
        old_i = cur[tid]
        if old_i == new_i: return 0.0
        dU = cache.get((tid, new_i), math.inf) - (cache.get((tid, old_i), 0.0) if old_i >= 0 else 0.0)
        dP = 0.0
        t_new = tiles[tid][new_i].time_ts if new_i >= 0 else None
        t_old = tiles[tid][old_i].time_ts if old_i >= 0 else None
        for nb in adj.get(tid, []):
            nb_i = cur.get(nb, -1)
            if nb_i < 0: continue
            t_nb = tiles[nb][nb_i].time_ts
            if t_old is not None:
                dP -= _pair_penalty_days(_days_from_ts(t_old, t_nb), cfg.trunc_tau_days)
            if t_new is not None:
                dP += _pair_penalty_days(_days_from_ts(t_new, t_nb), cfg.trunc_tau_days)
        return dU + cfg.lambda_S * dP

    for _ in range(cfg.max_iters):
        improved = False
        for tid, cands in tiles.items():
            if not cands: continue
            best_i = cur[tid]; best_delta = 0.0
            for i in range(len(cands)):
                dE = local_delta(tid, i)
                if dE < best_delta:
                    best_delta = dE; best_i = i
            if best_i != cur[tid]:
                cur[tid] = best_i; improved = True
        if not improved: break
    return cur

# ---------------------------
# Pipeline (≤3 args)
# ---------------------------

def solve(tiles: Tiles, adj: Adj, cfg: Config) -> Dict[str, List[Candidate]]:
    tiles1 = prune_candidates(tiles, cfg)
    tiles2 = rank_topk(tiles1, cfg)
    labels0 = initial_labels(tiles2, cfg)
    labelsF = refine(labels0, tiles2, adj, cfg)
    selected = {}
    for tid, li in labelsF.items():
        if li < 0:
            continue
        cand = tiles2[tid][li]
        if cand.union_group is not None:
            selected[tid] = [
                c for c in tiles2[tid] if c.union_group == cand.union_group
            ]
        else:
            selected[tid] = [cand]

    # return {tid: tiles2[tid][li] for tid, li in labelsF.items() if li >= 0}
    return selected


def _mgrs100k_center(code: str) -> Tuple[float, float]:
    """Center of a 100 km MGRS tile via '50000 50000' trick (lat, lon)."""
    center_1m = f"{code[:5]}5000050000"  # ensure it's exactly the 100k code + center offset
    lat, lon = _m.toLatLon(center_1m)
    return lat, lon

def grid_from_mgrs_100k(codes: Iterable[str], tol_deg: float = 0.3) -> List[List[str]]:
    """
    Order MGRS-100k codes into a 2D grid.
    - Rows are grouped by latitude (north→south).
    - Within each row, tiles are sorted by longitude (west→east).
    - tol_deg controls how tightly lats must match to be considered the same row.
    Returns a ragged list of rows (use your ragged-safe adjacency).
    """
    pts: Dict[str, Tuple[float, float]] = {c: _mgrs100k_center(c) for c in codes}
    # primary sort: lat DESC (north→south); secondary: lon ASC (west→east)
    ordered = sorted(codes, key=lambda c: (-pts[c][0], pts[c][1]))

    rows: List[List[str]] = []
    cur_row: List[str] = []
    cur_lat_ref: float = None

    for c in ordered:
        lat, lon = pts[c]
        if cur_lat_ref is None:
            cur_lat_ref = lat
            cur_row = [c]
            continue
        if abs(lat - cur_lat_ref) <= tol_deg:
            cur_row.append(c)
        else:
            # finalize previous row: sort west→east just to be safe
            cur_row.sort(key=lambda cc: pts[cc][1])
            rows.append(cur_row)
            # start new row
            cur_lat_ref = lat
            cur_row = [c]

    if cur_row:
        cur_row.sort(key=lambda cc: pts[cc][1])
        rows.append(cur_row)

    return rows



def items_to_featurecollection(items):
    """Convert list of pystac Items to GeoJSON FeatureCollection"""
    return {
        "type": "FeatureCollection",
        "features": [item.to_dict() for item in items]
    }

def search_and_save( client=None, bbox=None, name=None, collection="sentinel-2-l1c", limit=None):
    search = client.search(
        collections=[collection],
        bbox=bbox,
        max_items=limit,
        datetime="2024-01-01/2024-06-30",
    )
    output_dir = "items"
    items = list(search.get_items())
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    fc = items_to_featurecollection(items)
    outpath = os.path.join(output_dir, f"{name}_items.geojson")

    with open(outpath, "w") as f:
        json.dump(fc, f, indent=2)

    print(f"Saved {len(items)} items to {outpath}")


def search_and_save_2( client=None, bbox=None, name=None, collection="sentinel-2-l1c", limit=None):
    search = client.search(
        collections=[collection],
        max_items=limit,
        datetime="2025-01-01/2025-06-30",
        query={
            "grid:code": {"eq": "MGRS-21LXD"}
        }
    )
    for item in search.items():
        tile_info = get_tileinfo(item)
        if tile_info['dataCoveragePercentage'] > 95 and tile_info['cloudyPixelPercentage'] < 10:
            print(item.id, tile_info['dataCoveragePercentage'], tile_info['cloudyPixelPercentage'])




def mgrs_tile_polygon(mgrs_code):
    """Return shapely Polygon for a Sentinel-2 tile (100x100 km)."""
    ll_lat, ll_lon = _m.toLatLon(mgrs_code)

    zone = int(mgrs_code[:2])
    band = mgrs_code[2]

    utm_crs = pyproj.CRS.from_dict({"proj": "utm", "zone": zone, "south": band < "N"})
    wgs84 = pyproj.CRS.from_epsg(4326)
    project_to_wgs84 = pyproj.Transformer.from_crs(utm_crs, wgs84, always_xy=True).transform
    x0, y0 = project_to_wgs84(ll_lon, ll_lat)  # lower-left

    square = Polygon([
        (x0, y0),
        (x0 + 100000, y0),
        (x0 + 100000, y0 + 100000),
        (x0, y0 + 100000),
        (x0, y0)
    ])
    return transform(project_to_wgs84, square)


import math
from shapely.geometry import Polygon
import mgrs  # pip install mgrs

_m = mgrs.MGRS()


def covering_mgrs_tiles(bounds, sampling_ratio=10, out_geojson="polygon.geojson"):
    """
    Return all Sentinel-2 MGRS tiles intersecting bounds polygon.

    Parameters
    ----------
    bounds : tuple
        (minx, miny, maxx, maxy) in EPSG:4326 degrees
    sampling_ratio : int
        How many samples per 100km tile side (default=10 -> ~10 km spacing)
    out_geojson : str
        Path to save the bounding polygon as GeoJSON
    """
    minx, miny, maxx, maxy = bounds
    candidate_tiles = set()
    tile_size_m = 100000
    target_resolution_m = tile_size_m / sampling_ratio
    centroid_lat = (miny + maxy) / 2.0
    meters_per_degree_lat = 111320.0
    meters_per_degree_lon = 111320.0 * math.cos(math.radians(centroid_lat))
    step_lat = target_resolution_m / meters_per_degree_lat
    step_lon = target_resolution_m / meters_per_degree_lon
    # polygon = Polygon([
    #     (minx, miny),
    #     (minx, maxy),
    #     (maxx, maxy),
    #     (maxx, miny)
    # ])
    x = minx
    while x <= maxx + step_lon:
        y = miny
        while y <= maxy + step_lat:
            mgrs_code = _m.toMGRS(y, x, MGRSPrecision=0)
            candidate_tiles.add(mgrs_code)
            y += step_lat
        x += step_lon
    # gdf = gpd.GeoDataFrame(geometry=[polygon], crs="EPSG:4326")
    # gdf.to_file(driver="GeoJSON", filename=out_geojson)
    return sorted(candidate_tiles)

    # for code in candidate_tiles:
    #     try:
    #         tile_poly = mgrs_tile_polygon(code)
    #         if polygon.intersects(tile_poly):
    #             tiles.append(code)
    #     except Exception:
    #         raise
    # return tiles

def dynamic_search(
        client=None,
        bbox=None,
        name=None,
        collection="sentinel-2-l1c",
        limit=None,
        datetime_range=None
):
    # using the MGRS grid to get an exhaustive list of all the tiles I should cover based on the bbox given

    search = client.search(
        collections=[collection],
        bbox=bbox,
        max_items=limit,
        datetime=datetime_range
    )
    items = list(search.get_items())
    return items




def search(client=None, bbox=None, name=None, collection="sentinel-2-l1c", limit=None, datetime_range=None, unique_grids=None) -> Tiles:
    cache_path = f"/tmp/{name}.json"
    items = []
    try:
        with open(cache_path) as srcj:
            items = json.load(srcj)
    except Exception:
        stac = client.search(
            collections=[collection],
            bbox=bbox,
            max_items=limit,
            datetime=datetime_range,
        )
        items = [itm.to_dict() for itm in stac.items()]

        with open(cache_path, "w") as dstj:
            json.dump(items, dstj, indent=2)

    # tiles: Tiles = {}
    def process_item(it, unique_grids):
        props = it.get("properties", {})
        item_id = it.get("id", "")
        grid = _grid_from_props(props, item_id)

        if grid not in unique_grids:
            return None
        tile_info = get_tileinfo(it)
        dt_iso = props.get("datetime")
        if not grid or not dt_iso:
            return None
        cand = Candidate(
            id=item_id,
            time_ts=_iso_to_ts(dt_iso),
            cloud_cover=_cloud_from_props(props),
            assets={"hrefs": it.get("assets", {}), "grid": grid},
            nodata_coverage=100 - tile_info["dataCoveragePercentage"],
            tile_info=tile_info,
        )
        return grid, cand

    tiles = defaultdict(list)

    with ThreadPoolExecutor(max_workers=10) as executor:
        # futures = [executor.submit(process_item, it, unique_grids) for it in items if it['properties'].get('grid:code') == "MGRS-21LXD"]
        futures = [executor.submit(process_item, it, unique_grids) for it in items]
        for future in as_completed(futures):
            result = future.result()
            if result:
                grid, cand = result
                tiles[grid].append(cand)
    # (Optional) deterministic order
    for tid in tiles:
        tiles[tid].sort(key=lambda c: c.time_ts)
    return tiles



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

def search_items(client=None, bbox=None, name=None, collection="sentinel-2-l1c", limit=None, datetime_range=None):
    n = f'/tmp/{name}.json'
    try:
        with open(n) as srcj:
            items = json.load(srcj)
    except Exception as e:

        search = client.search(
            collections=[collection],
            bbox=bbox,
            max_items=limit,
            datetime=datetime_range
        )
        items = [itm.to_dict() for itm in search.items()]
        with open(n, 'w') as dstj:
            json.dump(items,dstj, indent=4)
    grids = set([itm['properties']['grid:code'].split('-')[-1] for itm in items])
    tiles = {}
    #print(json.dumps(items[0], indent=4))
    for grid in grids:
        tiles[grid] = [e['id'] for e in items if  grid in e['id']]
    return tiles

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

import os

def download_item(cand: Candidate) -> str | None:
    # for c in cands:
    print(f"Downloading {cand.id}")
    assets = cand.assets

    blue_link = assets.get("blue").get("href")
    if not blue_link:
        return None

    outpath = f"/tmp/{cand.id}_b3.tif"
    if os.path.exists(outpath):
        print(f"Already downloaded {outpath}")
        return outpath

    try:
        href = s3_to_https(blue_link)
        print(href)
        with requests.get(href, stream=True, timeout=30) as r:
            r.raise_for_status()
            total_size = int(r.headers.get("content-length", 0))
            block_size = 1024  # 1 KB
            # progress = tqdm(total=total_size, unit="B", unit_scale=True, desc=id)

            with open(outpath, "wb") as f:
                for chunk in r.iter_content(chunk_size=block_size):
                    if chunk:
                        f.write(chunk)
                        # progress.update(len(chunk))
            # progress.close()
        print(f"Downloaded {outpath}")
        return outpath

    except Exception as e:
        print(f"Error downloading {id} {e}")
        raise


if __name__ == '__main__':

    import pystac_client
    import json
    import os
    from rapida.components.landuse.search_utils.zones import generate_mgrs_tiles
    CATALOG_URL = "https://earth-search.aws.element84.com/v1"

    BRAZIL_BBOX = [-56.0, -15.0, -54.0, -13.0]
    NIGERIA_BBOX = [6.0, 7.0, 8.0, 9.0]
    CHINA_BBOX = [100.0, 30.0, 110.0, 40.0]
    TCHAD_BBOX = [15.0, 12.0, 20.0, 18.0]

    bbox = BRAZIL_BBOX
    # tiles = covering_mgrs_tiles(bbox)
    # # print(tiles)
    datetime_range="2024-01-01/2024-06-30"
    midpoint_unix_ts = midpoint_from_range(datetime_range)
    #
    client = pystac_client.Client.open(CATALOG_URL)

    # search_and_save_2(client=client, bbox=BRAZIL_BBOX)
    unique_grids = generate_mgrs_tiles(bbox=bbox)

    tiles = search(client=client,bbox=bbox, name='brazil', datetime_range=datetime_range, unique_grids=unique_grids)

    # print(tiles)

    grid = grid_from_mgrs_100k(codes=list(tiles.keys()))

    adj = build_queen_adjacency_ragged(grid_ids=grid)

    cfg = Config(t0_ts=midpoint_unix_ts, max_cloud=5.0, trunc_tau_days=60, max_nodata=0, lambda_N=1.7, lambda_C=1)

    chosen = solve(tiles, adj, cfg)
    for id, cands in chosen.items():
        # print(id, len(cands))
        for cand in cands:
            download_item(cand)
