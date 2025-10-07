from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Iterable
import math
from datetime import datetime, time, timezone
import mgrs
from rapida.components.landuse.codes import mgrs_100k_squares

# ---------------------------
# Data & Config
# ---------------------------

@dataclass(frozen=True)
class Candidate:
    id: str
    time_ts: int           # integer timestamp (e.g., Unix seconds)
    cloud: float           # 0..100
    nodata: float          # 0..100
    meta: Optional[dict] = None

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
            return v * 100.0 if 0.0 <= v <= 1.0 else v
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

def prune_candidates(tiles: Tiles, cfg: Config) -> Tiles:
    pruned: Tiles = {}
    for tid, cands in tiles.items():
        keep = []
        for c in cands:
            if c.cloud > cfg.max_cloud or c.nodata > cfg.max_nodata: continue
            if _days_from_ts(c.time_ts, cfg.t0_ts) > cfg.time_window_days: continue
            keep.append(c)
        pruned[tid] = keep
    return pruned

def rank_topk(tiles: Tiles, cfg: Config) -> Tiles:
    all_dt, all_cl, all_nd = [], [], []
    for cands in tiles.values():
        for c in cands:
            all_dt.append(_days_from_ts(c.time_ts, cfg.t0_ts))
            all_cl.append(c.cloud); all_nd.append(c.nodata)
    if not all_dt: return {tid: [] for tid in tiles}
    n_dt = _normalize(all_dt); n_cl = _normalize(all_cl); n_nd = _normalize(all_nd)

    def unary(c: Candidate) -> float:
        return (cfg.lambda_T * n_dt[_days_from_ts(c.time_ts, cfg.t0_ts)]
              + cfg.lambda_C * n_cl[c.cloud]
              + cfg.lambda_N * n_nd[c.nodata])

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
            all_cl.append(c.cloud); all_nd.append(c.nodata)
    n_dt = _normalize(all_dt) if all_dt else {}
    n_cl = _normalize(all_cl) if all_cl else {}
    n_nd = _normalize(all_nd) if all_nd else {}

    def unary(c: Candidate) -> float:
        return (cfg.lambda_T * n_dt.get(_days_from_ts(c.time_ts, cfg.t0_ts), 0.0)
              + cfg.lambda_C * n_cl.get(c.cloud, 0.0)
              + cfg.lambda_N * n_nd.get(c.nodata, 0.0))

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
            all_cl.append(c.cloud); all_nd.append(c.nodata)
    n_dt = _normalize(all_dt) if all_dt else {}
    n_cl = _normalize(all_cl) if all_cl else {}
    n_nd = _normalize(all_nd) if all_nd else {}

    cache = {}
    for tid, cands in tiles.items():
        for i, c in enumerate(cands):
            cache[(tid, i)] = (cfg.lambda_T * n_dt.get(_days_from_ts(c.time_ts, cfg.t0_ts), 0.0)
                             + cfg.lambda_C * n_cl.get(c.cloud, 0.0)
                             + cfg.lambda_N * n_nd.get(c.nodata, 0.0))
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

def solve(tiles: Tiles, adj: Adj, cfg: Config) -> Dict[str, Candidate]:
    tiles1 = prune_candidates(tiles, cfg)
    tiles2 = rank_topk(tiles1, cfg)
    labels0 = initial_labels(tiles2, cfg)
    labelsF = refine(labels0, tiles2, adj, cfg)
    return {tid: tiles2[tid][li] for tid, li in labelsF.items() if li >= 0}


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
        datetime="2025-01-01/2025-06-30"
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

def search(client=None, bbox=None, name=None, collection="sentinel-2-l1c", limit=None, datetime_range=None) -> Tiles:
    cache_path = f"/tmp/{name}.json"
    try:
        with open(cache_path) as srcj:
            items = json.load(srcj)  # list of STAC item dicts
    except Exception:
        stac = client.search(
            collections=[collection],
            bbox=bbox,
            max_items=limit,
            datetime=datetime_range
        )
        items = [itm.to_dict() for itm in stac.items()]
        with open(cache_path, "w") as dstj:
            json.dump(items, dstj, indent=2)

    tiles: Tiles = {}
    for it in items:
        props = it.get("properties", {})
        item_id = it.get("id", "")
        grid = _grid_from_props(props, item_id)
        dt_iso = props.get("datetime")
        if not grid or not dt_iso:
            continue  # cannot place this item
        cand = Candidate(
            id=item_id,
            time_ts=_iso_to_ts(dt_iso),
            cloud=_cloud_from_props(props),
            nodata=0.0,  # fill later via GDAL/rasterio
            meta={"hrefs": it.get("assets", {}), "grid": grid}
        )
        tiles.setdefault(grid, []).append(cand)

    # (Optional) deterministic order
    for tid in tiles:
        tiles[tid].sort(key=lambda c: c.time_ts)
    return tiles

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



def utm_lat_band(latitude: float = None) -> str | None:
    """
    Generate UTM MGRS 8-12 degrees band for a specific latitude
    :param latitude:
    :return:
    """
    bands = "CDEFGHJKLMNPQRSTUVWX"
    idx = int((latitude + 80) // 8)
    if latitude >= 72:
        return "X"
    return bands[idx]


def utm_grid_zone(longitude: float = None):
    """
    Generetae a UTM 6 degrees zone fro a given longitude
    :param longitude:
    :return:
    """

    return int(math.floor((longitude + 180) / 6) + 1)





def generate_mgrs_tiles(bbox=None):
    lonmin, latmin, lonmax, latmax = bbox

    # how many zone we have in the bbox
    start_zone = utm_grid_zone(lonmin)
    end_zone = utm_grid_zone(lonmax)
    start_band= utm_lat_band(latmin)
    end_band= utm_lat_band(latmax)
    zones = sorted(set([start_zone, end_zone]))
    bands = sorted(set([start_band, end_band]))
    i = 0
    for zone in zones:
        for band in bands:
            for grid in mgrs_100k_squares(zone, band):
                tile = f'{zone}{band}{grid}'
                i+=1
                print(i, tile)





if __name__ == '__main__':

    import pystac_client
    import json
    import os

    CATALOG_URL = "https://earth-search.aws.element84.com/v1"

    BRAZIL_BBOX = [-56.0, -15.0, -54.0, -13.0]
    NIGERIA_BBOX = [6.0, 7.0, 8.0, 9.0]
    CHINA_BBOX = [100.0, 30.0, 110.0, 40.0]

    bbox = BRAZIL_BBOX
    generate_mgrs_tiles(bbox=bbox)
    # datetime_range="2025-01-01/2025-06-30"
    # midpoint_unix_ts = midpoint_from_range(datetime_range)
    # client = pystac_client.Client.open(CATALOG_URL)
    #
    # tiles = search(client=client,bbox=bbox, name='china', datetime_range=datetime_range)
    #
    # grid = grid_from_mgrs_100k(codes=list(tiles.keys()))
    # adj = build_queen_adjacency_ragged(grid_ids=grid)
    # cfg = Config(t0_ts=midpoint_unix_ts, max_cloud=37.0, trunc_tau_days=60)
    # chosen = solve(tiles, adj, cfg)
    # print(len(chosen))
    # for e, v in chosen.items():
    #     print(v)