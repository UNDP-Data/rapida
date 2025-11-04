from typing import Dict, Tuple, List
from shapely.geometry import box, Polygon
from shapely.ops import transform as shapely_transform
from pyproj import CRS, Transformer
import math
import logging


logger = logging.getLogger(__name__)
EPS = 1e-12
def _parse_mgrs_100k(grid_id: str):
    s = grid_id.strip().upper()
    *z, band, l1, l2 = s
    zone = int(''.join(z))
    return zone, band, f"{l1}{l2}"

def _normalize_lon(lon: float) -> float:
    x = ((float(lon) + 180.0) % 360.0) - 180.0
    return -180.0 if math.isclose(x, 180.0, abs_tol=1e-12) else x

def utm_grid_zone(longitude: float) -> int:
    lon = _normalize_lon(float(longitude))
    if lon >= 180.0 - 1e-12:
        return 60
    return int(math.floor((lon + 180.0 - EPS) / 6.0)) + 1

def _zones_for_bbox(lon_min: float, lon_max: float) -> List[int]:
    a = _normalize_lon(lon_min)
    b = _normalize_lon(lon_max)
    zones: List[int] = []
    if a <= b:
        z = utm_grid_zone(a)
        zend = utm_grid_zone(math.nextafter(b, float('-inf')))
        zones.append(z)
        while z != zend:
            z = (z % 60) + 1
            zones.append(z)
    else:
        z = utm_grid_zone(a); zend = 60
        zones.append(z)
        while z != zend:
            z = (z % 60) + 1
            zones.append(z)
        z = 1; zend = utm_grid_zone(math.nextafter(b, float('-inf')))
        zones.append(z)
        while z != zend:
            z = (z % 60) + 1
            zones.append(z)
    return zones

def _zone_strip_lonlat(zone: int) -> Polygon:
    lon_min = -180.0 + (zone - 1) * 6.0
    lon_max = lon_min + 6.0
    return box(lon_min, -80.0, lon_max, 84.0)

def _project_geom(geom: Polygon, crs_src: CRS, crs_dst: CRS) -> Polygon:
    tr = Transformer.from_crs(crs_src, crs_dst, always_xy=True)
    return shapely_transform(tr.transform, geom)

def _utm_crs(zone: int, north: bool) -> CRS:
    return CRS.from_epsg((32600 if north else 32700) + zone)

def _wgs84_of_point(e: float, n: float, zone: int, north: bool) -> Tuple[float, float]:
    tr = Transformer.from_crs(_utm_crs(zone, north), CRS.from_epsg(4326), always_xy=True)
    lon, lat = tr.transform(e, n)
    return lon, lat

def _mgrs_100k_key_for_zone(lat: float, lon: float, zone: int) -> str:
    s = latlon2mgrs(lat, lon)          # e.g. "22L YE 12345 67890"
    zband, hundredk, *_ = s.split()
    band = zband[-1]
    return f"{zone}{band}{hundredk}"  # force zone; band+100k from dd2mgrs

def generate_utm_100k_grid(aoi_utm) -> List[Polygon]:
    if aoi_utm.is_empty:
        return []
    xmin, ymin, xmax, ymax = aoi_utm.bounds
    e0 = math.floor(xmin / 100000.0) * 100000.0
    e1 = math.ceil (xmax / 100000.0) * 100000.0
    n0 = math.floor(ymin / 100000.0) * 100000.0
    n1 = math.ceil (ymax / 100000.0) * 100000.0
    out = []
    e = e0
    while e < e1:
        n = n0
        while n < n1:
            cell = box(e, n, e + 100000.0, n + 100000.0)
            if cell.intersects(aoi_utm):
                out.append(cell)
            n += 100000.0
        e += 100000.0
    return out

def mgrs_100k_tiles_for_bbox(lon_min: float, lat_min: float, lon_max: float, lat_max: float) -> Dict[str, Tuple[Polygon, CRS]]:
    """
    Returns { '21LYE': (Polygon_UTM_full_cell, CRS_instance), ... } for all tiles
    intersecting the bbox. Handles:
      • zone seams,
      • equator crossing (north/south CRSs per zone-part).
    """
    aoi_wgs = box(lon_min, lat_min, lon_max, lat_max)
    crs_wgs = CRS.from_epsg(4326)
    out: Dict[str, Tuple[Polygon, CRS]] = {}

    for zone in _zones_for_bbox(lon_min, lon_max):
        strip = _zone_strip_lonlat(zone)
        part_wgs = aoi_wgs.intersection(strip)
        if part_wgs.is_empty:
            continue

        # Split this zone-part by hemisphere (equator at 0° lat)
        north_clip = part_wgs.intersection(box(-180, 0.0, 180, 90))
        south_clip = part_wgs.intersection(box(-180, -90, 180, 0.0))

        for north, hemi_geom in ((True, north_clip), (False, south_clip)):
            if hemi_geom.is_empty:
                continue

            crs_utm = _utm_crs(zone, north)
            part_utm = _project_geom(hemi_geom, crs_wgs, crs_utm)

            # Build 100k grid cells and label using geometric center
            for cell in generate_utm_100k_grid(part_utm):
                inter = cell.intersection(part_utm)
                if inter.is_empty:
                    continue

                # Use the exact geometric center of the 100k cell (always inside the cell)
                xmin, ymin, xmax, ymax = cell.bounds
                cx = 0.5 * (xmin + xmax)
                cy = 0.5 * (ymin + ymax)

                # Convert center to WGS84 to get band/100k via your existing helper
                lon_c, lat_c = _wgs84_of_point(cx, cy, zone, north)

                # Clamp longitude strictly inside THIS zone to avoid seam jitter
                lon_min = -180.0 + (zone - 1) * 6.0
                lon_max = lon_min + 6.0 - 1e-9
                if lon_c < lon_min:
                    lon_c = lon_min + 1e-9
                elif lon_c > lon_max:
                    lon_c = lon_max

                key = _mgrs_100k_key_for_zone(lat_c, lon_c, zone)  # still uses latlon2mgrs inside

                if key not in out:
                    out[key] = (box(*cell.bounds), crs_utm)

    return out


def latlon2mgrs(lat, lon):
    """
    Convert a point in geographic coordinates to MGRS
    :param lat:
    :param lon:
    :return: MGRS str coordinate
    """
    if lat < -80: return 'Too far South'
    if lat > 84:  return 'Too far North'

    # --- CRITICAL FIXES ---
    lon_n = _normalize_lon(lon)
    c = 1 + math.floor((lon_n + 180.0 - EPS) / 6.0)  # west-edge inclusive
    # use lon_n for all subsequent computations (not raw lon)
    e = c * 6 - 183
    k = lat * math.pi / 180.0
    l = lon_n * math.pi / 180.0
    m = e * math.pi / 180.0
    # --- rest unchanged ---
    n = math.cos(k)
    o = 0.006739496819936062 * (n ** 2)
    p = 40680631590769 / (6356752.314 * math.sqrt(1 + o))
    q = math.tan(k)
    r = q * q
    s = (r * r * r) - (q ** 6)
    t = l - m
    u = 1.0 - r + o
    v = 5.0 - r + 9 * o + 4.0 * (o * o)
    w = 5.0 - 18.0 * r + (r * r) + 14.0 * o - 58.0 * r * o
    x = 61.0 - 58.0 * r + (r * r) + 270.0 * o - 330.0 * r * o
    y = 61.0 - 479.0 * r + 179.0 * (r * r) - (r * r * r)
    z = 1385.0 - 3111.0 * r + 543.0 * (r * r) - (r * r * r)
    aa = p * n * t + (p / 6.0 * (n ** 3) * u * (t ** 3)) + (p / 120.0 * (n ** 5) * w * (t ** 5)) + (p / 5040.0 * (n ** 7) * y * (t ** 7))
    ab = 6367449.14570093 * (k - (0.00251882794504 * math.sin(2 * k)) + (0.00000264354112 * math.sin(4 * k)) - (0.00000000345262 * math.sin(6 * k)) + (0.000000000004892 * math.sin(8 * k))) + (q / 2.0 * p * (n ** 2) * (t ** 2)) + (q / 24.0 * p * (n ** 4) * v * (t ** 4)) + (q / 720.0 * p * (n ** 6) * x * (t ** 6)) + (q / 40320.0 * p * (n ** 8) * z * (t ** 8))
    aa = aa * 0.9996 + 500000.0
    ab = ab * 0.9996
    if ab < 0.0:
        ab += 10000000.0
    ad = 'CDEFGHJKLMNPQRSTUVWXX'[math.floor(lat / 8 + 10)]
    ae = math.floor(aa / 100000.0)
    af = ['ABCDEFGH', 'JKLMNPQR', 'STUVWXYZ'][(c - 1) % 3][ae - 1]
    ag = math.floor(ab / 100000.0) % 20
    ah = ['ABCDEFGHJKLMNPQRSTUV', 'FGHJKLMNPQRSTUVABCDE'][(c - 1) % 2][ag]

    def pad(val):
        return str(val).zfill(5)

    aa = pad(math.floor(aa % 100000.0))
    ab = pad(math.floor(ab % 100000.0))

    if c < 10:
        c = str(c).zfill(2)

    return f'{c}{ad} {af}{ah} {aa} {ab}'

def mgrs2latlon(mgrs):
    """
    Convert a MGRS coordinate to geographic coordinates
    :param mgrs:
    :return:
    """
    try:
        b = mgrs.strip().split()
        if not b or len(b) != 4:
            return False, None, None

        c = b[0][0] if len(b[0]) < 3 else b[0][:2]  # zone digits
        d = b[0][1] if len(b[0]) < 3 else b[0][2]  # band letter
        e = (int(c) * 6 - 183) * math.pi / 180.0

        f = ["ABCDEFGH", "JKLMNPQR", "STUVWXYZ"][(int(c) - 1) % 3].find(b[1][0]) + 1
        g = "CDEFGHJKLMNPQRSTUVWXX".find(d)

        h = ["ABCDEFGHJKLMNPQRSTUV", "FGHJKLMNPQRSTUVABCDE"][(int(c) - 1) % 2].find(b[1][1])
        i = [1.1, 2.0, 2.8, 3.7, 4.6, 5.5, 6.4, 7.3, 8.2, 9.1, 0, 0.8, 1.7, 2.6, 3.5, 4.4, 5.3, 6.2, 7.0, 7.9]
        j = [0, 2, 2, 2, 4, 4, 6, 6, 8, 8, 0, 0, 0, 2, 2, 4, 4, 6, 6, 6]
        k = i[g]
        l = j[g] + h / 10
        if l < k:
            l += 2

        m = f * 100000.0 + int(b[2])  # easting
        n = l * 1000000.0 + int(b[3]) # northing
        m -= 500000.0
        if d < 'N':
            n -= 10000000.0

        m /= 0.9996
        n /= 0.9996

        o = n / 6367449.14570093
        p = o + (0.0025188266133249035 * math.sin(2.0 * o)) + (0.0000037009491206268 * math.sin(4.0 * o)) + (0.0000000074477705265 * math.sin(6.0 * o)) + (0.0000000000170359940 * math.sin(8.0 * o))
        q = math.tan(p)
        r = q * q
        s = r * r
        t = math.cos(p)
        u = 0.006739496819936062 * (t ** 2)
        v = 40680631590769 / (6356752.314 * math.sqrt(1 + u))
        w = v
        x = 1.0 / (w * t)
        w *= v
        y = q / (2.0 * w)
        w *= v
        z = 1.0 / (6.0 * w * t)
        w *= v
        aa = q / (24.0 * w)
        w *= v
        ab = 1.0 / (120.0 * w * t)
        w *= v
        ac = q / (720.0 * w)
        w *= v
        ad = 1.0 / (5040.0 * w * t)
        w *= v
        ae = q / (40320.0 * w)

        lat = p + y * (-1.0 - u) * (m ** 2) + aa * (5.0 + 3.0 * r + 6.0 * u - 6.0 * r * u - 3.0 * (u ** 2) - 9.0 * r * (u ** 2)) * (m ** 4) + ac * (-61.0 - 90.0 * r - 45.0 * s - 107.0 * u + 162.0 * r * u) * (m ** 6) + ae * (1385.0 + 3633.0 * r + 4095.0 * s + 1575 * (s * r)) * (m ** 8)
        lng = e + x * m + z * (-1.0 - 2 * r - u) * (m ** 3) + ab * (5.0 + 28.0 * r + 24.0 * s + 6.0 * u + 8.0 * r * u) * (m ** 5) + ad * (-61.0 - 662.0 * r - 1320.0 * s - 720.0 * (s * r)) * (m ** 7)

        lat_deg = lat * 180.0 / math.pi
        lon_deg = _normalize_lon(lng * 180.0 / math.pi)  # normalize output
        return lat_deg, lon_deg

    except Exception as e:
        logger.error(f"Error converting MGRS: {e}")
        return None, None

if __name__ =='__main__':
    # print(mgrs2latlon('02C NS 00000 18414'))
    # print(latlon2mgrs(-80.00000,-171.0))

    lat, lon =  35.49653228, 12.60788626
    lat, lon =  -14, -54
    print(lat, lon)
    mgrs = latlon2mgrs(lat, lon)
    print(mgrs)
    rlan, rlon = mgrs2latlon(mgrs)
    print(rlan, rlon)
    # BRAZIL_BBOX = [-56.0, -15.0, -54.0, -13.0]
    # UGAKEN_BBOX = 33.280908600000004, -1.154692598710874, 36.59833289999998, 2.650255597059965
    # SAHARA = 13, 24, 16, 26
    # NIGERIA_BBOX = [6.0, 7.0, 8.0, 9.0]
    # tiles = mgrs_100k_tiles_for_bbox(*SAHARA)
    # print(len(tiles))
    # for grid, (poly, crs) in tiles.items():
    #     print(grid)
