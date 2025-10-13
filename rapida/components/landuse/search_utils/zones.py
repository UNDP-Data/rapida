import math
from typing import List, Tuple
from math import floor, ceil

import geopandas as gpd
from shapely.geometry import Polygon, box
from pyproj import CRS, Transformer
import mgrs as mgrs_lib
_mgrs = mgrs_lib.MGRS()

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



# Latitude bands (MGRS/UTM), degrees. Note: X band is 72–84 (12° tall).
_BAND_TO_LAT = {
    "C": (-80, -72), "D": (-72, -64), "E": (-64, -56), "F": (-56, -48),
    "G": (-48, -40), "H": (-40, -32), "J": (-32, -24), "K": (-24, -16),
    "L": (-16,  -8), "M": ( -8,   0), "N": (  0,   8), "P": (  8,  16),
    "Q": ( 16,  24), "R": ( 24,  32), "S": ( 32,  40), "T": ( 40,  48),
    "U": ( 48,  56), "V": ( 56,  64), "W": ( 64,  72), "X": ( 72,  84),
}

def _zone_lon_bounds(zone: int) -> Tuple[float, float]:
    """Return (min_lon, max_lon) for a UTM zone (degrees)."""
    min_lon = -180.0 + 6.0 * (zone - 1)
    return min_lon, min_lon + 6.0

def _utm_crs_for(zone: int, band: str) -> CRS:
    """Return appropriate UTM CRS (EPSG:326xx north, 327xx south) for the band."""
    # Bands N–X are northern hemisphere; C–M are southern.
    northern = band.upper() >= "N"
    epsg = 32600 + zone if northern else 32700 + zone
    return CRS.from_epsg(epsg)

def _project_polygon_lonlat_to_utm(poly_ll: Polygon, crs_utm: CRS) -> Polygon:
    transformer = Transformer.from_crs("EPSG:4326", crs_utm, always_xy=True)
    xys = [transformer.transform(x, y) for (x, y) in poly_ll.exterior.coords]
    return Polygon(xys)



def _parse_mgrs_100k(grid_id: str):
    s = grid_id.strip().upper()
    *z, band, l1, l2 = s
    zone = int(''.join(z))
    return zone, band, f"{l1}{l2}"

def utm_bounds(grid_id: str):
    """
    Exact 100 km square in UTM (integer coords).
    """
    zone, band, letters = _parse_mgrs_100k(grid_id)
    crs_utm = _utm_crs_for(zone, band)

    # Get a point INSIDE the square (center), transform to UTM
    lat_c, lon_c = _mgrs.toLatLon(f"{zone}{band}{letters}5000050000")  # 50km,50km center
    to_utm = Transformer.from_crs("EPSG:4326", crs_utm, always_xy=True)
    e_c, n_c = to_utm.transform(lon_c, lat_c)

    # Snap the center to the SW corner on a 100km grid
    e0 = math.floor(e_c / 100000.0) * 100000
    n0 = math.floor(n_c / 100000.0) * 100000

    poly_utm = box(e0, n0, e0 + 100000, n0 + 100000)
    return poly_utm, crs_utm

def wgs84_bounds(grid_id: str):
    """
    Return WGS84 (lon/lat) polygon for the same square.
    """
    poly_utm, crs_utm = utm_bounds(grid_id)
    to_ll = Transformer.from_crs(crs_utm, "EPSG:4326", always_xy=True)
    ring_ll = [to_ll.transform(x, y) for x, y in poly_utm.exterior.coords]
    return Polygon(ring_ll)

def mgrs_100k_squares(zone: int, band: str, filter_polygon) -> List[str]:
    """
    Return the list of MGRS 100 km two-letter grid IDs (e.g., 'YE','ZE',...)
    that exist within the given UTM zone and latitude band.

    Args:
        zone: 1..60
        band: one of C..X excluding I, O

    Returns:
        Sorted list of unique two-letter 100 km IDs.
    """
    band = band.upper()
    if zone < 1 or zone > 60:
        raise ValueError("zone must be in 1..60")
    if band not in _BAND_TO_LAT or band in ("I", "O"):
        raise ValueError("band must be C..X excluding I and O")

    # 1) Build the zone-band rectangle in lon/lat
    min_lon, max_lon = _zone_lon_bounds(zone)
    min_lat, max_lat = _BAND_TO_LAT[band]
    # UTM is defined between ~[-80, 84] anyway; the band table respects that.
    # Build a rectangle polygon (lon/lat, CCW)
    rect_ll = Polygon([
        (min_lon, min_lat),
        (max_lon, min_lat),
        (max_lon, max_lat),
        (min_lon, max_lat),
        (min_lon, min_lat),
    ])


    intersection_ll = filter_polygon.intersection(rect_ll)
    if not intersection_ll.geom_type in ("Polygon", "MultiPolygon"):
        return []


    # 2) Project zone-band rect to UTM
    crs_utm = _utm_crs_for(zone, band)
    rect_utm = _project_polygon_lonlat_to_utm(rect_ll, crs_utm)
    filter_polygon_intersection_utm = _project_polygon_lonlat_to_utm(intersection_ll, crs_utm)
    # 3) Create a 100 km grid covering the UTM bbox; only keep cells that intersect rect_utm
    minx, miny, maxx, maxy = rect_utm.bounds
    # snap to 100 km lines
    km = 100_000.0
    start_x = floor(minx / km) * km
    start_y = floor(miny / km) * km
    end_x   = ceil(maxx / km) * km
    end_y   = ceil(maxy / km) * km

    transformer_to_ll = Transformer.from_crs(crs_utm, "EPSG:4326", always_xy=True)

    ids = list()
    # polygons = list()
    # Iterate grid cells; intersect with rect_utm; sample centroid → MGRS; extract 100k letters
    y = start_y
    while y < end_y:
        x = start_x
        while x < end_x:
            cell = box(x, y, x + km, y + km)
            if cell.intersects(rect_utm) and cell.intersects(filter_polygon_intersection_utm):
                c = cell.intersection(rect_utm).centroid  # robust even if clipped at edges
                lon, lat = transformer_to_ll.transform(c.x, c.y)
                # MGRS with 0 precision returns e.g. '21LYE'
                mgrs_full = _mgrs.toMGRS(lat, lon, MGRSPrecision=0)
                # last two letters are the 100 km grid ID
                square = mgrs_full[-2:]
                ids.append((square, cell))
                # polygons.append(rect_utm)
            x += km
        y += km

    return ids



def full_mgrs_100k_squares(zone: int, band: str, filter = None) -> List[str]:
    """
    Return the list of MGRS 100 km two-letter grid IDs (e.g., 'YE','ZE',...)
    that exist within the given UTM zone and latitude band.

    Args:
        zone: 1..60
        band: one of C..X excluding I, O

    Returns:
        Sorted list of unique two-letter 100 km IDs.
    """
    band = band.upper()
    if zone < 1 or zone > 60:
        raise ValueError("zone must be in 1..60")
    if band not in _BAND_TO_LAT or band in ("I", "O"):
        raise ValueError("band must be C..X excluding I and O")

    # 1) Build the zone-band rectangle in lon/lat
    min_lon, max_lon = _zone_lon_bounds(zone)
    min_lat, max_lat = _BAND_TO_LAT[band]
    # UTM is defined between ~[-80, 84] anyway; the band table respects that.
    # Build a rectangle polygon (lon/lat, CCW)
    rect_ll = Polygon([
        (min_lon, min_lat),
        (max_lon, min_lat),
        (max_lon, max_lat),
        (min_lon, max_lat),
        (min_lon, min_lat),
    ])


    # 2) Project zone-band rect to UTM
    crs_utm = _utm_crs_for(zone, band)
    rect_utm = _project_polygon_lonlat_to_utm(rect_ll, crs_utm)

    # 3) Create a 100 km grid covering the UTM bbox; only keep cells that intersect rect_utm
    minx, miny, maxx, maxy = rect_utm.bounds
    # snap to 100 km lines
    km = 100_000.0
    start_x = floor(minx / km) * km
    start_y = floor(miny / km) * km
    end_x   = ceil(maxx / km) * km
    end_y   = ceil(maxy / km) * km

    transformer_to_ll = Transformer.from_crs(crs_utm, "EPSG:4326", always_xy=True)

    grids = dict()
    # Iterate grid cells; intersect with rect_utm; sample centroid → MGRS; extract 100k letters
    y = start_y
    while y < end_y:
        x = start_x
        while x < end_x:
            cell = box(x, y, x + km, y + km)
            if cell.intersects(rect_utm):
                c = cell.intersection(rect_utm).centroid  # robust even if clipped at edges
                lon, lat = transformer_to_ll.transform(c.x, c.y)
                # MGRS with 0 precision returns e.g. '21LYE'
                mgrs_full = _mgrs.toMGRS(lat, lon, MGRSPrecision=0)
                # last two letters are the 100 km grid ID
                square = mgrs_full[-2:]
                if filter is not None and square not in filter:continue
                grids[square] = cell
            x += km
        y += km

    return grids




def generate_mgrs_tiles(bbox=None):
    grids = []
    lonmin, latmin, lonmax, latmax = bbox

    bbox_poly = box(*bbox)
    start_zone = utm_grid_zone(lonmin)
    end_zone = utm_grid_zone(lonmax)
    start_band = utm_lat_band(latmin)
    end_band = utm_lat_band(latmax)

    zones = sorted({start_zone, end_zone})
    bands = sorted({start_band, end_band})

    rows = []
    for zone in zones:
        for band in bands:
            for grid, polygon in mgrs_100k_squares(zone, band, bbox_poly):
                grid_id = f"{zone}{band}{grid}"
                poly_utm, crs_utm = utm_bounds(grid_id)
                rows.append({
                    "tile": f"{zone}{band}{grid}",
                    "geometry": polygon
                })

    # create GeoDataFrame
    gdf = gpd.GeoDataFrame(rows, crs=crs_utm)

    # save to FlatGeobuf
    gdf.to_file("mgrs_100k_tiles.fgb", driver="FlatGeobuf")

    return gdf

if __name__ == '__main__':
    BRAZIL_BBOX = [-56.0, -15.0, -54.0, -13.0]
    generate_mgrs_tiles(bbox=BRAZIL_BBOX)