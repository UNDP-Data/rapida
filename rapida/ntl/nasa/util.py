import math
def get_intersecting_tiles(bbox: tuple[float, float, float, float]) -> list[tuple[int, int]]:
    """
    Identifies VIIRS Sinusoidal tiles (h, v) intersecting a geographic bounding box.
    bbox format: (min_lon, min_lat, max_lon, max_lat)
    :return tuple of ints representing pairs of tile coordinates (horizontal, vertical)
    """
    min_lon, min_lat, max_lon, max_lat = bbox

    # VIIRS standard sinusoidal grid is approx 10x10 degrees at the equator
    # h runs 0 to 35 (180W to 180E)
    # v runs 0 to 17 (90N to 90S)
    h_min = math.floor((min_lon + 180) / 10)
    h_max = math.floor((max_lon + 180) / 10)
    v_min = math.floor((90 - max_lat) / 10)
    v_max = math.floor((90 - min_lat) / 10)

    tiles = []
    for v in range(max(0, v_min), min(18, v_max + 1)):
        for h in range(max(0, h_min), min(36, h_max + 1)):
            tiles.append(f'h{h:02d}v{v:02d}')

    return tiles

TIMESTAMP_FORMATS = {
    "A1": "%Y%m%d",  # Daily: Year + Julian Day (e.g., 2026134)
    "A2": "%Y%m%d",  # Daily: Year + Julian Day (e.g., 2026134)
    "A3": "%Y%m",  # Monthly: Year + Month (e.g., 202605)
    "A4": "%Y"     # Yearly: Year only (e.g., 2026)
}

def timestamp_format(product_id: str) -> str:
    """Determine the correct temporal format string based on product name."""
    # Check if A1, A2, A3, or A4 is in the product string (e.g., 'VNP46A1')
    for identifier, time_format in TIMESTAMP_FORMATS.items():
        if identifier in product_id:
            return time_format
