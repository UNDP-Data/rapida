DYNAMIC_WORLD_COLORMAP = {
    0: {'color': '#419bdf', 'label': 'water'},
    1: {'color': '#397d49', 'label': 'trees'},
    2: {'color': '#88b053', 'label': 'grass'},
    3: {'color': '#7a87c6', 'label': 'flooded_vegetation'},
    4: {'color': '#e49635', 'label': 'crops'},
    5: {'color': '#dfc35a', 'label': 'shrub_and_scrub'},
    6: {'color': '#c4281b', 'label': 'built'},
    7: {'color': '#a59b8f', 'label': 'bare'},
    8: {'color': '#b39fe1', 'label': 'snow_and_ice'},
}

STAC_MAP = {
    'earth-search': 'https://earth-search.aws.element84.com/v1'
}

needed_assets = (
    'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B11', 'B12'
)
earth_search_assets = (
    'blue', 'green', 'red', 'rededge1', 'rededge2', 'rededge3', 'nir', 'swir16', 'swir22'
)
SENTINEL2_ASSET_MAP = dict(zip(earth_search_assets, needed_assets))