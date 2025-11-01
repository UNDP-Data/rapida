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

# mapping asset name and band name
SENTINEL2_ASSET_MAP = {
    "coastal": "B01", # cloud
    "blue": "B02", # land use, cloud
    "green": "B03", # land use
    "red": "B04", # land use, cloud
    "rededge1": "B05", # land use, cloud
    "rededge2": "B06", # land use
    "rededge3": "B07", # land use
    "nir": "B08", # land use, cloud
    "nir08": "B8A", # cloud
    "nir09": "B09", # cloud
    "cirrus": "B10", # cloud
    "swir16": "B11", # land use, cloud
    "swir22": "B12", # land use, cloud
}

SENTINEL2_BAND_MAP = {v:k for k,v in SENTINEL2_ASSET_MAP.items()}