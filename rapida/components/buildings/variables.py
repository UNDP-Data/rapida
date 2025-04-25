from collections import OrderedDict
GMOSM_BUILDINGS_ROOT = 'https://data.source.coop/vida/google-microsoft-osm-open-buildings/flatgeobuf/by_country'
GMOSM_BUILDINGS_URL = f'/vsicurl/{GMOSM_BUILDINGS_ROOT}/country_iso={{country}}/{{country}}.fgb::{{country}}'

def generate_variables():
    license = "Open Data Commons Open Database License (ODbL)"
    attribution = "Google, Microsoft, VIDA, Protomaps"

    variables = OrderedDict()
    variables['nbuildings'] = dict(title='Number of buildings', source=GMOSM_BUILDINGS_URL,  operator='count', percentage=True, license=license, attribution=attribution)
    variables['buildings_area'] = dict(title='Buildings surface area', source=GMOSM_BUILDINGS_URL,  operator='sum', percentage=True, license=license, attribution=attribution)
    return variables
