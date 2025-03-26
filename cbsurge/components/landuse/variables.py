from collections import OrderedDict


def generate_variables():
    # source format is mspc:{collection id}:{target band value}
    # https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_DYNAMICWORLD_V1#bands
    source = "mspc:sentinel-2-l2a"

    variables = OrderedDict()
    variables['built_area'] = dict(title='Built-up area',
                                   source=f"{source}:6",
                                   operator='sum',
                                   percentage=True,
                                   )
    variables['crops_area'] = dict(title='Cropland area',
                                   source=f"{source}:4",
                                   operator='sum',
                                   percentage=True,
                                   )
    return variables