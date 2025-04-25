from collections import OrderedDict


def generate_variables():
    # source format is mspc:{collection id}:{target band value}
    # https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_DYNAMICWORLD_V1#bands
    # https://earth-search.aws.element84.com/v1/collections/sentinel-2-l1c
    source = "earth-search:sentinel-2-l1c"

    license = "Proprietary"
    dynamic_world_citation = "Brown  et al. (2022). Dynamic World Near real-time global 10 m land use land cover mapping. Scientific Data 9(1)."
    attribution = f"ESA, Sinergise, AWS, Element 84, {dynamic_world_citation}"

    variables = OrderedDict()
    variables['built_area'] = dict(title='Built-up area',
                                   source=f"{source}:6",
                                   operator='sum',
                                   percentage=True,
                                   license=license,
                                   attribution=attribution,
                                   )
    variables['crops_area'] = dict(title='Cropland area',
                                   source=f"{source}:4",
                                   operator='sum',
                                   percentage=True,
                                   license = license,
                                   attribution = attribution,
                                   )
    return variables