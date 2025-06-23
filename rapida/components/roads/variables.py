from collections import OrderedDict
import logging

logger = logging.getLogger(__name__)

def generate_variables():
    """
    Generate roads variables dict
    :return Roads variables definition
    """

    variables = OrderedDict()

    license = "Creative Commons Zero 1.0 Universal"
    attribution = "Global biodiversity model for policy support, GLOBIO"

    for operator in ['sum', 'density']:
        name = operator
        if operator == 'sum':
            name = "length"
        variables[f'roads_{name}'] = dict(
            title=f'Total {name} of roads',
            source='geohub:/api/datasets/300da70781b7a53808aab824543e6c2b',
            operator=operator,
            percentage=True if operator != 'density' else False,
            license=license,
            attribution=attribution,
        )
    for road_type in ["Highways", "Local roads", "Primary roads", "Secondary roads", "Tertiary roads", "Others"]:
        for operator in ['sum', 'density']:
            name = operator
            if operator == 'sum':
                name = "length"
            variables[f'{road_type.replace(" ", "").lower()}_{name}'] = dict(
                title=f'Total {name} of {road_type} roads',
                source=f'geohub:/api/datasets/300da70781b7a53808aab824543e6c2b',
                operator=operator,
                percentage=True if operator != 'density' else False,
                license=license,
                attribution=attribution,
                source_column="GP_RTP",
                source_column_value=road_type
            )

    return variables