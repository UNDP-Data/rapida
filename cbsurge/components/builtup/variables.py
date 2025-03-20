from collections import OrderedDict


def generate_variables():
    variables = OrderedDict()
    variables['builtup'] = dict(title='Built-up area',
                                source='geohub:/api/datasets/305b980bdafeeff15151526c6ff1050a',
                                operator='sum')
    return variables