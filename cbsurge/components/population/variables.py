import json
import os.path
from cbsurge import util
from collections import OrderedDict
import logging

from cbsurge.session import Session

logger = logging.getLogger(__name__)
'''
Age groups

0–4 Months (Newborn)
5 months – 1 1/2 years (Baby)
1 1/2 to 3 years (Toddler)
3–5 years (Preschooler)
6- 9 years (Child)
10–12 1/2 years (Preteen or “Tween”)
13–17 years (Teenager)
18–21 years ( Young Adult)
21–39 years (Adult)
40–55 years (Middle Aged)
56- 65 years (Retiree)
66–75+ years (Senior Citizen)
'''


SEXES = 'male', 'female'
AGE_GROUPS = ('child', {0,1,5,10}), ('active', set(range(15, 65, 5) )), ('elderly', set(range(65, 85, 5)))
AGGREGATE = 'total'
UNDP_AZURE_WPOP_PATH = f'az:{{account_name}}:{{stac_container_name}}/worldpop/{{year}}/{{country}}'


def generate_variables(root=UNDP_AZURE_WPOP_PATH, aggregate=AGGREGATE, sexes=SEXES, age_groups=AGE_GROUPS):
    """
    Generate population variables dict
    :param root:
    :param aggregate:
    :param sexes:
    :param age_groups:
    :return:
    """
    with Session() as session:
        # if root is same with UNDP_AZURE_WPOP_PATH, replace account name and container name from session object
        if root == UNDP_AZURE_WPOP_PATH:
            root = root.replace("{account_name}", session.get_account_name()).replace("{stac_container_name}", session.get_stac_container_name())

    aggregate_root = os.path.join(root, 'aggregate')

    variables = OrderedDict()
    for sex in sexes:
        for age_item in age_groups:
            age_group, age_seq = age_item
            name = f'{sex}_{age_group}'
            title = f'{sex.capitalize()} {age_group} population'
            source = os.path.join(aggregate_root, f'{{country}}_{sex}_{age_group}.tif')
            sources = list()
            for int_age in age_seq:
                fname_template = f'{{country_lower}}_{sex[0]}_{int_age}_{{year}}_constrained_UNadj.tif'
                path_template = os.path.join(root, sex, age_group, fname_template)
                sources.append(path_template)

            variables[name] = dict(title=title, source=source, sources=sources, operator='sum')

            #age group aggregates
            name = f'{age_group}_{aggregate}'
            title = f'{age_group.capitalize()} population'
            source = os.path.join(aggregate_root, f'{{country}}_{age_group}_{aggregate}.tif')
            sources = '+'.join([f'{e}_{age_group}' for e in sexes])
            variables[name] = dict(title=title, source=source, sources=sources, operator='sum')
        # sex group aggregates
        name = f'{sex}_{aggregate}'
        title = f'{sex.capitalize()} population'
        source = os.path.join(aggregate_root, f'{{country}}_{sex}_{aggregate}.tif')
        sources = '+'.join([f'{sex}_{e[0]}' for e in age_groups])
        variables[name] = dict(title=title, source=source, sources=sources, operator='sum')
    #total aggregate
    name = aggregate
    title = f'{aggregate.capitalize()} population'
    source = os.path.join(aggregate_root, f'{{country}}_{aggregate}.tif')
    sources = '+'.join([f'{e}_{aggregate}' for e in SEXES])
    variables[name] = dict(title=title, source=source, sources=sources, operator='sum')

    #dependencies
    variables['dependency'] = dict(title='Total dependency ratio', sources='((child_total+elderly_total)/active_total)*100')
    variables['child_dependency'] = dict(title='Child dependency ratio', sources='(child_total/active_total)*100')
    variables['elderly_dependency'] = dict(title='Elderly dependency ratio', sources='(elderly_total/active_total)*100' )
    return variables





if __name__ == '__main__':
    logger = util.setup_logger(name='rapida', level=logging.INFO)

    variables = generate_variables()
    print(json.dumps(variables, indent=2))

    # with Session() as ses:
    #     pop = ses.config['variables']['population']
    #
    #     ses.config['variables']['population'] = variables
    #     print(json.dumps(ses.config, indent=2))
    #     ses.save_config()

