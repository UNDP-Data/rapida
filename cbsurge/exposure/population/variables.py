import json
import os.path
from azure.storage.blob import BlobServiceClient
from cbsurge.exposure.population import worldpop
from cbsurge import util
from cbsurge.session import Session
import logging
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


SEX_GROUPS = 'male', 'female'
AGE_GROUPS = ('child', {0,1,5,10}), ('active', set(range(15, 65, 5) )), ('elderly', set(range(65, 85, 5)))
AGGREGATE = 'total'
UNDP_AZURE_WPOP_PATH = f'az:undpgeohub:/stacdata/worldpop/{{year}}/{{country}}'



class  Variable():
    pass


def generate_wpop_files(root=UNDP_AZURE_WPOP_PATH, aggregate=AGGREGATE, sex_groups=SEX_GROUPS, age_groups=AGE_GROUPS):
    """
    Generate population variables dict
    :param root:
    :param aggregate:
    :param sex_groups:
    :param age_groups:
    :return:
    """
    aggregate_root = os.path.join(root, 'aggregate')

    variables = dict()
    for sex_group in sex_groups:
        for age_item in age_groups:
            age_group, age_seq = age_item
            var_name = f'{sex_group}_{age_group}'
            var_title = f'{sex_group.capitalize()} {age_group} population'
            var_source = os.path.join(root, sex_group, age_group)
            var_files = list()
            for int_age in age_seq:
                fname_template = f'{{country_lower}}_{sex_group[0]}_{int_age}_{{year}}_constrained_UNadj.tif'
                path_template = os.path.join(root, sex_group, age_group, fname_template)
                var_files.append(path_template)

            variables[var_name] = dict(title=var_title, source=var_source, files=var_files)

            #age group aggregates
            var_name = f'{age_group}_{aggregate}'
            var_title = f'{age_group.capitalize()} population'
            var_files = [os.path.join(aggregate_root, f'{{country}}_{age_group}_{aggregate}.tif')]
            var_source = '+'.join([f'{e}_{age_group}' for e in sex_groups])
            variables[var_name] = dict(title=var_title, source=var_source, files=var_files)
        # sex group aggregates
        var_name = f'{sex_group}_{aggregate}'
        var_title = f'{sex_group.capitalize()} population'
        var_files = [os.path.join(aggregate_root, f'{{country}}_{sex_group}_{aggregate}.tif')]
        var_source = '+'.join([f'{sex_group}_{e[0]}' for e in age_groups])
        variables[var_name] = dict(title=var_title, source=var_source, files=var_files)
    #total aggregate
    var_name = aggregate
    var_title = f'{aggregate.capitalize()} population'
    var_files = [os.path.join(aggregate_root, f'{{country}}_{aggregate}.tif')]
    var_source = '+'.join([f'{e}_{aggregate}' for e in SEX_GROUPS])

    variables[var_name] = dict(title=var_title, source=var_source, files=var_files)

    return variables




if __name__ == '__main__':
    logger = util.setup_logger(name='rapida', level=logging.INFO)






    variables = generate_wpop_files()
    print(json.dumps(variables, indent=2))

