# Description: Constants for the population module

AZ_ROOT_FILE_PATH = "worldpop"
CONTAINER_NAME = "stacdata"
WORLDPOP_AGE_MAPPING = {
    "child": [0, 14],
    "active": [15, 64],
    "elderly": [65, 100],
}
SEX_MAPPING = {
    "M": "male",
    "F": "female",
}

DATA_YEAR = 2020

AGESEX_STRUCTURE_COMBINATIONS = [
            {"sexes": ["M"], "age_group": None, "label": "male_total"},
            {"sexes": ["F"], "age_group": None, "label": "female_total"},
            {"sexes": ["M"], "age_group": "active", "label": "male_active"},
            {"sexes": ["F"], "age_group": "active", "label": "female_active"},
            {"sexes": ["M"], "age_group": "child", "label": "male_child"},
            {"sexes": ["F"], "age_group": "child", "label": "female_child"},
            {"sexes": ["M"], "age_group": "elderly", "label": "male_elderly"},
            {"sexes": ["F"], "age_group": "elderly", "label": "female_elderly"},
            {"sexes": ["M", "F"], "age_group": "elderly", "label": "elderly_total"},
            {"sexes": ["M", "F"], "age_group": "child", "label": "child_total"},
            {"sexes": ["M", "F"], "age_group": "active", "label": "active_total"},
        ]