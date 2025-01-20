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
            {"sexes": ["male"], "age_group": None, "label": "male_total"},
            {"sexes": ["female"], "age_group": None, "label": "female_total"},
            {"sexes": ["male"], "age_group": "active", "label": "male_active"},
            {"sexes": ["female"], "age_group": "active", "label": "female_active"},
            {"sexes": ["male"], "age_group": "child", "label": "male_child"},
            {"sexes": ["female"], "age_group": "child", "label": "female_child"},
            {"sexes": ["male"], "age_group": "elderly", "label": "male_elderly"},
            {"sexes": ["female"], "age_group": "elderly", "label": "female_elderly"},
            {"sexes": ["male", "female"], "age_group": "elderly", "label": "elderly_total"},
            {"sexes": ["male", "female"], "age_group": "child", "label": "child_total"},
            {"sexes": ["male", "female"], "age_group": "active", "label": "active_total"},
        ]