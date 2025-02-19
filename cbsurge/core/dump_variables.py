import importlib
import os.path
from pkgutil import walk_packages


def dump_variables(root_package_name = 'cbsurge', variables_module='variables', function_name='generate_variables'):
    """
    Iterate over all modules in path  that start with root_package_name, end with variables_module and are
    not packages. Consequently, import the function_name function and run it.

    :param root_package_name: str, cbsurge
    :param variables_module: str, variables
    :param function_name: 'generate_variables'
    :return: dict['variables'] = {'component': component_vars}
    """
    vars_dict = {}
    vars_dict[variables_module] = {}
    # dir_path should be a parent folder of cbsurge package
    dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    for p in walk_packages(path=[dir_path]):
        if p.name.startswith(root_package_name) and p.name.endswith(variables_module) and not p.ispkg:
            m = importlib.import_module(name=p.name)
            if hasattr(m, function_name):
                var_dict = getattr(m, 'generate_variables')()
                component_name = p.name.split('.')[-2]

                vars_dict[variables_module][component_name] = var_dict
    return vars_dict
