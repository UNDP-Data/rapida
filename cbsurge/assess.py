import logging
import os
import importlib
import sys
import typing
from collections import OrderedDict
import click
from rich.progress import Progress
from cbsurge.session import Session
from cbsurge.project import Project
import inspect
logger = logging.getLogger(__name__)
s = Session()
components = s.get_components()
options = {}
for c in components:
    v = s.get_variables(component=c)
    options[c] = v

def get_callable_args(callable_obj = None):
    skip = 'cls','self'
    args = inspect.signature(callable_obj).parameters
    for k, v in args.items():
        print(type(v))
    return [e for e in args if e not in skip]

def get_comp_args(comp_class = None):
    skip = 'cls','self', 'args', 'kwargs'
    methods = '__init__', 'assess', 'download', '__call__'
    u = list()
    rd = OrderedDict()
    for mname in methods:
        m = getattr(comp_class, mname)
        args = inspect.signature(m).parameters
        for pname, p in args.items():
            if pname in u or pname in skip:continue
            rd[pname] = p
    return rd



def import_class(fqcn: str):
    """Dynamically imports a class using its fully qualified class name.

    Args:
        fqcn (str): Fully qualified class name (e.g., 'package.module.ClassName').

    Returns:
        type: The imported class.
    """
    module_name, class_name = fqcn.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def convert_params_to_click_options(params: dict, func):
    """Convert a dictionary of inspect.Parameter objects to Click options dynamically."""
    for name, param in reversed(params.items()):  # Reverse to maintain order
        click_type = str  # Default type
        multiple=False
        if param.annotation in [int, float, bool, str]:
            click_type = param.annotation
        elif param.annotation == typing.List[str]:  # Handle List[str]
            click_type = str
            multiple = True
        elif param.annotation == typing.List[int]:  # Handle List[int]
            click_type = int
            multiple = True
        elif param.default is not inspect.Parameter.empty:
            click_type = type(param.default)

        required = param.default in [inspect.Parameter.empty, None]
        default_value = None if required else param.default
        if multiple:
            default_value = []

        # Append "(optional)" to help text if it's not required
        help_text = f"Auto-generated option for {name}"
        if not required:
            help_text += " (optional)"
        func = click.option(
            f'--{name.replace("_", "-")}',
            required=required,
            default=default_value,
            type=click_type,
            help=help_text,
            multiple=multiple,
            show_default=True
        )(func)

    return func

@click.command()

@click.option(
    '--components', '-c', required=False, type=click.STRING,
    help=f'One or more components to be assessed. Valid input example: --components "{", ".join(components)}". '

)

@click.option('-v', '--variables', required=False, type=click.STRING,
               help=f'The variable/s to be assessed. Valid input example: --variables' )

def assess( components=None,  variables=None):

    """ Asses/evaluate a specific geospatial exposure components/variables"""
    os.chdir('ap') #TODO delete me
    current_folder = os.getcwd()
    project = Project(path=current_folder)
    if project.is_valid:
        logger.info(f'Current project/folder: {project.path}')
        with Progress(disable=False) as progress:
            with Session() as session:
                all_components = session.get_components()
                component_list = components.split(",") if components else []
                components_task = progress.add_task(
                    description=f'[green]Assessing {"".join(component_list)} components', total=len(component_list))
                for comp_name in component_list:
                    if not comp_name in all_components:
                        msg = f'Component {comp_name} is invalid. Valid options  are: "{",".join(all_components)}"'
                        logger.error(msg)
                        #click.echo(assess.get_help(ctx))
                        progress.remove_task(components_task)
                        sys.exit(1)
                    fqcn = f'{__package__}.components.{c}.{comp_name.capitalize()}Component'
                    cls = import_class(fqcn=fqcn)
                    component = cls()
                    component(progress=progress, variables=variables)


            #         all_variables = session.get_variables(component=component_to_assess)
            #         variables_to_assess = all_variables.intersection(variable) or all_variables
            #         progress.update(components_task, advance=1, description=f'Assessing component {component_to_assess}')
            #         variables_task = progress.add_task(description=f'[red] Going to assess {len(variables_to_assess)} variables', total=len(variables_to_assess))
            #         for variable_to_assess in variables_to_assess:
            #             logger.debug(f'Assessing {variables_to_assess} for {component_to_assess}')
            #             progress.update(variables_task, advance=1, description=f'Assessing {variable_to_assess}')
            #
            #             time.sleep(random.randint(1,3))



