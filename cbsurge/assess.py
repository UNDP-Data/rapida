import datetime
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

@click.command(short_help='TAGA')

@click.option(
    '--components', '-c', required=False, multiple=True,
    #help=f'One or more components to be assessed. Valid input example: --components "{", ".join(components)}". '
    help=f'One or more components to be assessed. Valid input example: -c component1 -c component2 '

)

@click.option('--variables', '-v', required=False, type=click.STRING, multiple=True,
               help=f'The variable/s to be assessed. Valid input example: -v variable1 -v variable2' )
@click.option('--year', '-y', required=False, type=int, multiple=False,default=datetime.datetime.now().year,
              show_default=True,help=f'The year for which to compute population' )

@click.option('--force_compute', '-f', default=False, show_default=True,is_flag=True,
              help=f'Force recomputation from sources that are files')

@click.option('--debug', '-d', default=False, show_default=True, is_flag=True,
              help=f'Turn on debug mode')


@click.pass_context
def assess(ctx, components=None,  variables=None, year=None, force_compute=False, debug=False):
    """Assess the effect of natural or social hazard """
    """ Asses/evaluate a specific geospatial exposure components/variables"""
    logger = logging.getLogger('rapida')
    if debug:
        logger.setLevel(logging.DEBUG)

    try:
        current_folder = os.getcwd()
        project = Project(path=current_folder)
    except Exception as e:
        logger.error(f'"{current_folder}" is not a valid rapida project folder. {e}')
    else:
        if project.is_valid:
            logger.info(f'Current project/folder: {project.path}')
            with Progress(disable=False) as progress:
                with Session() as session:
                    all_components = session.get_components()
                    components = components or all_components
                    components_task = progress.add_task(
                        description=f'[green]Assessing {"".join(components)} ', total=len(components))
                    for component_name in components:
                        if not component_name in all_components:
                            msg = f'Component {component_name} is invalid. Valid options  are: "{",".join(all_components)}"'
                            logger.error(msg)
                            #click.echo(assess.get_help(ctx))
                            progress.remove_task(components_task)
                            sys.exit(1)
                        fqcn = f'{__package__}.components.{component_name}.{component_name.capitalize()}Component'
                        cls = import_class(fqcn=fqcn)
                        component = cls()
                        component(progress=progress, variables=variables, target_year=year, force_compute=force_compute)
                    progress.remove_task(components_task)
        else:
            logger.info(f'"{current_folder}" is not a valid rapida project folder')







