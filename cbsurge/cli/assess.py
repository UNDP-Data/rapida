import datetime
import logging
import os
import importlib
import sys
import click
from rich.progress import Progress
from cbsurge.session import Session, is_rapida_initialized
from cbsurge.project.project import Project
from cbsurge.util.setup_logger import setup_logger


logger = setup_logger()

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


def get_parent_package():
    if not __package__:
        return None
    return __package__.rsplit('.', 1)[0] if '.' in __package__ else None


def getComponents():
    try:
        with Session() as session:
            components = session.get_components()
            return list(components)
    except:
        return []


available_components = getComponents()


def getVariables():
    try:
        with Session() as session:
            variables = []
            for component in available_components:
                res = session.get_variables(component=component)
                variables.extend(res)
            return list(variables)
    except:
        return []


available_variables = getVariables()


@click.command(short_help='assess the effect of natural or social hazards')
@click.option(
    '--components', '-c', required=False, multiple=True,
    type=click.Choice(available_components, case_sensitive=False),
    help=f'One or more components to be assessed. Valid input example: {" ".join([f"-c {var}" for var in available_components[:2]])}'
)
@click.option('--variables', '-v', required=False, multiple=True,
              type=click.Choice(available_variables, case_sensitive=False),
              help=f'The variable/s to be assessed. Valid input example: {" ".join([f"-v {var}" for var in available_variables[:2]])}' )
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

    if debug:
        logger = setup_logger(level=logging.DEBUG)
    else:
        logger = setup_logger()

    if not is_rapida_initialized():
        return

    try:
        current_folder = os.getcwd()
        project = Project(path=current_folder)
    except Exception as e:
        logger.error(f'"{current_folder}" is not a valid rapida project folder. {e}')
    else:
        if len(available_components) == 0 or len(available_variables) == 0:
            logger.warning("There are no available components. Please run `rapida init` to setup the tool first")
            sys.exit(0)

        if project.is_valid:
            logger.info(f'Current project/folder: {project.path}')
            with Progress(disable=False) as progress:
                with Session() as session:
                    all_components = session.get_components()
                    components = components or all_components
                    comp_word = 'components' if len(components)> 1 else 'component'
                    # components_task = progress.add_task(
                    #     description=f'[green]Assessing [red]{"".join(components)}[green] {comp_word}', total=len(components))
                    for component_name in components:
                        if not component_name in all_components:
                            msg = f'Component {component_name} is invalid. Valid options  are: "{",".join(all_components)}"'
                            logger.error(msg)
                            click.echo(assess.get_help(ctx))
                            #progress.remove_task(components_task)
                            sys.exit(1)

                        component_parts = component_name.split('.')
                        class_name = f"{component_name.capitalize()}Component"
                        if len(component_parts) > 1:
                            class_name = f"{component_parts[-1].capitalize()}Component"

                        fqcn = f'{get_parent_package()}.components.{component_name}.{class_name}'
                        cls = import_class(fqcn=fqcn)
                        component = cls()
                        component(progress=progress, variables=variables, target_year=year, force_compute=force_compute)

                        #progress.update(components_task,  advance=1, )


        else:
            logger.info(f'"{current_folder}" is not a valid rapida project folder')







