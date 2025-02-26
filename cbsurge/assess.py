import datetime
import logging
import os
import importlib
import sys
import click
from rich.progress import Progress
from cbsurge.session import Session
from cbsurge.project import Project
from cbsurge.util.setup_logger import setup_logger


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


@click.command(short_help='assess the effect of natural or social hazards')

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

    if debug:
        logger = setup_logger(level=logging.DEBUG)
    else:
        logger = setup_logger()
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
                            click.echo(assess.get_help(ctx))
                            progress.remove_task(components_task)
                            sys.exit(1)
                        fqcn = f'{__package__}.components.{component_name}.{component_name.capitalize()}Component'
                        cls = import_class(fqcn=fqcn)
                        component = cls()
                        component(progress=progress, variables=variables, target_year=year, force_compute=force_compute)
                        progress.update(components_task,description=f'Assessed {component_name}', advance=1)

                    #progress.remove_task(components_task)
        else:
            logger.info(f'"{current_folder}" is not a valid rapida project folder')







