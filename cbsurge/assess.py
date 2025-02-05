import logging
import os
import random
import time

import click
from rich.progress import Progress
from cbsurge.session import Session
from cbsurge.project import Project
logger = logging.getLogger(__name__)

s = Session()
components = s.get_components()
options = {}
for c in components:
    v = s.get_variables(component=c)
    options[c] = v
@click.command()


@click.option('-c', '--component', required=False, type=str, multiple=True,
               help=f'The component/s to be assessed. Valid values: "{"".join(components)}"' )

@click.option('-v', '--variable', required=False, type=click.Choice([]), multiple=True,
               help='The variable/s to be assessed' )


def assess(component=None, variable=None, **kwargs):
    """ Asses/evaluate a specific geospatial exposure components/variables"""

    os.chdir('ap') #TODO delete me
    current_folder = os.getcwd()
    config_file = os.path.join(current_folder, Project.config_file_name)
    project = Project.from_config(config_file=config_file)
    if project.is_valid:
        logger.info(f'Current project/folder: {project.path}')

        with Progress(disable=False) as progress:

            with Session() as session:
                all_components = session.get_components()
                components_to_assess = all_components.intersection(component) or all_components
                components_task = progress.add_task(
                    description=f'[green]Going to asses {len(components_to_assess)} components', total=len(components_to_assess))
                for component_to_assess in components_to_assess:
                    all_variables = session.get_variables(component=component_to_assess)
                    variables_to_assess = all_variables.intersection(variable) or all_variables
                    progress.update(components_task, advance=1, description=f'Assessing component {component_to_assess}')
                    variables_task = progress.add_task(description=f'[red] Going to assess {len(variables_to_assess)} variables', total=len(variables_to_assess))
                    for variable_to_assess in variables_to_assess:
                        logger.debug(f'Assessing {variables_to_assess} for {component_to_assess}')
                        progress.update(variables_task, advance=1, description=f'Assessing {variable_to_assess}')

                        time.sleep(random.randint(1,3))



