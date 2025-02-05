import logging
import os
import importlib
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
                comp_names = session.get_components()
                comp_names_to_assess = comp_names.intersection(component)
                if  not comp_names_to_assess:
                    for c in component:
                        msg = f'Component {c} is invalid. Valid options  are: "{",".join(comp_names)}"'
                        logger.error(msg)
                        raise NameError(msg)
                components_task = progress.add_task(
                    description=f'[green]Going to asses {len(comp_names_to_assess)} components', total=len(comp_names_to_assess))
                for comp_name_to_assess in comp_names_to_assess:
                    fqcn = f'{__package__}.components.{comp_name_to_assess}.component.{comp_name_to_assess.capitalize()}Component'

                    cls = import_class(fqcn=fqcn)
                    print(cls)
            #         all_variables = session.get_variables(component=component_to_assess)
            #         variables_to_assess = all_variables.intersection(variable) or all_variables
            #         progress.update(components_task, advance=1, description=f'Assessing component {component_to_assess}')
            #         variables_task = progress.add_task(description=f'[red] Going to assess {len(variables_to_assess)} variables', total=len(variables_to_assess))
            #         for variable_to_assess in variables_to_assess:
            #             logger.debug(f'Assessing {variables_to_assess} for {component_to_assess}')
            #             progress.update(variables_task, advance=1, description=f'Assessing {variable_to_assess}')
            #
            #             time.sleep(random.randint(1,3))



