import datetime
import logging
import os
import importlib
import sys
import click
from rich.progress import Progress
from rich.console import Console
from rapida.session import Session, is_rapida_initialized
from rapida.project.project import Project
from rapida.util.setup_logger import setup_logger


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

def get_variables_by_components(components):
    result = {}
    try:
        with Session() as session:
            for comp in components:
                result[comp] = session.get_variables(component=comp)
    except:
        pass
    return result


component_variable_map = get_variables_by_components(available_components)


def validate_variables(ctx, param, value):
    """
    click callback function to validate -v/--variable value

    If `-a/--all` is passed, all variables are validated and shown as valid options in the error message.
    If `-c/--component` is passed, only relevant component variables are shown as valid options in the error message.
    """
    selected_components = ctx.params.get('components')
    use_all = ctx.params.get('all')

    if not value:
        return value

    # when -a/--all is used
    if use_all or not selected_components:
        all_vars = set(var for vars_list in component_variable_map.values() for var in vars_list)
        invalid = [v for v in value if v not in all_vars]
        if invalid:
            raise click.BadParameter(f"Invalid variable: {', '.join(invalid)}. Valid options: {', '.join(all_vars)}")
        return value

    # when -c/--component is used
    valid_vars = set()
    for comp in selected_components:
        valid_vars.update(component_variable_map.get(comp, []))

    invalid = [v for v in value if v not in valid_vars]
    if invalid:
        raise click.BadParameter(f"Invalid variable{'s' if len(invalid) > 1 else ''}: {', '.join(invalid)} for selected component{'s:' if len(selected_components) > 1 else ':'} {', '.join(selected_components)}. {', '.join(invalid)}. Valid options: {', '.join(valid_vars)}")

    return value


def validate_datetime_range(ctx, param, value):
    """
    click callback function to validate --datetime value
    """
    if not value:
        return value

    selected_components = ctx.params.get('components')

    if not 'landuse' in selected_components:
        return None

    try:
        if '/' in value:
            # user pass the date range
            start_str, end_str = value.split('/', 1)
            start_date = datetime.datetime.strptime(start_str, '%Y-%m-%d').date()
            end_date = datetime.datetime.strptime(end_str, '%Y-%m-%d').date()
        else:
            # user pass a single date
            end_date = datetime.datetime.strptime(value, '%Y-%m-%d').date()
            start_date = end_date.replace(year=end_date.year - 1)

        today = datetime.date.today()

        # ensure the end date is before today's date
        if not (end_date <= today):
            raise click.BadParameter(f"End date ({end_date}) must be before today ({today})")

        # ensure having at least one day between start date and end date
        if (end_date - start_date).days < 1:
            raise click.BadParameter(f"Date range must be at least 1 day apart. Start: {start_date}, End: {end_date}")

        # ensure starting date is after sentinel 2 available date
        SENTINEL2_START_DATE = datetime.date(2015, 6, 27)
        if start_date <= SENTINEL2_START_DATE:
            raise click.BadParameter(
                f"Start date ({start_date}) must be after Sentinel-2 operational start date ({SENTINEL2_START_DATE})")

        return f"{start_date.strftime('%Y-%m-%d')}/{end_date.strftime('%Y-%m-%d')}"

    except ValueError as e:
        if "time data" in str(e):
            raise click.BadParameter(f"Invalid date format. Use YYYY-MM-DD or YYYY-MM-DD/YYYY-MM-DD format")
        raise click.BadParameter(str(e))


def build_variable_help():
    """
    build help message for variable option
    """
    parts = []
    for comp, vars_list in component_variable_map.items():
        if vars_list:
            vars_str = ", ".join(vars_list)
            parts.append(f"{comp} ({vars_str})")
    return "The variable/s to be assessed. Will be filtered by selected components. Available variables per component:\n\n" + "\n\n".join(parts)


@click.command(short_help='assess/evaluate a specific geospatial exposure components/variables', no_args_is_help=True)
@click.option(
    '--all', '-a', is_flag=True, default=False,
    help="compute all components and variables if this option is set"
)
@click.option(
    '--components', '-c', required=False, multiple=True,
    type=click.Choice(available_components, case_sensitive=False),
    help=f'One or more components to be assessed. Valid input example: {" ".join([f"-c {var}" for var in available_components[:2]])}'
)
@click.option('--variables', '-v', required=False, multiple=True,
              type=str, callback=validate_variables,
              help=f"{build_variable_help()}")
@click.option('--year', '-y', required=False, type=int, multiple=False,default=datetime.datetime.now().year,
              show_default=True,help=f'The year for which to compute population' )
@click.option('--datetime-range', '-dt', required=False, type=str, callback=validate_datetime_range, default=datetime.date.today().strftime('%Y-%m-%d'),
              help=f"Optional. Date range for landuse component in YYYY-MM-DD/YYYY-MM-DD format or single date YYYY-MM-DD (12 months range). Only valid when 'landuse' component is selected. Start date must be after end date, end date must be before today, and at least 1 day apart.")
@click.option('--cloud-cover', '-cc', required=False, type=int, multiple=False, default=5,
              show_default=True,help=f"Optional. Minimum cloud cover rate to search items for landuse component.")
@click.option('-p', '--project',
              default=None,
              type=click.Path(file_okay=False, dir_okay=True, resolve_path=True),
              help="Optional. A project folder with rapida.json can be specified. If not, current directory is considered as a project folder.")
@click.option('--force', '-f', default=False, show_default=True,is_flag=True,
              help=f'Force assess components. Downloaded data or computed data will be ignored and recomputed.')
@click.option('--debug',
              is_flag=True,
              default=False,
              help="Set log level to debug"
              )
@click.pass_context
def assess(ctx, all=False, components=None,  variables=None, year=None, datetime_range=None, cloud_cover=None, project: str = None, force=False, debug=False):
    """
    Assess/evaluate a specific geospatial exposure components/variables

    `-a/--all` option to assess all components (it may take longer time).

    `-c/--component` to assess only specific components.

    `-v/--variable` to assess only specific variables. If a variable is specified, but it is not in specified components, the variable will be ignored.

    `-p/--project` to assess in a specific project folder other than current directory.

    As default, this command tries to avoid download/compute again if they already exist. If you wish to redownload or recompute by force, use `-f/--force` flag explicitly.

    Usage:

    rapida assess --all: assess all components

    rapida assess -c rwi: assess RWI component only.

    rapida assess -c rwi -c population: assess RWI and population component only

    rapida assess -c population -v male_total -v female_total: assess only male and female total population.

    rapida assess -c rwi -p ./data/sample_project: assess RWI component for RAPIDA project stored at sample_project folder.

    rapida assess -c landuse -dt 2025-02-01/2025-05-31 -cc 10: Search Sentinel 2 item which is less than 10% of cloud cover from February to May 2025.

    """
    setup_logger(name='rapida', level=logging.DEBUG if debug else logging.INFO)

    if not is_rapida_initialized():
        return

    if project is None:
        project = os.getcwd()
    else:
        os.chdir(project)

    prj = Project(path=project)
    if not prj.is_valid:
        logger.error(f'Project "{project}" is not a valid RAPIDA project')
        return

    available_variables = list({var for vars_list in component_variable_map.values() for var in vars_list})
    if len(available_components) == 0 or len(available_variables) == 0:
        logger.warning("There are no available components. Please run `rapida init` to setup the tool first")
        sys.exit(0)

    logger.info(f'Current project/folder: {prj.path}')
    with Progress(disable=False, console=None) as progress:
        with Session() as session:
            all_components = session.get_components()
            target_components = components
            if len(components) == 0:
                if all:
                    target_components = set(filter(lambda x: x != "landuse", all_components))
                else:
                    logger.warning(f"At least one component is required. If you want to assess all components, use --all option")
                    return
            else:
                if all:
                    logger.warning(f"--all option is ignored and to process {", ".join(components)}")

            for component_name in target_components:
                if not component_name in all_components:
                    msg = f'Component {component_name} is invalid. Valid options  are: "{",".join(all_components)}"'
                    logger.error(msg)
                    click.echo(assess.get_help(ctx))
                    sys.exit(1)

                component_parts = component_name.split('.')
                class_name = f"{component_name.capitalize()}Component"
                if len(component_parts) > 1:
                    class_name = f"{component_parts[-1].capitalize()}Component"

                fqcn = f'{get_parent_package()}.components.{component_name}.{class_name}'
                cls = import_class(fqcn=fqcn)
                component = cls()

                component(progress=progress,
                          variables=variables,
                          target_year=year,
                          datetime_range=datetime_range,
                          cloud_cover=cloud_cover,
                          force=force)



