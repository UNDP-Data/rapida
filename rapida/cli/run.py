from rapida.util.in_notebook import in_notebook
import shlex
import click
from rapida.cli import cli

def run_cmd(rapida_command_name:str = None, command_string: str = None):
    """
    Invoke a Click entry‐point programmatically using a shell‐style string.


    :param command:        the exact command‐line you’d type in your shell,
                           e.g. "-c population -f"
    :returns:              whatever your click command returns (or None)
    :raises:               ClickException on non‐zero exit
    """
    # split like a shell would (handles quoted args, etc.)
    args = shlex.split(command_string)
    rapida_command_names = list(cli.list_commands(None))
    rapida_command_names.remove('auth')
    assert rapida_command_name in rapida_command_names, f'Invalid command {rapida_command_name}. Valid options are {",".join(rapida_command_names)}'

    try:
        # call Click’s entrypoint, passing in our args list
        command = cli.commands.get(rapida_command_name)
        return command.main(args=args, standalone_mode=False)
    except SystemExit as e:
        # catch Click’s “exit” and rewrap non‐zero into a ClickException
        if e.code != 0:
            raise click.ClickException(
                f"Command `{command_string}` exited with status {e.code}."
            )
