from rapida.util.in_notebook import in_notebook
import shlex
import click
from rapida.cli import cli

def run_cmd_jin(rapida_command_name:str = None, command_string: str = None):
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


def run_cmd(rapida_command_name: str = None, command_string: str = None):
    """
    Invoke a Click entry-point programmatically in Jupyter.
    Handles no-arg help triggers, parameter errors, and async execution safely.
    """
    if not rapida_command_name:
        raise ValueError("A command name must be provided (e.g., run_cmd('connectivity')).")

    # Guard against None for command_string
    args = shlex.split(command_string) if command_string else []
    rapida_command_names = list(cli.list_commands(None))

    if "auth" in rapida_command_names:
        rapida_command_names.remove("auth")

    assert rapida_command_name in rapida_command_names, (
        f"Invalid command '{rapida_command_name}'. Valid options are: {', '.join(rapida_command_names)}"
    )

    full_args = [rapida_command_name] + args

    try:
        # Run through root group so context (ctx.obj['progress']) is initialized
        return cli.main(args=full_args, standalone_mode=False)

    except SystemExit as e:
        # Exit code 0 indicates normal termination (e.g., --help or no_args_is_help)
        if e.code != 0:
            raise click.ClickException(
                f"Command `{rapida_command_name} {command_string or ''}` exited with status {e.code}."
            )
        return None

    except click.ClickException as e:
        # Formats Click usage/validation errors (e.g., missing required options) cleanly
        e.show()
        return None