import inspect
import click
from functools import wraps
import asyncio
import logging

def setup_debug_logging(ctx, param, value):
    """Callback that switches the log level to DEBUG instantly."""
    if value:
        logger = logging.getLogger('rapida')
        logger.setLevel(logging.DEBUG)
        for handler in logger.handlers:
            handler.setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled globally.")
    return value
class AsyncCommand(click.Command):
    """
    Async wrapper designed to work alongside nest_asyncio in Jupyter
    and standard terminal environments.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        orig_callback = self.callback

        if orig_callback is not None:
            @wraps(orig_callback)
            def wrapped_callback(*c_args, **c_kwargs):
                actual_func = inspect.unwrap(orig_callback)

                if inspect.iscoroutinefunction(actual_func):
                    # Safely get or create the event loop
                    try:
                        loop = asyncio.get_event_loop()
                    except RuntimeError:
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)

                    return loop.run_until_complete(orig_callback(*c_args, **c_kwargs))

                return orig_callback(*c_args, **c_kwargs)

            self.callback = wrapped_callback
class RapidaCommandGroup(click.Group):
    """
    Combined group that handles async subcommands, forces help on no arguments,
    and lists keys accurately.
    """

    def list_commands(self, ctx):
        return self.commands.keys()

    def add_command(self, cmd: click.Command, name: str = None) -> None:
        # Catch-all: forces help display if a subcommand is run empty
        #cmd.no_args_is_help = True
        # 2. Automatically inject --debug into every registered subcommand
        cmd.params.append(
            click.Option(
                ['--debug'],
                is_flag=True,
                help='Enable debug logging.',
                expose_value=False,
                callback=setup_debug_logging
            )
        )
        super().add_command(cmd, name)

    def command(self, *args, **kwargs):
        # Automatically wrap all inline @group.command() calls in AsyncCommand
        kwargs.setdefault('cls', AsyncCommand)
        return super().command(*args, **kwargs)

    def group(self, *args, **kwargs):
        # Ensure nested groups inherit this behavior
        kwargs.setdefault('cls', RapidaCommandGroup)
        return super().group(*args, **kwargs)