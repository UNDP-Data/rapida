import inspect
import click
from functools import wraps
import asyncio


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
    Combined group that handles async subcommands and lists keys.
    """

    def list_commands(self, ctx):
        return self.commands.keys()

    def command(self, *args, **kwargs):
        # Automatically wrap all @group.command() calls in AsyncCommand
        kwargs.setdefault('cls', AsyncCommand)
        return super().command(*args, **kwargs)

    def group(self, *args, **kwargs):
        # Ensure nested groups inherit this behavior
        kwargs.setdefault('cls', RapidaCommandGroup)
        return super().group(*args, **kwargs)