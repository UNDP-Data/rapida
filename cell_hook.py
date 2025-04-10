from IPython import get_ipython
from cbsurge.az.authwidget import load_ui
def pre_cell_execution(exec_info):
    print(f"Cell is about to execute: {exec_info.raw_cell}")


# Get the current IPython shell
ip = get_ipython()

# Register the pre_run_cell event
ip.events.register('pre_run_cell', pre_cell_execution)
