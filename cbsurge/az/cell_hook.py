from IPython import get_ipython
from cbsurge.az.authwidget import load_ui
from cbsurge.util.in_notebook import in_notebook

def pre_cell_execution(exec_info):
    if in_notebook():
       a = load_ui()


# Get the current IPython shell
ip = get_ipython()

# Register the pre_run_cell event
ip.events.register('pre_run_cell', pre_cell_execution)