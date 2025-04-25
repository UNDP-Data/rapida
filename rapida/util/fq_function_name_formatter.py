import logging

logger = logging.getLogger(__name__)

class FQFunctionNameFormatter(logging.Formatter):
    """Custom formatter to display the fully qualified function name in logs."""

    def format(self, record):
        # Extract the full module path (e.g., 'my_project.module.submodule')
        full_module_path = record.name

        # Extract the top-level package name
        package = full_module_path.split(".")[0] if "." in full_module_path else full_module_path

        # Construct fully qualified function name with package
        record.fqfunc = f"{package}.{full_module_path}.{record.funcName}"

        return super().format(record)