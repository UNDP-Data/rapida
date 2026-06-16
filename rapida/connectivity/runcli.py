import subprocess
import logging
logger = logging.getLogger(__name__)

class NativeEngineError(Exception):
    """Custom exception to cleanly format C++ engine crashes."""
    pass

def run_cli(cmd: list[str]) -> str:
    """
    Executes a native C++ command, safely capturing all outputs.
    Raises a beautifully formatted NativeEngineError if the process fails.
    """
    cmd_str = " ".join(cmd)
    logger.debug(f"Executing: {cmd_str}")

    try:
        # capture_output=True automatically grabs both stdout and stderr
        # text=True decodes the raw bytes into strings
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True
        )
        return result.stdout.strip()

    except subprocess.CalledProcessError as e:
        # Construct a clean, highly readable diagnostic report
        error_msg = (
            f"\n{'='*60}\n"
            f"🚨 NATIVE ENGINE CRASH [Exit Code: {e.returncode}]\n"
            f"Command: {cmd_str}\n"
            f"{'-'*60}\n"
            f"STDERR:\n{e.stderr.strip() or '[No standard error output]'}\n"
            f"{'-'*60}\n"
            f"STDOUT:\n{e.stdout.strip() or '[No standard output]'}\n"
            f"{'='*60}"
        )
        # Using 'from None' hides the messy subprocess traceback
        raise NativeEngineError(error_msg) from None