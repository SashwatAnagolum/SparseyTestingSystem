"""
update_docs.py - script to update this project's Sphinx documentation.
"""
import subprocess
import sys

# Detecting the operating system (Windows, Linux, or macOS)
is_windows = sys.platform.startswith('win')

# Function to run shell commands
def run_command(command: str):
    """
    Runs a command using the system shell. Terminates the program on command failure.

    Args:
        command (str): the command to run
    """
    try:
        subprocess.check_call(command, shell=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {command}\n{e}")
        sys.exit(1)

# Generating .rst files from your docstrings
if is_windows:
    run_command('sphinx-apidoc -o docs\\source\\src')
else:
    run_command('sphinx-apidoc -o docs/source/src')

# Build the documentation in HTML format
if is_windows:
    run_command('sphinx-build -b html docs\\source\\docs\\build\\html')
else:
    run_command('sphinx-build -b html docs/source/build/html')

print("Documentation updated successfully.")
