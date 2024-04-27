import subprocess
import os
import sys

# Configuration:
docs_directory = 'docs'
project_directory = 'src'

# Detecting the operating system (Windows, Linux, or macOS)
is_windows = sys.platform.startswith('win')

# Function to run shell commands
def run_command(command):
    try:
        subprocess.check_call(command, shell=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {command}\n{e}")
        sys.exit(1)

# Generating .rst files from your docstrings
if is_windows:
    run_command(f'sphinx-apidoc -o docs\\source\\ {project_directory}')
else:
    run_command(f'sphinx-apidoc -o source/ {project_directory}')

# Build the documentation in HTML format
if is_windows:
    run_command('sphinx-build -b html docs\\source\\ docs\\build\\html')
else:
    run_command('sphinx-build -b html source/build/html')

print("Documentation updated successfully.")
