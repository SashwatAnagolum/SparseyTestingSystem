Write-Output "Creating new virtual environment with Python 3.11"
# create virtual environment specifically using Python 3.11
python3.11 -m venv .venv
Write-Output "Activating virtual environment."
# activate virtual environment
. .\.venv\Scripts\Activate.ps1
Write-Output "Installing project requirements."
# install requirements
python -m pip install -r requirements.txt
Write-Output "Installing project as editable package."
# install editable sparseypy
python -m pip install -e .
Write-Output "Environment setup complete! You are ready to edit in VS Code."