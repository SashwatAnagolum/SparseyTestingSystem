# create virtual environment specifically using Python 3.11
echo "Creating new virtual environment."
python3.11 -m venv .venv
# activate virtual environment
echo "Activating virtual environment."
./.venv/Scripts/activate
# install requirements
echo "Installing project requirements."
python -m pip install -r requirements.txt
# install editable sparseypy
echo "Installing sparseypy as an editable package."
python -m pip install -e .
echo "Setup complete. You are now ready to develop in VS Code."
