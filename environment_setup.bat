REM creating new virtual environment using Python 3.11
python3.11 -m venv .venv
REM activating new virtual environment
CALL .\.venv\Scripts\activate.bat
REM installing project requirements
python -m pip install -r requirements.txt
REM installing sparseypy as an editable package
python -m pip install -e .