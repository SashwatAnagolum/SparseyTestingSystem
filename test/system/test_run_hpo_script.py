import pytest
import subprocess
import os
#move to high level if run from lower level
os.chdir(os.path.dirname(__file__))
os.chdir('..')
os.chdir('..')
# Paths to the script and configuration files
SCRIPT_PATH = 'scripts\\run_hpo.py'
ORIGINAL_HPO_CONFIG = 'test\\reference_configs\\hpo.yaml'
TEST_HPO_CONFIG = 'test\\reference_configs\\hpo.yaml'

# Prepare configuration files for each category
CONFIG_FILES = {
    'preprocessing_config': 'test\\reference_configs\\preprocessing.yaml',
    'dataset_config': 'test\\reference_configs\\dataset.yaml',
    'hpo_config': TEST_HPO_CONFIG,
    'system_config': 'test\\reference_configs\\system.yaml'
}


def run_script_with_configs():
    """
    Helper function to run the script with given configurations.
    """
    command = ['python', SCRIPT_PATH] + [f'--{k}={v}' for k, v in CONFIG_FILES.items()]
    return subprocess.run(command, text=True, capture_output=True)

def test_successful_run():
    """
    Test to verify that the script runs successfully with the original configuration.
    """
    result = run_script_with_configs()
    assert result.returncode == 0, f"Script failed with error: {result.stderr}"

def test_error_handling():
    """
    Test the script's ability to handle erroneous hyperparameter values.
    """
    TEST_HPO_CONFIG = 'test\\reference_configs\\hpo_invalid.yaml'
    CONFIG_FILES = {
    'preprocessing_config': 'test\\reference_configs\\preprocessing.yaml',
    'dataset_config': 'test\\reference_configs\\dataset.yaml',
    'hpo_config': TEST_HPO_CONFIG,
    'system_config': 'test\\reference_configs\\system.yaml'
    }
    command = ['python', SCRIPT_PATH] + [f'--{k}={v}' for k, v in CONFIG_FILES.items()]
    result = subprocess.run(command, text=True, capture_output=True)
    assert result.returncode != 0, "Script should have failed but it didn't."
    assert "Missing" in result.stdout, "Error handling failed."
    
def test_weights_and_biases_integration():
    """
    Test that data is correctly logged to Weights and Biases with correct hyperparameters.
    """
    # Manual inspection of Weights and Biases dashboard currently required
    pass

def test_hyperparameter_application():
    """
    Test that hyperparameters are correctly applied as per configuration during HPO runs.
    """
    command = ['python', SCRIPT_PATH] + [f'--{k}={v}' for k, v in CONFIG_FILES.items()]
    result = subprocess.run(command, text=True, capture_output=True)
    assert "Objective value: 1." in result.stdout, "Check if hyperparameter are applied correctly."