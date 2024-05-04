import pytest
import subprocess
import os
import yaml

# Ensure the working directory is high-level if run from a lower level directory
os.chdir(os.path.dirname(__file__))
for _ in range(2):  # Adjust depth as needed to reach the root directory
    os.chdir('..')

# Paths to the script and configuration files
SCRIPT_PATH = 'src/sparseypy/scripts/evaluate_model.py'

# Prepare configuration files for each category
CONFIG_FILES = {
    'preprocessing_config': 'test/reference_configs/preprocessing.yaml',
    'dataset_config': 'test/reference_configs/eval_dataset.yaml',
    'training_recipe_config': 'test/reference_configs/trainer.yaml',
    'system_config': 'test/reference_configs/system.yaml'
}

def run_script_with_configs(model_name="TestModel"):
    """ Helper function to run the evaluation script with configuration files """
    command = [
        "python", SCRIPT_PATH,
        "--model_name", model_name,
        "--training_recipe_config", CONFIG_FILES['training_recipe_config'],
        "--preprocessing_config", CONFIG_FILES['preprocessing_config'],
        "--dataset_config", CONFIG_FILES['dataset_config'],
        "--system_config", CONFIG_FILES['system_config']
    ]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return result

def test_evaluation_metrics_logging():
    """ Test Case ID TC-21-01: Verify evaluation metrics storage """
    # Run the evaluation script using the full configuration
    result = run_script_with_configs()
    assert 'basis_set_size' in result.stdout, "Metrics logging failed or output mismatch"

def test_model_object_usage():
    """ Test Case ID TC-21-02: Confirm that the Model object uses the TrainingRecipe correctly """
    result = run_script_with_configs()
    assert 'basis_set_size' in result.stdout, "Model object usage incorrect"

if __name__ == "__main__":
    pytest.main()
