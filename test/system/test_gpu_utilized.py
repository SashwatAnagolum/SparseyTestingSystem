import pytest
import torch
import subprocess
import os
#move to high level if run from lower level
os.chdir(os.path.dirname(__file__))
os.chdir('..')
os.chdir('..')
# Paths to the script and configuration files
SCRIPT_PATH = 'src\\sparseypy\\scripts\\run_hpo.py'
ORIGINAL_HPO_CONFIG = 'test\\reference_configs\\hpo.yaml'
TEST_HPO_CONFIG = 'test\\reference_configs\\hpo.yaml'

# Prepare configuration files for each category
CONFIG_FILES = {
    'preprocessing_config': 'test\\reference_configs\\preprocessing.yaml',
    'dataset_config': 'test\\reference_configs\\dataset.yaml',
    'hpo_config': TEST_HPO_CONFIG,
    'system_config': 'test\\reference_configs\\system.yaml'
}

def check_gpu_usage():
    # This function checks if GPU memory is being used for TC-24-05.
    if torch.cuda.is_available():
        initial_mem = torch.cuda.memory_allocated()
        command = ['python', SCRIPT_PATH] + [f'--{k}={v}' for k, v in CONFIG_FILES.items()]
        subprocess.run(command, text=True, capture_output=True)
        used_mem = torch.cuda.memory_allocated() - initial_mem
        return used_mem > 0
    else:
        return False

@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU is not available")
def test_gpu_usage():
    assert check_gpu_usage(), "GPU is available but not utilized"

@pytest.mark.skipif(torch.cuda.is_available(), reason="GPU is available, this test is for when it's not")
def test_no_gpu_usage():
    # You may want to assert something else here when GPU is not available
    assert not check_gpu_usage(), "GPU should not be utilized"
