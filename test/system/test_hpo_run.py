# -*- coding: utf-8 -*-

"""
Test HPO Run Task: tests covering the execution of hpo_run_task.py script 
to ensure it runs without errors.
"""

import pytest

# Import the main function from hpo_run_task.py
# Ensure that hpo_run_task.py is in the Python path
from sparsepy.tasks.run_hpo_task import main

class TestHPORunTask:
    """
    TestHPORunTask: class containing tests to verify the correct execution
    of the hpo_run_task.py script.
    """

    def test_run_hpo_task_no_errors(self) -> None:
        """
        This test calls the main function of hpo_run_task.py and checks
        if it completes without exceptions.
        """
        try:
            config_file_path = 'config.yaml'
            main(config_file_path)
            assert True

        except Exception as e:
            pytest.fail(f"Execution of hpo_run_task.py failed: {e}")
