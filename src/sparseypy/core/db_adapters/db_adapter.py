"""
firestore_db_adapter.py - file containing the Firestore database adapter
"""

import abc

import numpy as np

from sparseypy.access_objects.models.model import Model
from sparseypy.core.results import HPOResult, HPOStepResult, TrainingResult, TrainingStepResult

class DbAdapter:
    """
    DbAdapter: a base class for database adapters.
        Database adapters provide a read/write interface to a particular
        database for use as backend storage.

    Attributes:
        config (dict): the validated configuration information.
    """

    def __init__(self, config: dict, metric_config: dict):
        self.config = config
        self.saved_metrics = [metric["name"] for metric in metric_config if metric["save"] is True]


    @abc.abstractmethod
    def create_hpo_sweep(self, sweep: HPOResult):
        """
        Creates an entry in the database for the given HPO sweep.

        Stores basic metadata that Weights & Biases tracks automatically
        but needs to be manually created in database adapters for other
        storage functions (such as save_hpo_step()) to work correctly.

        Args:
            sweep (HPOResult): the sweep for which to create an entry
        """


    @abc.abstractmethod
    def create_experiment(self, experiment: TrainingResult):
        """
        Creates a new entry for the current experiment in the database.
        
        Args:
            experiment (TrainingResult): the TrainingResult for the new experiment
            for which to create a database entry
        """


    @abc.abstractmethod
    def save_model(self, experiment: str, m: Model, model_config: dict, wandb_location: str):
        """
        Saves a model object to the database.

        Args:
            experiment (str): the experiment ID to which the model should be saved
            m (Model): the model object to be saved
            model_config (dict): the configuration file for the model to be saved
            wandb_location (str): the qualified name/location of the model binary in
                Weights & Biases.
        """


    def save_evaluation_step(self, parent: str, result: TrainingStepResult):
        """
        Saves a single evaluation step to the database.

        Args:
            parent (str): the experiment ID to which to log this step
            result (TrainingStepResult): the step results to save
        """
        self.save_training_step(
            parent, result, phase="evaluation"
        )


    @abc.abstractmethod
    def save_training_step(self, parent: str, result: TrainingStepResult,
                           phase: str = "training"):
        """
        Saves a single training step to the database.

        Args:
            parent (str): the experiment ID to which to log this step
            result (TrainingStepResult): the step results to save
            phase (str): the type of step (training/validation/evaluation) to save
        """


    @abc.abstractmethod
    def save_training_result(self, result: TrainingResult):
        """
        Saves the summary-level training results for the current run
        to the database. 

        Only saves the training summary--you still need to save the individual 
        training steps by calling save_training_step().

        Args:
            result (TrainingResult): the completed training results
            to save
        """


    @abc.abstractmethod
    def save_evaluation_result(self, result: TrainingResult):
        """
        Saves the summary-level evaluation results for the current run
        to Firestore. 

        Only saves the evaluation summary--you still need to save the individual 
        evaluation steps by calling save_evaluation_step().

        Args:
            result (TrainingResult): the completed evaluation results
            to save
        """


    @abc.abstractmethod
    def save_hpo_step(self, parent: str, result: HPOStepResult):
        """
        Saves a single HPO step to the database.

        Saves objective data and HPO configuration to the run in
        the database Firestore.

        Also marks this experiment in the database as belonging to the
        parent sweep and updates its best runs.
        
        Args:
            parent (str): the ID of the parent sweep in the HPO table
            that should be updated with this run's results
            result (HPOStepResult): the results of the HPO step to save
        """


    @abc.abstractmethod
    def save_hpo_result(self, result: HPOResult):
        """
        Saves the final results of an HPO run to the database and
        marks it as completed.

        Does not save the individual steps--you need to use
        save_hpo_step() for that.

        Args:
            result (HPOResult): the results of the completed HPO sweep to
            summarize and save
        """


    @abc.abstractmethod
    def get_training_result(
            self,
            experiment_id: str,
            result_type: str = "training"
        ) -> TrainingResult:
        """
        Retrieves the training result for a given experiment.

        This method compiles the results of individual training steps within an experiment 
        into a single TrainingResult object. It includes overall metrics, step-by-step results, 
        and information about the start and end times of the experiment, as well as the 
        best performing steps.

        Args:
            experiment_id (str): The unique identifier for the experiment.

        Returns:
            TrainingResult: An instance of TrainingResult containing aggregated 
            metrics and outcomes from the experiment's training steps.
        """


    @abc.abstractmethod
    def get_training_step_result(
            self,
            experiment_id: str,
            step_index: int,
            result_type: str ="training"
        ) -> TrainingStepResult:
        """
        Retrieves the result of a specific training step within an experiment.

        Args:
            experiment_id (str): The unique identifier for the experiment.
            step_index (int): The index of the training step to retrieve.
            result_type (str): The type of result to retrieve. Defaults to "training".

        Returns:
            TrainingStepResult: An instance of TrainingStepResult containing the step's metrics.

        Raises:
            ValueError: If the step index is out of bounds for the given experiment.
        """


    @abc.abstractmethod
    def get_hpo_step_result(self, hpo_run_id, experiment_id):
        """
        Retrieves the result of a specific experiment step within an HPO run.

        This method combines experiment data and HPO configuration to create a comprehensive
        step result for hpo.

        Args:
            hpo_run_id (str): The unique identifier for the HPO run.
            experiment_id (str): The unique identifier for the experiment within the HPO run.

        Returns:
            HPOStepResult: An instance of HPOStepResult representing the experiment step 
            within the HPO run.
        """


    @abc.abstractmethod
    def get_hpo_result(self, hpo_run_id: str) -> HPOResult:
        """
        Retrieves the overall result of a specific hyperparameter optimization (HPO) run.

        This method aggregates the results of individual experiments within an HPO run, 
        and provides a comprehensive view of the HPO run, including start and end times, 
        configuration settings, and the best-performing experiment.

        Args:
            hpo_run_id (str): The unique identifier for the HPO run.

        Returns:
            HPOResult: An instance of HPOResult containing aggregated results 
            and configuration info from the HPO run.
        """


    def get_evaluation_result(self, experiment_id: str) -> TrainingResult:
        """
        Get the evaluation result for a given experiment.

        Args:
            experiment_id (str): The ID of the experiment.

        Returns:
            EvaluationResult: the EvaluationResult for the experiment of this id in w&b
        """
        return self.get_training_result(experiment_id=experiment_id, result_type="evaluation")


    def get_evaluation_step_result(
            self,
            experiment_id: str,
            step_index: int
        ) -> TrainingStepResult:
        """
        Retrieves the result of a specific evaluation step within an experiment.

        Args:
            experiment_id (str): The unique identifier for the experiment.
            step_index (int): The index of the training step to retrieve.

        Returns:
            TrainingStepResult: An instance of TrainingStepResult containing the 
                evaluation step's metrics.

        Raises:
            ValueError: If the step index is out of bounds for the given experiment.
        """
        return self.get_training_step_result(
            experiment_id=experiment_id,
            step_index=step_index,
            result_type="evaluation"
        )


    def get_config(self):
        """
        Returns the configuration data of this database adapter.
        """
        return self.config


    def average_nested_data(self, data):
        """
        Averages an arbitrarily deep data structure
        and returns the result as a single value.

        Used here to reduce the granularity of data in order
        to store a single value for each step in W&B.

        Args:
            data: the value(s) to reduce
        Returns:
            a single value representing the averaged data
        """
        if isinstance(data, list):
            if len(data) == 0:
                data=[0]
            ret = np.mean(np.nan_to_num([self.average_nested_data(item) for item in data]))
        elif hasattr(data, 'tolist'):  # numpy array
            if len(data) == 0:
                data=[0]
            ret = np.mean(np.nan_to_num(data))
        else:
            # Scalar value
            ret = data

        return ret.item() if isinstance(ret, np.generic) else ret
