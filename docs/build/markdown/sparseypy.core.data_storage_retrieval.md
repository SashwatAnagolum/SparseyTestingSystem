# sparseypy.core.data_storage_retrieval package

## Submodules

## sparseypy.core.data_storage_retrieval.data_fetcher module

DataFetcher: Fetches data from weights and biases and the database (firestore)

### *class* sparseypy.core.data_storage_retrieval.data_fetcher.DataFetcher

Bases: `object`

A class for fetching data from a Firestore database, including experiment data,
HPO run data, and model weights.

This class provides methods to access and deserialize data related to Sparsey
experiments stored in Firestore. It supports caching for efficient data retrieval.

#### convert_firestore_timestamp(firestore_timestamp: DatetimeWithNanoseconds)

Converts a Firestore DatetimeWithNanoseconds object to a standard Python datetime object.

* **Parameters:**
  **firestore_timestamp** (*DatetimeWithNanoseconds*) -- The Firestore timestamp to convert.
* **Returns:**
  *datetime* -- A standard Python datetime object representing the same point in time.

#### get_evaluation_result(experiment_id: str)

Get the evaluation result for a given experiment.

* **Parameters:**
  **experiment_id** (*str*) -- The ID of the experiment.
* **Returns:**
  *EvaluationResult* -- the EvaluationResult for the experiment of this id in w&b

#### get_hpo_result(hpo_run_id: str)

Retrieves the overall result of a specific hyperparameter optimization (HPO) run.

This method aggregates the results of individual experiments within an HPO run,
and provides a comprehensive view of the HPO run, including start and end times,
configuration settings, and the best-performing experiment.

* **Parameters:**
  **hpo_run_id** (*str*) -- The unique identifier for the HPO run.
* **Returns:**
  *HPOResult* -- An instance of HPOResult containing aggregated results
  and configuration info from the HPO run.

#### get_hpo_step_result(hpo_run_id, experiment_id)

Retrieves the result of a specific experiment step within an HPO run.

This method combines experiment data and HPO configuration to create a comprehensive
step result for hpo.

* **Parameters:**
  * **hpo_run_id** (*str*) -- The unique identifier for the HPO run.
  * **experiment_id** (*str*) -- The unique identifier for the experiment within the HPO run.
* **Returns:**
  *HPOStepResult* -- An instance of HPOStepResult representing the experiment step
  within the HPO run.

#### get_model_weights(model_id: str)

Fetches model weights for a given model ID.

* **Parameters:**
  **model_id** (*str*) -- A unique identifier for the model.
* **Returns:**
  *dict* -- A dictionary of model weights.

#### get_training_result(experiment_id: str, result_type: str = 'training')

Retrieves the training result for a given experiment.

This method compiles the results of individual training steps within an experiment
into a single TrainingResult object. It includes overall metrics, step-by-step results,
and information about the start and end times of the experiment, as well as the
best performing steps.

* **Parameters:**
  **experiment_id** (*str*) -- The unique identifier for the experiment.
* **Returns:**
  *TrainingResult* -- An instance of TrainingResult containing aggregated
  metrics and outcomes from the experiment's training steps.

#### get_training_step_result(experiment_id, step_index)

Retrieves the result of a specific training step within an experiment.

* **Parameters:**
  * **experiment_id** (*str*) -- The unique identifier for the experiment.
  * **step_index** (*int*) -- The index of the training step to retrieve.
* **Returns:**
  *TrainingStepResult* -- An instance of TrainingStepResult containing the step's metrics.
* **Raises:**
  **ValueError** -- If the step index is out of bounds for the given experiment.

## sparseypy.core.data_storage_retrieval.data_storer module

DataStorer: Saves data to Weights & Biases and the system database (Firestore)

### *class* sparseypy.core.data_storage_retrieval.data_storer.DataStorer(metric_config: dict)

Bases: `object`

DataStorer: Stores data to Weights & Biases

#### average_nested_data(data)

Averages an arbitrarily deep data structure
and returns the result as a single value.

Used here to reduce the granularity of data in order
to store a single value for each step in W&B.

* **Parameters:**
  **data** -- the value(s) to reduce
* **Returns:**
  a single value representing the averaged data

#### *static* configure(ds_config: dict)

Configures the DataStorer by logging into Weights & Biases and
initializing its database connection.

Because all configuration is tracked inside firebase_admin and
wandb, calling this method also configures the DataFetcher.

* **Parameters:**
  **ds_config** (*dict*) -- the validated system.yaml configuration

#### create_artifact(content: dict)

Creates a W&B artifact for saving in the database.

Currently unused.

* **Parameters:**
  **content** (*dict*) -- the data to encapsulate in the Artifact
* **Returns:**
  *wandb.Artifact* -- the encapsulated data

#### create_experiment(experiment: [TrainingResult](sparseypy.core.results.md#sparseypy.core.results.training_result.TrainingResult))

Creates a new entry for the current experiment in Firestore.

* **Parameters:**
  * **experiment** (*TrainingResult*) -- the TrainingResult for the new experiment
  * **for which to create a database entry**

#### create_hpo_sweep(sweep: [HPOResult](sparseypy.core.results.md#sparseypy.core.results.hpo_result.HPOResult))

Creates an entry in Firestore for the given HPO sweep.

Stores basic metadata that Weights & Biases tracks automatically
but needs to be manually created in Firestore for other
storage functions (such as save_hpo_step()) to work correctly.

* **Parameters:**
  **sweep** (*HPOResult*) -- the sweep for which to create an entry

#### firestore_config *= {}*

#### is_initialized *= False*

#### save_evaluation_result(result: [TrainingResult](sparseypy.core.results.md#sparseypy.core.results.training_result.TrainingResult))

Saves the summary-level evaluation results for the current run
to Firestore.

Only saves the evaluation summary--you still need to save the individual
evaluation steps by calling save_evaluation_step().

* **Parameters:**
  * **result** (*TrainingResult*) -- the completed evaluation results
  * **to save**

#### save_evaluation_step(parent: str, result: [TrainingStepResult](sparseypy.core.results.md#sparseypy.core.results.training_step_result.TrainingStepResult))

Saves a single evaluation step to Weights & Biases and Firestore.

* **Parameters:**
  * **parent** (*str*) -- the experiment ID to which to log this step
  * **result** (*TrainingStepResult*) -- the step results to save

#### save_hpo_result(result: [HPOResult](sparseypy.core.results.md#sparseypy.core.results.hpo_result.HPOResult))

Saves the final results of an HPO run to Firestore and
marks it as completed.

Includes end times, best run ID, and an ordered list of runs
by objective value.

Does not save the individual steps--you need to use
save_hpo_step() for that.

* **Parameters:**
  * **result** (*HPOResult*) -- the results of the completed HPO sweep to
  * **summarize and save**

#### save_hpo_step(parent: str, result: [HPOStepResult](sparseypy.core.results.md#sparseypy.core.results.hpo_step_result.HPOStepResult))

Saves a single HPO step to Weights & Biases and Firestore.

Saves objective data and HPO configuration to the run in
both Weights & Biases and Firestore.

Also marks this experiment in Firestore as belonging to the
parent sweep and updates its best runs.

* **Parameters:**
  * **parent** (*str*) -- the ID of the parent sweep in the HPO table
  * **that should be updated with this run's results**
  * **result** (*HPOStepResult*) -- the results of the HPO step to save

#### save_model(experiment: str, m: Model)

Saves a model to Weights & Biases.

* **Parameters:**
  * **experiment** (*str*) -- the experiment ID to which the model should be saved
  * **m** (*Model*) -- the model object to be saved

#### save_training_result(result: [TrainingResult](sparseypy.core.results.md#sparseypy.core.results.training_result.TrainingResult))

Saves the summary-level training results for the current run
to Firestore.

Only saves the training summary--you still need to save the individual
training steps by calling save_training_step().

* **Parameters:**
  * **result** (*TrainingResult*) -- the completed training results
  * **to save**

#### save_training_step(parent: str, result: [TrainingStepResult](sparseypy.core.results.md#sparseypy.core.results.training_step_result.TrainingStepResult))

Saves a single training step to Weights & Biases and Firestore.

* **Parameters:**
  * **parent** (*str*) -- the experiment ID to which to log this step
  * **result** (*TrainingStepResult*) -- the step results to save

#### wandb_config *= {}*

## Module contents
