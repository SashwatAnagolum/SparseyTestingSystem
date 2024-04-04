# sparseypy.access_objects.hpo_runs package

## Submodules

## sparseypy.access_objects.hpo_runs.hpo_run module

HPO Run: file holding the HPORun class.

### *class* sparseypy.access_objects.hpo_runs.hpo_run.HPORun(hpo_config: dict, dataset_config: dict, preprocessing_config: dict, system_config: dict)

Bases: `object`

HPORun: class for performing HPO Runs.

#### num_steps_to_perform

the total number of
candidates to try out during the HPO process

* **Type:**
  int

#### check_is_value_constraint(config)

Checks if a piece of config is a constraint set for
: a hyperparameter.

* **Parameters:**
  **config** (*dict*) -- the piece of config to check.
* **Returns:**
   *(bool)* --

  whether config is a constraint set for
  : a hyperparameter or not.

#### *classmethod* close_tqdm()

Closes the tqdm progress bar, if it exists.

#### construct_sweep_config(hpo_config: dict)

Construct the sweep configuration for the Weights and Biases
sweep to be performed as part of the HPO run.

* **Parameters:**
  **hpo_config** (*dict*) -- configuration info for the HPO run.
* **Returns:**
   *(dict)* -- the WandB sweep configuration.

#### extract_hyperparams(hyperparams_config: dict)

Extract the hyperparameters for a WandB sweep.

* **Parameters:**
  **hyperparams_config** (*dict*) -- config information for the
  network hyperparameters
* **Returns:**
   *(dict)* --

  parameter information required for the WandB sweep
  : construction.

#### generate_model_config(wandb_config: dict)

Generate the model configuration for the next run to be performed as part of the sweep.

* **Parameters:**
  **wandb_config** (*dict*) -- the Weights & Biases configuration for the current run in the sweep
* **Returns:**
  *dict* -- the model configuration in the system format

#### generate_trainer_config(wandb_config: dict)

Generate the trainer configuration for the next run to be performed as part of the sweep.

* **Parameters:**
  **wandb_config** (*dict*) -- the Weights & Biases configuration for the current run in the sweep
* **Returns:**
  *dict* -- the trainer configuration in the system format

#### run_sweep()

Run the HPO process based on the WandB sweep config created.

#### step()

Perform one HPO step, which includes sampling
a set of model hyperparameters, training the created
model, and computing the user-specified objective function
using the trained model.

#### tqdm_bar *= None*

## Module contents
