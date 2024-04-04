# sparseypy.tasks package

## Submodules

## sparseypy.tasks.evaluate_model_task module

## sparseypy.tasks.run_hpo module

Run HPO Task: script to run HPO.

### sparseypy.tasks.run_hpo.run_hpo(hpo_config: dict, dataset_config: dict, preprocessing_config: dict, system_config: dict)

Runs hyperparameter optimization
over the specified network hyperparameters
to optimize for the specified objective.

* **Parameters:**
  * **hpo_config** (*dict*) -- config info used to build the
    HPORun object.
  * **dataset_config** (*dict*) -- config info used to build the
    dataset object.
  * **preprocessing_config** (*dict*) -- config info used to build the
    preprocessing stack.
  * **system_config** (*dict*) -- config info for the overall system

## sparseypy.tasks.train_model module

Train Model: script to train models.

### sparseypy.tasks.train_model.train_model(model_config: dict, trainer_config: dict, preprocessing_config: dict, dataset_config: dict, system_config: dict)

Builds a model using the model_config, and trains
it using the trainer built using trainer_config on
the dataset built using dataset_config, with preprocessing
defined in preprocessing_config.

* **Parameters:**
  * **model_config** (*dict*) -- config info to build the model.
  * **trainer_config** (*dict*) -- config info to build the trainer.
  * **preprocessing_config** (*dict*) -- config info to build the
    preprocessing stack.
  * **dataset_config** (*dict*) -- config info to build the dataset
    to train on.
  * **system_config** (*dict*) -- config info for the overall system

## sparseypy.tasks.visualize_results_task module

## Module contents

Init: initialization for the tasks module.
