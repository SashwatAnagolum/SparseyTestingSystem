# sparseypy.cli.config_validation.saved_schemas.training_recipe package

## Submodules

## sparseypy.cli.config_validation.saved_schemas.training_recipe.sparsey module

Sparsey Trainer Schema: the schema for Sparsey trainer config files.

### *class* sparseypy.cli.config_validation.saved_schemas.training_recipe.sparsey.SparseyTrainingRecipeSchema

Bases: [`AbstractSchema`](sparseypy.cli.config_validation.saved_schemas.md#sparseypy.cli.config_validation.saved_schemas.abs_schema.AbstractSchema)

SparseyTrainerSchema: schema for Sparsey trainers.

#### build_precheck_schema()

Builds the precheck schema for the config information
passed in by the user. This is used to verify that all parameters
can be collected in order to build the actual schema that will
be used to verify the entire configuration passed in by the
user.

* **Returns:**
   *(Schema)* -- the precheck schema.

#### build_schema(schema_params: dict)

Builds a schema that can be used to validate the passed in
config info.

* **Parameters:**
  **schema_params** -- a dict containing all the required
  parameters to build the schema.
* **Returns:**
  a Schema that can be used to validate the config info.

#### check_if_metric_exists(metric_name)

Checks if a metric exists or not.

* **Returns:**
   *(bool)* -- whether the metric exists or not.

#### check_if_optimizer_exists(optimizer_name)

Checks if the optimizer with optimizer_name exists or not.

* **Parameters:**
  **optimizer_name** (*str*) -- the name of the optimizer.
* **Returns:**
   *(bool)* -- whether the optimizer exists or not.

#### extract_schema_params(config_info: dict)

Extracts the required schema parameters from the config info dict
in order to build the schema to validate against.

* **Parameters:**
  **config_info** -- a dict containing the config info from the
  user.
* **Returns:**
  a dict (might be None) containing all the required parameters
  : to build the schema.

#### transform_schema(config_info: dict)

Transforms the config info passed in by the user to
construct the config information required by the model builder.

* **Parameters:**
  **config_info** -- dict containing the config information
* **Returns:**
   *(dict)* -- the transformed config info

#### validate_metrics_in_order(metrics: list, metric_schemas: list[Schema])

Validates the metrics in the provided list in order to prevent
emitting exceptions.

Currently a bit hacky--if the validation fails then an exception
will be raised and this method will not return. Otherwise if you
reach the return statement all metrics validated successfully.

* **Returns:**
   *(list)* -- validated metric configuration.

## Module contents

Init: initialization for the Trainer sub-package.
