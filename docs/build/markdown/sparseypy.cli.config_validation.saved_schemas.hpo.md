# sparseypy.cli.config_validation.saved_schemas.hpo package

## Submodules

## sparseypy.cli.config_validation.saved_schemas.hpo.default module

Default HPO Schema: the schema for HPO runs.

### *class* sparseypy.cli.config_validation.saved_schemas.hpo.default.DefaultHpoSchema

Bases: [`AbstractSchema`](sparseypy.cli.config_validation.saved_schemas.md#sparseypy.cli.config_validation.saved_schemas.abs_schema.AbstractSchema)

Default HPO Schema: class for HPO run schemas.

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

#### check_if_model_family_exists(model_family)

Checks if a model family with the name model_family exists.

* **Returns:**
   *(bool)* -- whether the model family exists or not

#### check_optimized_hyperparams_validity(config_info)

Checks whether the config for the hyperparameters to be
optimized is valid or not.

* **Returns:**
   *(bool)* -- whether the config is valid or not.

#### extract_schema_params(config_info: dict)

Extracts the required schema parameters from the config info dict
in order to build the schema to validate against.

* **Parameters:**
  **config_info** -- a dict containing the config info from the
  user.
* **Returns:**
  a dict (might be None) containing all the required parameters
  : to build the schema.

#### get_max_num_layers(num_layers_info: dict)

Get the maximum value that can be assigned to the
num_layers hyperparameter.

* **Returns:**
   *(int)* -- the maximum number of layers possible.

#### has_enough_layer_configs(hyperparams_info: dict, num_layers_required: int)

Checks if the layer configs specified contains
enough layers to allow model generation even if
the model with the maximum number of layers
specified in the hyerparameter ranges is constructed.
:Parameters: \* **hyperparams_info** (*dict*) -- the hyperparams configs

> * **nu_layers_required** (*int*) -- the minimum number
>   of layers required in the config file.
* **Returns:**
   *(bool)* -- whether the model can be constructed or not.

#### validate_metrics_in_order(metrics: list, metric_schemas: list[Schema])

Validates the metrics in the provided list in order to prevent
emitting exceptions.

Currently a bit hacky--if the validation fails then an exception
will be raised and this method will not return. Otherwise if you
reach the return statement all metrics validated successfully.

* **Returns:**
   *(list)* -- validated metric configuration.

## Module contents
