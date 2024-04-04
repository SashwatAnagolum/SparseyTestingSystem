# sparseypy.cli.config_validation.saved_schemas.optimizer package

## Submodules

## sparseypy.cli.config_validation.saved_schemas.optimizer.hebbian module

Hebbian Optimizer Schema: the schema for Sparsey trainer config files.

### *class* sparseypy.cli.config_validation.saved_schemas.optimizer.hebbian.HebbianOptimizerSchema

Bases: [`AbstractSchema`](sparseypy.cli.config_validation.saved_schemas.md#sparseypy.cli.config_validation.saved_schemas.abs_schema.AbstractSchema)

HebbianOptimizerSchema: schema for hebbian optimizers.

#### build_schema(schema_params: dict)

Builds a schema that can be used to validate the passed in
config info.

* **Parameters:**
  **schema_params** -- a dict containing all the required
  parameters to build the schema.
* **Returns:**
  a Schema that can be used to validate the config info.

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

## Module contents
