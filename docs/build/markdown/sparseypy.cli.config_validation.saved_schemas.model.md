# sparseypy.cli.config_validation.saved_schemas.model package

## Submodules

## sparseypy.cli.config_validation.saved_schemas.model.sparsey module

Sparsey Model Schema: the schema for Sparsey model config files.

### *class* sparseypy.cli.config_validation.saved_schemas.model.sparsey.SparseyModelSchema

Bases: [`AbstractSchema`](sparseypy.cli.config_validation.saved_schemas.md#sparseypy.cli.config_validation.saved_schemas.abs_schema.AbstractSchema)

SparseyModelSchema: schema for Sparsey networks.

#### build_schema(schema_params: dict)

Builds a schema that can be used to validate the passed in
config info.

* **Parameters:**
  **schema_params** -- a dict containing all the required
  parameters to build the schema.
* **Returns:**
  a Schema that can be used to validate the config info.

#### check_if_hook_exists(hook_name)

Checks if a hook exists.

* **Parameters:**
  **hook_name** (*str*) -- name of the hook
* **Returns:**
   *(bool)* -- whether the hook exists in the system or not.

#### compute_factor_pair(num: int)

Returns the pair of factors whose product is num
whose elements are closest to sqrt(num)

* **Parameters:**
  **num** (*int*) -- the number to find the factors of.
* **Returns:**
   *(Tuple[int, int])* -- the chosen factors

#### compute_grid_size(num_macs: int)

Finds the smallest grid with at least 2 rows
that can accomodate num_macs.

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
  dict containing the transformed config info

## Module contents

Init: initialization for the Model sub-package.
