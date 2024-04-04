# sparseypy.cli.config_validation.saved_schemas.preprocessing_stack package

## Submodules

## sparseypy.cli.config_validation.saved_schemas.preprocessing_stack.default module

Default Preprocessing stack schema: file holding the schema for default
: preprocessing stacks.

### *class* sparseypy.cli.config_validation.saved_schemas.preprocessing_stack.default.DefaultPreprocessingStackSchema

Bases: [`AbstractSchema`](sparseypy.cli.config_validation.saved_schemas.md#sparseypy.cli.config_validation.saved_schemas.abs_schema.AbstractSchema)

Default Preprocessing Stack Schema: class for preprocessing stack schemas.

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

#### check_if_transform_exists(transform_name)

Checks if a model family with the name model_family exists.

* **Parameters:**
  **transform_name** (*str*) -- the name of the transform to check.
* **Returns:**
   *(bool)* -- whether the model famly exists or not

#### check_transform_schema_validity(transform_configs: dict, transform_schemas: list[Schema])

Checks if all of the transform config information
if valid or not.

* **Parameters:**
  * **transform_configs** (*dict*) -- the transform config
    information to check
  * **transofmr_schemas** (*list[Schema]*) -- the schemas
    to validate against.
* **Returns:**
   *(bool)* --

  whether all of the transform configs
  : are valid or not.

#### extract_schema_params(config_info: dict)

Extracts the required schema parameters from the config info dict
in order to build the schema to validate against.

* **Parameters:**
  **config_info** -- a dict containing the config info from the
  user.
* **Returns:**
  a dict (might be None) containing all the required parameters
  : to build the schema.

## Module contents
