# sparseypy.cli.config_validation.saved_schemas.system package

## Submodules

## sparseypy.cli.config_validation.saved_schemas.system.default module

Default System Schema: the schema for system.yaml.

### *class* sparseypy.cli.config_validation.saved_schemas.system.default.DefaultSystemSchema

Bases: [`AbstractSchema`](sparseypy.cli.config_validation.saved_schemas.md#sparseypy.cli.config_validation.saved_schemas.abs_schema.AbstractSchema)

Default System Schema: class for system.yaml schema.

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

#### check_if_db_adapter_exists(db_adapter_name)

Checks if a database adapter exists or not.

* **Returns:**
   *(bool)* -- whether the database adapter exists or not.

#### extract_schema_params(config_info: dict)

Extracts the required schema parameters from the config info dict
in order to build the schema to validate against.

* **Parameters:**
  **config_info** -- a dict containing the config info from the
  user.
* **Returns:**
  a dict (might be None) containing all the required parameters
  : to build the schema.

#### make_env_schema(env_name: str)

Builds a schema that can be used to validate a string that is either
the name of an environment variable (with $ prefix) or a value.

* **Parameters:**
  **env_name** (*str*) -- the value or environment variable name
* **Returns:**
  the value or a Use that can be used to validate the value

## Module contents
