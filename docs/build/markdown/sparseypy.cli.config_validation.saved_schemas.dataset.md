# sparseypy.cli.config_validation.saved_schemas.dataset package

## Submodules

## sparseypy.cli.config_validation.saved_schemas.dataset.built_in module

Named dataset schema: the schema for named dataset config files.

Does NOT correspond to a NamedDataset class.

### *class* sparseypy.cli.config_validation.saved_schemas.dataset.built_in.BuiltInDatasetSchema

Bases: [`AbstractSchema`](sparseypy.cli.config_validation.saved_schemas.md#sparseypy.cli.config_validation.saved_schemas.abs_schema.AbstractSchema)

BuiltInDatasetSchema: schema for built-in datasets (created by name, e.g. MNIST).

#### build_schema(schema_params: dict)

Builds a schema that can be used to validate the passed in
config info.

* **Parameters:**
  **schema_params** -- a dict containing all the required
  parameters to build the schema.
* **Returns:**
  a Schema that can be used to validate the config info.

#### check_if_dataset_exists(dataset_name: str)

Checks if a builtin torchvision dataset exists with the
name specified.

* **Parameters:**
  **dataset_name** (*str*) -- the name of the dataset.
* **Returns:**
   *(bool)* -- whether the dataset exists or not.

#### check_if_transform_exists(transform_name)

Checks if a (Torchvision v2) transform with the name transform_name exists.

* **Parameters:**
  **transform_name** (*str*) -- the name of the transform to check.
* **Returns:**
   *(bool)* -- whether the transform exists or not

#### convert_transform_name(transform_name: str)

Converts the transform name from the format used in
the dataset config file to the naming format used by PyTorch.

* **Parameters:**
  **transform_name** (*str*) -- the name of the transform.
* **Returns:**
   *(str)* -- the converted transform name.

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

## sparseypy.cli.config_validation.saved_schemas.dataset.image module

Image dataset schema: the schema for Image dataset config files.

### *class* sparseypy.cli.config_validation.saved_schemas.dataset.image.ImageDatasetSchema

Bases: [`AbstractSchema`](sparseypy.cli.config_validation.saved_schemas.md#sparseypy.cli.config_validation.saved_schemas.abs_schema.AbstractSchema)

SparseyTrainerSchema: schema for Sparsey trainers.

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

## sparseypy.cli.config_validation.saved_schemas.dataset.sparsey module

Image dataset schema: the schema for Image dataset config files.

### *class* sparseypy.cli.config_validation.saved_schemas.dataset.sparsey.SparseyDatasetSchema

Bases: [`AbstractSchema`](sparseypy.cli.config_validation.saved_schemas.md#sparseypy.cli.config_validation.saved_schemas.abs_schema.AbstractSchema)

SparseyDatasetSchema: schema for a Sparsey-style raw binary dataset.

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
