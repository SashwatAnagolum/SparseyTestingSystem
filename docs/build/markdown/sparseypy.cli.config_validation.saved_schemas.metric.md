# sparseypy.cli.config_validation.saved_schemas.metric package

## Submodules

## sparseypy.cli.config_validation.saved_schemas.metric.basis_average module

Basis Average: file holding the BasisAverageMetricSchema class.

### *class* sparseypy.cli.config_validation.saved_schemas.metric.basis_average.BasisAverageMetricSchema

Bases: [`AbstractSchema`](sparseypy.cli.config_validation.saved_schemas.md#sparseypy.cli.config_validation.saved_schemas.abs_schema.AbstractSchema)

#### build_schema(schema_params: dict)

Builds a schema that can be used to validate the passed in
config info.

* **Parameters:**
  **schema_params** -- a dict containing all the required
  parameters to build the schema.
* **Returns:**
  a Schema that can be used to validate the config info.

## sparseypy.cli.config_validation.saved_schemas.metric.basis_set_size module

### *class* sparseypy.cli.config_validation.saved_schemas.metric.basis_set_size.BasisSetSizeMetricSchema

Bases: [`AbstractSchema`](sparseypy.cli.config_validation.saved_schemas.md#sparseypy.cli.config_validation.saved_schemas.abs_schema.AbstractSchema)

#### build_schema(schema_params: dict)

Builds a schema that can be used to validate the passed in
config info.

* **Parameters:**
  **schema_params** -- a dict containing all the required
  parameters to build the schema.
* **Returns:**
  a Schema that can be used to validate the config info.

## sparseypy.cli.config_validation.saved_schemas.metric.basis_set_size_increase module

Basis Set Size Increase: file holding the BasisSetSizeIncreaseMetricSchema class.

### *class* sparseypy.cli.config_validation.saved_schemas.metric.basis_set_size_increase.BasisSetSizeIncreaseMetricSchema

Bases: [`AbstractSchema`](sparseypy.cli.config_validation.saved_schemas.md#sparseypy.cli.config_validation.saved_schemas.abs_schema.AbstractSchema)

#### build_schema(schema_params: dict)

Builds a schema that can be used to validate the passed in
config info.

* **Parameters:**
  **schema_params** -- a dict containing all the required
  parameters to build the schema.
* **Returns:**
  a Schema that can be used to validate the config info.

## sparseypy.cli.config_validation.saved_schemas.metric.feature_coverage module

### *class* sparseypy.cli.config_validation.saved_schemas.metric.feature_coverage.FeatureCoverageMetricSchema

Bases: [`AbstractSchema`](sparseypy.cli.config_validation.saved_schemas.md#sparseypy.cli.config_validation.saved_schemas.abs_schema.AbstractSchema)

#### build_schema(schema_params: dict)

Builds a schema that can be used to validate the passed in
config info.

* **Parameters:**
  **schema_params** -- a dict containing all the required
  parameters to build the schema.
* **Returns:**
  a Schema that can be used to validate the config info.

## sparseypy.cli.config_validation.saved_schemas.metric.match_accuracy module

### *class* sparseypy.cli.config_validation.saved_schemas.metric.match_accuracy.MatchAccuracyMetricSchema

Bases: [`AbstractSchema`](sparseypy.cli.config_validation.saved_schemas.md#sparseypy.cli.config_validation.saved_schemas.abs_schema.AbstractSchema)

#### build_schema(schema_params: dict)

Builds a schema that can be used to validate the passed in
config info.

* **Parameters:**
  **schema_params** -- a dict containing all the required
  parameters to build the schema.
* **Returns:**
  a Schema that can be used to validate the config info.

## sparseypy.cli.config_validation.saved_schemas.metric.num_activations module

Num Activations: file holding the NumActivationsMetricSchema class.

### *class* sparseypy.cli.config_validation.saved_schemas.metric.num_activations.NumActivationsMetricSchema

Bases: [`AbstractSchema`](sparseypy.cli.config_validation.saved_schemas.md#sparseypy.cli.config_validation.saved_schemas.abs_schema.AbstractSchema)

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

In this instance, there are no parameters.

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
