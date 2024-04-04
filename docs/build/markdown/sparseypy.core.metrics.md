# sparseypy.core.metrics package

## Submodules

## sparseypy.core.metrics.basis_average module

Basis Average: file holding the BasisAverageMetric class.

### *class* sparseypy.core.metrics.basis_average.BasisAverageMetric(model: ~torch.nn.modules.module.Module, reduction: str | None = None, best_value: ~typing.Callable | None = <function max_by_layerwise_mean>)

Bases: [`Metric`](#sparseypy.core.metrics.metrics.Metric)

BasisAverageMetric: metric computing the feature
: coverage of MACs and layers in a Sparsey model.

#### reduction

the type of reduction to apply
onto the raw per-layer, per-sample feature coverage
results. Valid options are None and 'sparse'. Choosing
'sparse' will return the raw averaged inputs to each MAC.
Choosing None will return the inputs inserted into
their positions in a tensor of the same size as the
input samples to the model.

* **Type:**
  str

#### hook

the hook registered with the model
being evaluated to obtain references to each layer,
and layerwise inputs and outputs.

* **Type:**
  [LayerIOHook](sparseypy.core.hooks.md#sparseypy.core.hooks.layer_io.LayerIOHook)

#### compute(m: Model, last_batch: Tensor, labels: Tensor, training: bool = True)

Computes the basis average of a model.

* **Parameters:**
  * **m** (*Model*) -- Model to evaluate.
  * **last_batch** (*torch.Tensor*) -- the model input for the current step
  * **labels** (*torch.Tensor*) -- the model output for the current step
  * **training** (*bool*) -- whether the model is training or evaluating
* **Returns:**
   *(list[torch.Tensor])* --

  a list of Tensors containing the average
  : feature that each MAC has seen.

#### get_projected_receptive_fields(layers, input_shape)

Compute the projected receptive fields of each MAC in the model,
i.e. what input elements in each sample can be seen by each MAC.

* **Parameters:**
  * **layers** (*list[list[MAC]]*) -- collection of MACS in the model.
  * **input_shape** (*int*) -- shape of each input sample.

#### initialize_shapes(layers, last_batch: Tensor)

Initialize the shapes of different storage objects in the model
based on the shape of the inputs and the model structure.

* **Parameters:**
  * **layers** (*list[list[MAC]]*) -- collection of MACs making up
    the model.
  * **last_batch** (*torch.Tensor*) -- the last set of inputs shown
    to the model.

## sparseypy.core.metrics.basis_set_size module

### *class* sparseypy.core.metrics.basis_set_size.BasisSetSizeMetric(model: ~torch.nn.modules.module.Module, reduction: str | None = None, best_value: ~typing.Callable | None = <function min_by_layerwise_mean>)

Bases: [`Metric`](#sparseypy.core.metrics.metrics.Metric)

#### compute(m: Model, last_batch: Tensor, labels: Tensor, training: bool = True)

Computes a metric.

* **Parameters:**
  * **m** -- the model currently being trained.
  * **last_batch** -- the inputs to the current batch being evaluated
  * **labels** -- the output from the current batch being evaluated
* **Returns:**
  the Metric's results as a dict.

## sparseypy.core.metrics.basis_set_size_increase module

### *class* sparseypy.core.metrics.basis_set_size_increase.BasisSetSizeIncreaseMetric(model: ~torch.nn.modules.module.Module, reduction: str | None = None, best_value: ~typing.Callable | None = <function min_by_layerwise_mean>)

Bases: [`Metric`](#sparseypy.core.metrics.metrics.Metric)

BasisSetSizeIncreaseMetric: metric to keep track
: of basis set sizes across a Sparsey model.

#### compute(m: Model, last_batch: Tensor, labels: Tensor, training: bool = True)

Computes a metric.

* **Parameters:**
  * **m** -- the model currently being trained.
  * **last_batch** -- the inputs to the current batch being evaluated
  * **labels** -- the output from the current batch being evaluated
* **Returns:**
  the Metric's results as a dict.

## sparseypy.core.metrics.comparisons module

comparsions.py - contains comparison functions for determining the "best" value of a metric

### sparseypy.core.metrics.comparisons.average_nested_data(data)

Averages an arbitrarily deep data structure
and returns the result as a single value.

### sparseypy.core.metrics.comparisons.max_by_layerwise_mean(x, y)

Returns the maximum value by layerwise average of x and y.

### sparseypy.core.metrics.comparisons.min_by_layerwise_mean(x, y)

Returns the minimum value by layerwise average of x and y.

## sparseypy.core.metrics.feature_coverage module

### *class* sparseypy.core.metrics.feature_coverage.FeatureCoverageMetric(model: ~torch.nn.modules.module.Module, reduction: str | None = None, best_value: ~typing.Callable | None = <function max_by_layerwise_mean>)

Bases: [`Metric`](#sparseypy.core.metrics.metrics.Metric)

FeatureCoverageMetric: metric computing the feature
: coverage of MACs and layers in a Sparsey model.

#### reduction

the type of reduction to apply
onto the raw per-layer, per-sample feature coverage
results.

* **Type:**
  str

#### hook

the hook registered with the model
being evaluated to obtain references to each layer,
and layerwise inputs and outputs.

* **Type:**
  [LayerIOHook](sparseypy.core.hooks.md#sparseypy.core.hooks.layer_io.LayerIOHook)

#### compute(m: Model, last_batch: Tensor, labels: Tensor, training: bool = True)

Computes the feature coverage of a model for a given batch of inputs.

* **Parameters:**
  * **m** (*Model*) -- Model to evaluate.
  * **last_batch** (*torch.Tensor*) -- the model input for the current step
  * **labels** (*torch.Tensor*) -- the model output for the current step
  * **training** (*bool*) -- whether the model is training or evaluating

Output:
: (float): feature coverage as a fraction.

## sparseypy.core.metrics.match_accuracy module

### *class* sparseypy.core.metrics.match_accuracy.MatchAccuracyMetric(model: ~torch.nn.modules.module.Module, reduction: str | None = None, best_value: ~typing.Callable | None = <function max_by_layerwise_mean>)

Bases: [`Metric`](#sparseypy.core.metrics.metrics.Metric)

#### compute(m: Model, last_batch: Tensor, labels: Tensor, training: bool = True)

Computes the approximate match accuracy of a model for a given batch of inputs.

* **Parameters:**
  * **m** -- Model to evaluate.
  * **last_batch** -- the model input for the current step (as a Tensor)
  * **labels** -- the model output for the current step (as a Tensor)
  * **training** -- boolean - whether the model is training (store codes)
    or evaluating (determine approximate match
    accuracy using stored codes)

Output:
: approximate match accuracy as a list of accuracies:
  one pertaining to each layer

#### get_normalized_hamming_distance(stored_code: Tensor, selected_code: Tensor)

Computes the normalized hamming distance between two codes.

* **Parameters:**
  * **stored_code** (*list[torch.Tensor]*) -- the code generated for a
    particular input during training
  * **selected_code** (*list[torch.Tensor]*) -- the code generated for
    the same input during evaluation

## sparseypy.core.metrics.metric_factory module

### *class* sparseypy.core.metrics.metric_factory.MetricFactory

Bases: `object`

#### allowed_comparisons *= {'average_nested_data', 'max_by_layerwise_mean', 'min_by_layerwise_mean'}*

#### allowed_modules *= {'BasisAverageMetric', 'BasisSetSizeIncreaseMetric', 'BasisSetSizeMetric', 'FeatureCoverageMetric', 'MatchAccuracyMetric', 'NumActivationsMetric', 'basis_average', 'basis_set_size', 'basis_set_size_increase', 'comparisons', 'feature_coverage', 'match_accuracy', 'metrics', 'num_activations'}*

#### *static* create_metric(metric_name, \*\*kwargs)

Creates a layer passed in based on the layer name and kwargs.

#### *static* get_comparison_function(comparison_name: str)

#### *static* get_metric_class(metric_name)

Gets the class corresponding to the name passed in.
Throws an error if the name is not valid.

#### *static* is_valid_comparision(comparison_name: str)

Checks whether a given comparison function exists.

#### *static* is_valid_metric_class(metric_name: str)

Checks whether a metric class exists corresponding to the passed-in name.

## sparseypy.core.metrics.metrics module

### *class* sparseypy.core.metrics.metrics.Metric(model: Module, name: str, best_comparison: Callable)

Bases: `object`

Metric: a base class for metrics.
: Metrics are used to compute different measurements requested by the user
  to provide estimations of model progress and information
  required for Dr. Rinkus' experiments.

#### *abstract* compute(m: Model, last_batch: Tensor, labels: Tensor, training: bool = True)

Computes a metric.

* **Parameters:**
  * **m** -- the model currently being trained.
  * **last_batch** -- the inputs to the current batch being evaluated
  * **labels** -- the output from the current batch being evaluated
* **Returns:**
  the Metric's results as a dict.

#### get_best_comparison_function()

Returns the function to use to obtain the "best" instance of this metric.

#### get_name()

Returns the name of this metric.

## sparseypy.core.metrics.num_activations module

Num Activations: file holding the NumActivationsMetric class.

### *class* sparseypy.core.metrics.num_activations.NumActivationsMetric(model: ~torch.nn.modules.module.Module, reduction: str | None = None, best_value: ~typing.Callable | None = <function min_by_layerwise_mean>)

Bases: [`Metric`](#sparseypy.core.metrics.metrics.Metric)

NumActivationsMetric: metric computing the number of activations
: across MACs in a Sparsey model.

#### reduction

the type of reduction to apply
onto the raw per-layer, per-sample feature coverage
results.

* **Type:**
  str

#### hook

the hook registered with the model
being evaluated to obtain references to each layer,
and layerwise inputs and outputs.

* **Type:**
  [LayerIOHook](sparseypy.core.hooks.md#sparseypy.core.hooks.layer_io.LayerIOHook)

#### num_activations

the number of activations
fr each MAC in each layer of the model.

* **Type:**
  list[list[int]]

#### compute(m: Model, last_batch: Tensor, labels: Tensor, training: bool = True)

Computes the number of activations of a model for a given batch of inputs.

* **Parameters:**
  * **m** (*Model*) -- Model to evaluate.
  * **last_batch** (*torch.Tensor*) -- the model input for the current step
  * **labels** (*torch.Tensor*) -- the labels for the current step
  * **training** (*bool*) -- whether the model is training or evaluating

Output:
: Union[float | list[float] | list[list[float]]]:
  : the number of activations across MACs in the model

#### initialize_activation_counts(layers)

Initializes the activation counts of the NumActivations object.

* **Parameters:**
  **layers** (*list[list[MAC]]*) -- a list of MACs in the model.

## Module contents
