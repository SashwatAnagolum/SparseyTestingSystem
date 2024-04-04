# sparseypy.core.optimizers package

## Submodules

## sparseypy.core.optimizers.hebbian module

Hebbian: file holding the Hebbian optimizer class.

### *class* sparseypy.core.optimizers.hebbian.HebbianOptimizer(model: Module)

Bases: `Optimizer`

#### calculate_freezing_mask(weights, layer_index)

#### step(closure=None)

Performs a weight update.

* **Parameters:**
  **closure** -- callable returning the model output.

## sparseypy.core.optimizers.optimizer_factory module

Optimizer Factory: file holding the Optimizer Factory class.

### *class* sparseypy.core.optimizers.optimizer_factory.OptimizerFactory

Bases: `object`

#### allowed_modules *= {'HebbianOptimizer', 'hebbian'}*

#### *static* create_optimizer(opt_name, \*\*kwargs)

Creates a layer passed in based on the layer name and kwargs.

#### *static* get_optimizer_class(opt_name)

Gets the class corresponding to the name passed in.
Throws an error if the name is not valid.

## Module contents

Init: initialization for te Optimizers module.
