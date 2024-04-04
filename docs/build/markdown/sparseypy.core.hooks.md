# sparseypy.core.hooks package

## Submodules

## sparseypy.core.hooks.hook module

Hook: file hlding the abstract Hook class represeting
: a PyTorch hook.

### *class* sparseypy.core.hooks.hook.Hook(module: Module)

Bases: `object`

Hook: abstract base class for all Hooks in the system.

#### *abstract* hook()

Register this hook with the model pased in during initialization.

Concrete hooks need to implement this method to register
the required hooks.

#### remove()

Remove the hooks set by the class.

## sparseypy.core.hooks.hook_factory module

Hook Factory: factory for creating model hooks.

### *class* sparseypy.core.hooks.hook_factory.HookFactory

Bases: `object`

Hook Factory: Factory for generating hooks.

#### *static* create_hook(hook_name: str, model: Module)

Creates a new hook based on the name passed in, and initializes
the hook with the model passed in.

* **Parameters:**
  * **hook_name** (*str*) -- name of the hook to create
  * **model** (*torch.nn.Module*) -- model to initialize the hook with.

## sparseypy.core.hooks.layer_io module

Layer IO: file hlding the LayerIOHook class.

### *class* sparseypy.core.hooks.layer_io.LayerIOHook(module: Module, flatten=False)

Bases: [`Hook`](#sparseypy.core.hooks.hook.Hook)

Layer IO Hook: simple hook to get the output
: and input of a layer.

#### forward_hook(module: Module, input: Tensor, output: Tensor)

Call the hook.

* **Parameters:**
  * **module** (*torch.nn.Module*) -- the module that the hook was
    registered to.
  * **input** (*torch.Tensor*) -- module input
  * **output** (*torch.Tensor*) -- module output

#### get_layer_io()

#### hook()

Register this hook with the model pased in during initialization.

Concrete hooks need to implement this method to register
the required hooks.

#### pre_hook(module: Module, input: Tensor)

## Module contents
