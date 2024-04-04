# sparseypy.core.model_layers package

## Submodules

## sparseypy.core.model_layers.layer_factory module

Layer Factory: file holding the Layer Factory class.

### *class* sparseypy.core.model_layers.layer_factory.LayerFactory

Bases: `object`

#### allowed_modules *= {'SparseyLayer', 'sparsey_layer'}*

#### *static* create_layer(layer_name, \*\*kwargs)

Creates a layer passed in based on the layer name and kwargs.

#### *static* get_layer_class(layer_name)

Gets the class corresponding to the name passed in.
Throws an error if the name is not valid.

## sparseypy.core.model_layers.sparsey_layer module

Sparsey Layer: code for building and using individual layers
: in a Sparsey model.

### *class* sparseypy.core.model_layers.sparsey_layer.MAC(num_cms: int, num_neurons: int, input_filter: Tensor, num_cms_per_mac_in_input: int, num_neurons_per_cm_in_input: int, layer_index: int, mac_index: int, sigmoid_lambda: float, sigmoid_phi: float, permanence: float, activation_threshold_min: float, activation_threshold_max: float, sigmoid_chi: float, min_familiarity: float)

Bases: `Module`

MAC: class to represent macrocolumns in the Sparsey model.

#### weights

torch.Tensor containing the weights for the MAC.

#### input_filter

torch.Tensor containing the indices of the
MACs in the previous layer that are in the receptive field
of the current MAC.

#### forward(x: Tensor)

Passes data through a MAC.

* **Parameters:**
  * **x** -- torch.Tensor of size (
    batch_size,
    num_macs_in_prev_layer,
    prev_layer_num_cms_per_mac,
    prev_layer_num_neurons_per_cm
  * **) with dtype torch.float32**
* **Returns:**
  torch.Tensor of size (
  : batch_size,
    num_cms_per_mac,
    num_neurons_per_cm

  ) with dtype torch.float32

#### get_input_filter()

Returns the input filter for the MAC.

#### train(mode: bool = True)

Sets the module in training mode.

This has any effect only on certain modules. See documentations of
particular modules for details of their behaviors in training/evaluation
mode, if they are affected, e.g. `Dropout`, `BatchNorm`,
etc.

* **Parameters:**
  **mode** (*bool*) -- whether to set training mode (`True`) or evaluation
  mode (`False`). Default: `True`.
* **Returns:**
  *Module* -- self

### *class* sparseypy.core.model_layers.sparsey_layer.SparseyLayer(autosize_grid: bool, grid_layout: str, num_macs: int, num_cms_per_mac: int, num_neurons_per_cm: int, mac_grid_num_rows: int, mac_grid_num_cols: int, mac_receptive_field_radius: float, prev_layer_num_cms_per_mac: int, prev_layer_num_neurons_per_cm: int, prev_layer_mac_grid_num_rows: int, prev_layer_mac_grid_num_cols: int, prev_layer_num_macs: int, prev_layer_grid_layout: str, layer_index: int, sigmoid_phi: float, sigmoid_lambda: float, saturation_threshold: float, permanence: float, activation_threshold_min: float, activation_threshold_max: float, min_familiarity: float, sigmoid_chi: float)

Bases: `Module`

SparseyLayer: class representing layers in the Sparsey model.

#### num_macs

int containing the number of macs in the layer.

#### receptive_field_radius

float containing the radius of the
receptive field for the MAC.

#### mac_positions

list[Tuple[int, int]] containing the positions
of each MAC in the layer on the grid.

#### input_list

list[list[int]] cotaining the indices of the
MACs in the previous layer within the receptive field of
each MAC in this layer.

#### mac_list

list[MAC] containing all the MACs in this layer.

#### sigmoid_lambda

parameter for the familiarity computation.

* **Type:**
  float

#### sigmoid_phi

parameter for the familiarity computation.

* **Type:**
  float

#### activation_thresholds

a list
of lists containing activation thresholds for each MAC in
the Sparsey layer.

* **Type:**
  list[list[Or[int, float]]]

#### compute_mac_positions(num_macs: int, mac_grid_num_rows: int, mac_grid_num_cols: int, grid_layout: str)

Computes the positions of each MAC in this layer.

* **Parameters:**
  * **num_macs** -- int representing the number of macs in the layer.
  * **mac_grid_num_rows** -- int representing the number of rows
    in the grid for this layer.
  * **mac_grid_num_cols** -- int representing the number of columns
    in the grid for this layer.
  * **grid_layout** -- the type of grid layout (rectangular or hexagonal)
    for the layer.
* **Returns:**
   *(list[Tuple(int, int)])* -- the positions of all MACs in the layer.

#### find_connected_macs_in_prev_layer(mac_positions: list[Tuple[float, float]], prev_layer_mac_positions: list[Tuple[float, float]])

Finds the list of connected MACs in the previous layer
for each MAC in the current layer.

* **Parameters:**
  * **mac_positions** (*list[Tuple[int, int]]*) -- list
    of positions of MACs in the current layer.
  * **prev_layer_mac_positions** (*list[Tuple[int, int]]*) -- list of positions of MACs in the previous layer.
* **Returns:**
   *(list[torch.Tesnor])* --

  list of tensors containing the indices
  : of connected MACs from the previous layer for each
    MAC in the current layer.

#### forward(x: Tensor)

Passes data through a Sparsey layer.

* **Parameters:**
  * **x** -- torch.Tensor of size (
    batch_size,
    prev_layer_mac_grid_num_rows,
    prev_layer_mac_grid_num_cols,
    prev_layer_num_cms_per_mac,
    prev_layer_num_neurons_per_cm
  * **) of dtype torch.float32**
* **Returns:**
  torch.Tensor of size (
  : batch_size,
    mac_grid_num_rows,
    mac_grid_num_cols,
    num_cms_per_mac,
    num_neurons_per_cm

  ) of dtype torch.float32

#### get_macs()

Returns the MACs making up the layer.

* **Returns:**
   *(list[MAC])* -- the MACs making up the layer.

## Module contents

Init: initialization for the model_layers subpackage.
