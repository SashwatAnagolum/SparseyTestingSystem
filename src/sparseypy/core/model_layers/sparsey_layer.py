 # -*- coding: utf-8 -*-

"""
Sparsey Layer: code for building and using individual layers
    in a Sparsey model.
"""


from typing import List, Tuple

import torch
from torch.distributions.categorical import Categorical


class SparseyLayer(torch.nn.Module):
    """
    SparseyLayer: class representing layers in the Sparsey model.

    Attributes:
        num_macs: int containing the number of macs in the layer.
        receptive_field_radius: float containing the radius of the 
            receptive field for the MAC.
        mac_positions: list[Tuple[int, int]] containing the positions
            of each MAC in the layer on the grid.
        input_list: list[list[int]] cotaining the indices of the 
            MACs in the previous layer within the receptive field of
            each MAC in this layer.
        mac_list: list[MAC] containing all the MACs in this layer.
        sigmoid_lambda (float): parameter for the familiarity computation.
        sigmoid_phi (float): parameter for the familiarity computation.
        activation_thresholds (list[list[Or[int, float]]]): a list
            of lists containing activation thresholds for each MAC in
            the Sparsey layer.
    """
    def __init__(self, autosize_grid: bool, grid_layout: str,
        num_macs: int, num_cms_per_mac: int, num_neurons_per_cm: int,
        mac_grid_num_rows: int, mac_grid_num_cols: int,
        mac_receptive_field_size: float,
        prev_layer_num_cms_per_mac: int,
        prev_layer_num_neurons_per_cm: int,
        prev_layer_mac_grid_num_rows: int,
        prev_layer_mac_grid_num_cols: int,
        prev_layer_num_macs: int, prev_layer_grid_layout: str,
        layer_index: int,
        sigmoid_phi: float, sigmoid_lambda: float,
        saturation_threshold: float,
        permanence_steps: float, permanence_convexity: float,
        activation_threshold_min: float,
        activation_threshold_max: float,
        min_familiarity: float, sigmoid_chi: float,
        device: torch.device):
        """
        Initializes the SparseyLayer object.
        Args:
            autosize_grid (bool): whether the grid should be autosized.
            grid_layout (str): the layout of the grid (rectangular or hexagonal).
            num_macs (int): the number of MACs in the layer.
            num_cms_per_mac (int): the number of CMs in each MAC.
            num_neurons_per_cm (int): the number of neurons in each CM.
            mac_grid_num_rows (int): the number of rows in the grid.
            mac_grid_num_cols (int): the number of columns in the grid.
            mac_receptive_field_size (float): the size of the receptive field.
            prev_layer_num_cms_per_mac (int): the number of CMs per MAC in the
                previous layer.
            prev_layer_num_neurons_per_cm (int): the number of neurons per CM
                in the previous layer.
            prev_layer_mac_grid_num_rows (int): the number of rows in the grid
                for the previous layer.
            prev_layer_mac_grid_num_cols (int): the number of columns in the grid
                for the previous layer.
            prev_layer_num_macs (int): the number of MACs in the previous layer.
            prev_layer_grid_layout (str): the layout of the grid for the previous
                layer.
            layer_index (int): the index of the layer.
            sigmoid_phi (float): the phi parameter for the sigmoid function.
            sigmoid_lambda (float): the lambda parameter for the sigmoid function.
            saturation_threshold (float): the threshold for saturation.
            permanence_steps (float): the number of permanence steps.
            permanence_convexity (float): the convexity of the permanence.
            activation_threshold_min (float): the minimum activation threshold.
            activation_threshold_max (float): the maximum activation threshold.
            min_familiarity (float): the minimum familiarity.
            sigmoid_chi (float): the chi parameter for the sigmoid function.
            device (torch.device): the device to run the model on.
        """
        super().__init__()

        self.device = device
        self.layer_index = layer_index
        self.is_grid_autosized = autosize_grid
        self.num_macs = num_macs
        self.num_cms_per_mac = num_cms_per_mac
        self.num_neurons_per_cm = num_neurons_per_cm
        self.min_familiarity = min_familiarity
        self.sigmoid_phi = sigmoid_phi
        self.sigmoid_lambda = sigmoid_lambda
        self.sigmoid_chi = sigmoid_chi
        self.permanence_steps = permanence_steps
        self.permanence_convexity = permanence_convexity
        self.saturation_threshold = saturation_threshold
        self.is_active = None
        self.receptive_field_size = mac_receptive_field_size

        self.grid_size = (
            mac_grid_num_rows,
            mac_grid_num_cols
        )

        self.prev_layer_grid_size = (
            prev_layer_mac_grid_num_rows,
            prev_layer_mac_grid_num_cols
        )

        self.prev_layer_output_shape = (
            prev_layer_num_macs,
            prev_layer_num_cms_per_mac * prev_layer_num_neurons_per_cm
        )

        self.mac_positions = self.compute_mac_positions(
            num_macs, mac_grid_num_rows, mac_grid_num_cols,
            grid_layout
        )

        prev_layer_mac_positions = self.compute_mac_positions(
            prev_layer_num_macs, prev_layer_mac_grid_num_rows,
            prev_layer_mac_grid_num_cols, prev_layer_grid_layout
        )

        (
            self.input_connections, mac_rf_sizes
        ) = self.find_connected_macs_in_prev_layer(
            self.mac_positions, prev_layer_mac_positions, prev_layer_num_macs
        )

        self.receptive_field_num_macs = self.input_connections.shape[1]

        self.activation_threshold_min = torch.mul(
            mac_rf_sizes,
            activation_threshold_min * prev_layer_num_cms_per_mac
        ).unsqueeze(-1).unsqueeze(0)

        self.activation_threshold_max = torch.mul(
            mac_rf_sizes,
            activation_threshold_max * prev_layer_num_cms_per_mac
        ).unsqueeze(-1).unsqueeze(0)

        self.weights = torch.nn.Parameter(
            torch.zeros(
                self.num_macs,
                self.receptive_field_num_macs *
                prev_layer_num_cms_per_mac *
                prev_layer_num_neurons_per_cm,
                self.num_cms_per_mac * self.num_neurons_per_cm,
                dtype=torch.float32, device=self.device,
                requires_grad=False
            )
        )


    def compute_mac_positions(
        self, num_macs: int, mac_grid_num_rows: int,
        mac_grid_num_cols: int,
        grid_layout: str) -> List[Tuple[float, float]]:
        """
        Computes the positions of each MAC in this layer.

        Args:
            num_macs: int representing the number of macs in the layer.
            mac_grid_num_rows: int representing the number of rows
                in the grid for this layer.
            mac_grid_num_cols: int representing the number of columns
                in the grid for this layer.   
            grid_layout: the type of grid layout (rectangular or hexagonal)
                for the layer.   

        Returns:
            (list[Tuple(int, int)]): the positions of all MACs in the layer.      
        """
        mac_positions = []
        global_col_offset = 0.5 if grid_layout == 'hex' else 0

        grid_col_spacing = 0.0

        if mac_grid_num_rows == 1:
            row_locations = [0.5]
        else:
            grid_row_spacing = 1 / (mac_grid_num_rows - 1)

            row_locations = [
                i * grid_row_spacing
                for i in range(mac_grid_num_rows)
            ]

        if mac_grid_num_cols == 1:
            col_locations = [0.5]
        else:
            grid_col_spacing = 1 / (mac_grid_num_cols - 1)

            col_locations = [
                i * grid_col_spacing
                for i in range(mac_grid_num_cols)
            ]

        for i in range(num_macs):
            mac_positions.append(
                (
                    row_locations[i // mac_grid_num_cols],
                    col_locations[i % mac_grid_num_cols] + (
                        global_col_offset * (
                            (i % mac_grid_num_rows) % 2
                        ) * grid_col_spacing
                    )
                 )
            )

        return mac_positions


    def _compute_distance(self,
        position_1: Tuple[float, float],
        position_2: Tuple[float, float]) -> float:
        """
        Computes the Euclidean distance between two positions.

        Args:
            position_1 (Tuple[int, int]): x and y coordinates of the
                first point.
        """
        return (
            abs(position_1[0] - position_2[0]) ** 2 +
            abs(position_1[1] - position_2[1]) ** 2
        ) ** 0.5


    def find_connected_macs_in_prev_layer(
        self, mac_positions: list[Tuple[float, float]],
        prev_layer_mac_positions: list[Tuple[float, float]],
        prev_layer_num_macs: int
    ) -> list[torch.Tensor]:
        """
        Finds the list of connected MACs in the previous layer
        for each MAC in the current layer.

        Args:
            mac_positions (list[Tuple[int, int]]): list
                of positions of MACs in the current layer.
            prev_layer_mac_positions (list[Tuple[int, int]]):
                list of positions of MACs in the previous layer.
            prev_layer_num_macs (int): the number of MACS
                in the previous layer.

        Returns:
            torch.tensor: list of tensors containing the indices
                of connected MACs from the previous layer for each
                MAC in the current layer.
        """
        connections = []
        mac_rf_sizes = []
        max_len = 0

        for mac_position in mac_positions:
            mac_connections = []

            for (
                index, prev_layer_mac_position
            ) in enumerate(prev_layer_mac_positions):
                if self._compute_distance(
                    mac_position,
                    prev_layer_mac_position
                ) <= self.receptive_field_size:
                    mac_connections.append(index)

            connections.append(mac_connections)
            max_len = max(max_len, len(mac_connections))

        for i in range(len(connections)):
            mac_rf_sizes.append(len(connections[i]))

            while len(connections[i]) < max_len:
                connections[i].append(prev_layer_num_macs)

        return torch.tensor(
            connections, dtype=torch.long, device=self.device
        ), torch.tensor(
            mac_rf_sizes, dtype=torch.float32, device=self.device
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Passes data through a Sparsey layer.

        Args:
            x: torch.Tensor of size (
                batch_size,
                prev_layer_num_macs,
                prev_layer_num_cms_per_mac *
                prev_layer_num_neurons_per_cm
            ) of dtype torch.float32

        Returns:
            torch.Tensor of size (
                batch_size,
                num_macs,
                num_cms_per_mac * num_neurons_per_cm
            ) of dtype torch.float32
        """
        if tuple(x.shape[1:]) != self.prev_layer_output_shape:
            raise ValueError(
                'Input shape is incorrect! '
                f'Expected shape {self.prev_layer_output_shape} but received '
                f'{tuple(x.shape[1:])} instead.'    
            )

        x = torch.cat(
            (
                x,
                torch.zeros(
                    (x.shape[0], 1, *self.prev_layer_output_shape[1:]),
                    dtype=torch.float32,
                    device=self.device
                )
            ), dim=1
        )

        batch_size = x.shape[0]

        with torch.no_grad():
            mac_inputs = x[:, self.input_connections].view(
                batch_size, self.num_macs, -1
            )

            num_active_inputs = torch.sum(mac_inputs, dim=2, keepdim=True)

            macs_are_active = torch.logical_and(
                torch.ge(
                    num_active_inputs,
                    self.activation_threshold_min
                ),
                torch.le(
                    num_active_inputs,
                    self.activation_threshold_max
                )
            )

            self.is_active = macs_are_active.view(batch_size, self.num_macs)

            raw_activations = torch.matmul(
                mac_inputs.transpose(0, 1),
                self.weights
            ).transpose(0, 1)

            torch.div(raw_activations, num_active_inputs, out=raw_activations)
            torch.nan_to_num(raw_activations, nan=0.0, out=raw_activations)

            raw_activations = raw_activations.view(
                batch_size, self.num_macs,
                self.num_cms_per_mac,
                self.num_neurons_per_cm
            )

            if self.training:
                familiarities = torch.max(
                    raw_activations, dim=3, keepdim=True
                )[0]

                etas = torch.mean(familiarities, dim=2, keepdim=True)
                torch.sub(etas, self.min_familiarity, out=etas)
                torch.div(etas, 1.0 - self.min_familiarity, out=etas)
                torch.mul(etas, self.sigmoid_chi, out=etas)
                torch.maximum(
                    etas, torch.zeros(
                        (), dtype=torch.float32
                    ), out=etas
                )

                probs = raw_activations
                torch.mul(-self.sigmoid_lambda, probs, out=probs)
                torch.add(probs, self.sigmoid_phi, out=probs)
                torch.exp(probs, out=probs)
                torch.add(probs, 1.0, out=probs)
                torch.div(etas, probs, out=probs)
                torch.add(probs, 1e-6, out=probs)

                prob_dist = Categorical(probs=raw_activations)
                active_neurons = prob_dist.sample().unsqueeze(-1)
            else:
                active_neurons = torch.argmax(
                    raw_activations, dim=3, keepdim=True
                )

            output = raw_activations
            torch.zeros(
                output.shape, out=output,
                dtype=torch.float32, device=self.device
            )

            output.scatter_(
                3, active_neurons,
                torch.ones(
                    output.shape,
                    dtype=torch.float32,
                    device=self.device
                )
            )

            output = output.view((batch_size, self.num_macs, -1))
            torch.mul(output, macs_are_active, out=output)

        return output


class SparseyLayerV2(torch.nn.Module):
    """
    SparseyLayer: class representing layers in the Sparsey model.
    Attributes:
        num_macs: int containing the number of macs in the layer.
        receptive_field_radius: float containing the radius of the 
            receptive field for the MAC.
        mac_positions: list[Tuple[int, int]] containing the positions
            of each MAC in the layer on the grid.
        input_list: list[list[int]] cotaining the indices of the 
            MACs in the previous layer within the receptive field of
            each MAC in this layer.
        mac_list: list[MAC] containing all the MACs in this layer.
        sigmoid_lambda (float): parameter for the familiarity computation.
        sigmoid_phi (float): parameter for the familiarity computation.
        activation_thresholds (list[list[Or[int, float]]]): a list
            of lists containing activation thresholds for each MAC in
            the Sparsey layer.
    """
    def __init__(self, autosize_grid: bool, grid_layout: str,
        num_macs: int, num_cms_per_mac: int, num_neurons_per_cm: int,
        mac_grid_num_rows: int, mac_grid_num_cols: int,
        mac_receptive_field_size: float,
        prev_layer_num_cms_per_mac: int,
        prev_layer_num_neurons_per_cm: int,
        prev_layer_mac_grid_num_rows: int,
        prev_layer_mac_grid_num_cols: int,
        prev_layer_num_macs: int, prev_layer_grid_layout: str,
        layer_index: int,
        sigmoid_phi: float, sigmoid_lambda: float,
        saturation_threshold: float,
        permanence_steps: float, permanence_convexity: float,
        activation_threshold_min: float,
        activation_threshold_max: float,
        min_familiarity: float, sigmoid_chi: float,
        device: torch.device):
        """
        Initializes the SparseyLayer object.
        Args:
            autosize_grid (bool): whether the grid should be autosized.
            grid_layout (str): the layout of the grid (rectangular or hexagonal).
            num_macs (int): the number of MACs in the layer.
            num_cms_per_mac (int): the number of CMs in each MAC.
            num_neurons_per_cm (int): the number of neurons in each CM.
            mac_grid_num_rows (int): the number of rows in the grid.
            mac_grid_num_cols (int): the number of columns in the grid.
            mac_receptive_field_size (float): the radius of the receptive field.
            prev_layer_num_cms_per_mac (int): the number of CMs per MAC in the
                previous layer.
            prev_layer_num_neurons_per_cm (int): the number of neurons per CM
                in the previous layer.
            prev_layer_mac_grid_num_rows (int): the number of rows in the grid
                for the previous layer.
            prev_layer_mac_grid_num_cols (int): the number of columns in the grid
                for the previous layer.
            prev_layer_num_macs (int): the number of MACs in the previous layer.
            prev_layer_grid_layout (str): the layout of the grid for the previous
                layer.
            layer_index (int): the index of the layer.
            sigmoid_phi (float): the phi parameter for the sigmoid function.
            sigmoid_lambda (float): the lambda parameter for the sigmoid function.
            saturation_threshold (float): the threshold for saturation.
            permanence_steps (float): the number of permanence steps.
            permanence_convexity (float): the convexity of the permanence.
            activation_threshold_min (float): the minimum activation threshold.
            activation_threshold_max (float): the maximum activation threshold.
            min_familiarity (float): the minimum familiarity.
            sigmoid_chi (float): the chi parameter for the sigmoid function.
            device (torch.device): the device to run the model on.
        """
        super().__init__()

        self.device = device
        self.layer_index = layer_index
        self.is_grid_autosized = autosize_grid
        self.num_macs = num_macs
        self.num_cms_per_mac = num_cms_per_mac
        self.num_neurons_per_cm = num_neurons_per_cm
        self.receptive_field_radius = mac_receptive_field_size
        self.min_familiarity = min_familiarity
        self.sigmoid_phi = sigmoid_phi
        self.sigmoid_lambda = sigmoid_lambda
        self.sigmoid_chi = sigmoid_chi
        self.permanence_steps = permanence_steps
        self.permanence_convexity = permanence_convexity
        self.saturation_threshold = saturation_threshold

        self.grid_size = (
            mac_grid_num_rows,
            mac_grid_num_cols
        )

        self.prev_layer_grid_size = (
            prev_layer_mac_grid_num_rows,
            prev_layer_mac_grid_num_cols
        )

        self.prev_layer_output_shape = (
            prev_layer_num_macs,
            prev_layer_num_cms_per_mac * prev_layer_num_neurons_per_cm
        )

        self.mac_positions = self.compute_mac_positions(
            num_macs, mac_grid_num_rows, mac_grid_num_cols,
            grid_layout
        )

        prev_layer_mac_positions = self.compute_mac_positions(
            prev_layer_num_macs, prev_layer_mac_grid_num_rows,
            prev_layer_mac_grid_num_cols, prev_layer_grid_layout
        )

        self.input_connections = self.find_connected_macs_in_prev_layer(
            self.mac_positions, prev_layer_mac_positions
        )

        self.weights = torch.nn.Parameter(
            torch.nested.nested_tensor(
                [
                    torch.rand(
                        (
                            len(input_connection) *
                            prev_layer_num_cms_per_mac *
                            prev_layer_num_neurons_per_cm,
                            num_cms_per_mac * num_neurons_per_cm
                        ),
                        dtype=torch.float32, device=self.device
                    )
                    for input_connection in self.input_connections
                ], device=device, requires_grad=False
            )
        )

        self.activation_threshold_mins = torch.FloatTensor(
            [
                activation_threshold_min * len(input_connection) *
                prev_layer_num_cms_per_mac
                for input_connection in self.input_connections
            ]
        ).unsqueeze(0)

        self.activation_threshold_maxes = torch.FloatTensor(
            [
                activation_threshold_max * len(input_connection) *
                prev_layer_num_cms_per_mac
                for input_connection in self.input_connections
            ]
        ).unsqueeze(0)


    def compute_mac_positions(
        self, num_macs: int, mac_grid_num_rows: int,
        mac_grid_num_cols: int,
        grid_layout: str) -> List[Tuple[float, float]]:
        """
        Computes the positions of each MAC in this layer.
        Args:
            num_macs: int representing the number of macs in the layer.
            mac_grid_num_rows: int representing the number of rows
                in the grid for this layer.
            mac_grid_num_cols: int representing the number of columns
                in the grid for this layer.   
            grid_layout: the type of grid layout (rectangular or hexagonal)
                for the layer.   
        Returns:
            (list[Tuple(int, int)]): the positions of all MACs in the layer.      
        """
        mac_positions = []
        global_col_offset = 0.5 if grid_layout == 'hex' else 0

        grid_col_spacing = 0.0

        if mac_grid_num_rows == 1:
            row_locations = [0.5]
        else:
            grid_row_spacing = 1 / (mac_grid_num_rows - 1)

            row_locations = [
                i * grid_row_spacing
                for i in range(mac_grid_num_rows)
            ]

        if mac_grid_num_cols == 1:
            col_locations = [0.5]
        else:
            grid_col_spacing = 1 / (mac_grid_num_cols - 1)

            col_locations = [
                i * grid_col_spacing
                for i in range(mac_grid_num_cols)
            ]

        for i in range(num_macs):
            mac_positions.append(
                (
                    row_locations[i // mac_grid_num_cols],
                    col_locations[i % mac_grid_num_cols] + (
                        global_col_offset * (
                            (i % mac_grid_num_rows) % 2
                        ) * grid_col_spacing
                    )
                 )
            )

        return mac_positions


    def _compute_distance(self,
        position_1: Tuple[float, float],
        position_2: Tuple[float, float]) -> float:
        """
        Computes the Euclidean distance between two positions.
        Args:
            position_1 (Tuple[int, int]): x and y coordinates of the
                first point.
        """
        return (
            abs(position_1[0] - position_2[0]) ** 2 +
            abs(position_1[1] - position_2[1]) ** 2
        ) ** 0.5


    def find_connected_macs_in_prev_layer(
        self, mac_positions: list[Tuple[float, float]],
        prev_layer_mac_positions: list[Tuple[float, float]]
    ) -> list[torch.Tensor]:
        """
        Finds the list of connected MACs in the previous layer
        for each MAC in the current layer.
        Args:
            mac_positions (list[Tuple[int, int]]): list
                of positions of MACs in the current layer.
            prev_layer_mac_positions (list[Tuple[int, int]]):
                list of positions of MACs in the previous layer.
        Returns:
            (list[torch.Tesnor]): list of tensors containing the indices
                of connected MACs from the previous layer for each
                MAC in the current layer.
        """
        connections = []

        for mac_position in mac_positions:
            mac_connections = []

            for (
                index, prev_layer_mac_position
            ) in enumerate(prev_layer_mac_positions):
                if self._compute_distance(
                    mac_position,
                    prev_layer_mac_position
                ) <= self.receptive_field_radius:
                    mac_connections.append(index)

            connections.append(mac_connections)

        return connections


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Passes data through a Sparsey layer.
        Args:
            x: torch.Tensor of size (
                batch_size,
                prev_layer_num_macs,
                prev_layer_num_cms_per_mac *
                prev_layer_num_neurons_per_cm
            ) of dtype torch.float32
        Returns:
            torch.Tensor of size (
                batch_size,
                num_macs,
                num_cms_per_mac * num_neurons_per_cm
            ) of dtype torch.float32
        """
        if tuple(x.shape[1:]) != self.prev_layer_output_shape:
            raise ValueError(
                'Input shape is incorrect! '
                f'Expected shape {self.prev_layer_output_shape} but received '
                f'{tuple(x.shape[1:])} instead.'    
            )

        batch_size = x.shape[0]

        with torch.no_grad():
            mac_inputs = torch.nested.nested_tensor(
                [
                    x[:, input_connection].view(batch_size, -1)
                    for input_connection in self.input_connections
                ], requires_grad=False
            )

            num_active_inputs = torch.sum(x, dim=2)
            num_active_input_neurons = torch.stack(
                [
                    torch.sum(
                        num_active_inputs[: , input_connection],
                        dim=1
                    )
                    for input_connection in self.input_connections
                ], dim=0
            ).transpose(0, 1)

            macs_are_active = torch.logical_and(
                torch.ge(
                    num_active_input_neurons,
                    self.activation_threshold_mins
                ),
                torch.le(
                    num_active_input_neurons,
                    self.activation_threshold_maxes
                )
            )

            raw_activations = torch.stack(
                torch.matmul(
                    mac_inputs,
                    self.weights
                ).unbind()
            ).transpose(0, 1)

            torch.div(
                raw_activations,
                num_active_input_neurons.unsqueeze(-1),
                out=raw_activations
            )

            torch.nan_to_num(raw_activations, nan=0.0, out=raw_activations)
            raw_activations = raw_activations.view(
                batch_size, self.num_macs,
                self.num_cms_per_mac,
                self.num_neurons_per_cm
            )

            if self.training:
                familiarities = torch.max(
                    raw_activations, dim=3, keepdim=True
                )[0]

                etas = torch.mean(familiarities, dim=2, keepdim=True)
                torch.sub(etas, self.min_familiarity, out=etas)
                torch.div(etas, 1.0 - self.min_familiarity, out=etas)
                torch.mul(etas, self.sigmoid_chi, out=etas)
                torch.maximum(
                    etas, torch.zeros(
                        (), dtype=torch.float32
                    ), out=etas
                )

                probs = raw_activations
                torch.mul(-self.sigmoid_lambda, probs, out=probs)
                torch.add(probs, self.sigmoid_phi, out=probs)
                torch.exp(probs, out=probs)
                torch.add(probs, 1.0, out=probs)
                torch.div(etas, probs, out=probs)
                torch.add(probs, 1e-6, out=probs)

                prob_dist = Categorical(probs=raw_activations)
                active_neurons = prob_dist.sample().unsqueeze(-1)
            else:
                active_neurons = torch.argmax(
                    raw_activations, dim=3, keepdim=True
                )

            output = raw_activations
            torch.zeros(
                output.shape, out=output,
                dtype=torch.float32, device=self.device
            )

            output.scatter_(
                3, active_neurons,
                torch.ones(
                    output.shape,
                    dtype=torch.float32,
                    device=self.device
                )
            )

            output = output.view((batch_size, self.num_macs, -1))

        return output


class MAC(torch.nn.Module):
    """
    MAC: class to represent macrocolumns in the Sparsey model.

    Attributes:
        weights: torch.Tensor containing the weights for the MAC.
        input_filter: torch.Tensor containing the indices of the
            MACs in the previous layer that are in the receptive field
            of the current MAC.
    """
    def __init__(self, num_cms: int,
        num_neurons: int, input_filter: torch.Tensor,
        num_cms_per_mac_in_input: int,
        num_neurons_per_cm_in_input: int,
        layer_index: int, mac_index: int,
        sigmoid_lambda: float, sigmoid_phi: float,
        permanence_steps: float, permanence_convexity: float,
        activation_threshold_min: float,
        activation_threshold_max: float,
        sigmoid_chi: float, min_familiarity: float,
        device: torch.device, prev_layer_num_cms_per_mac: int,
        prev_layer_num_neurons_per_cm: int
    ) -> None:
        """
        Initializes the MAC object.

        Args:
            layer_index (int): the layer number
            mac_index (int): the max number within the layer 
            num_cms: int repesenting the number of CMs the MAC should contain.
            num_neurons: int representing the number of neurons per CM.
            input_filter: 1d torch.Tensor of dtype torch.long
                containing the indices of the MACs in the previous layer
                that are connected to this MAC.
            num_cms_per_mac_in_input: the number of CMs per mac in the input
            num_neurons_per_cm_in_input: the number of neurons per CM in
                the input.
            sigmoid_lambda (float): parameter for the familiarity computation.
            sigmoid_phi (float): parameter for the familiarity computation.
            activation_threshold_min (float): lower
                bound for the number of MACs that need to be active in the 
                receptive field of the MAC for it to become active.
            activation_threshold_max (float): upper
                bound for the number of MACs that need to be active in the 
                receptive field of the MAC for it to become active.
            sigmoid_chi: expansion factor for the sigmoid used to compute
                the distribution over CMs for the CSA.
            min_familiarity: the minimum average global familiarty required
                for the CSA to not construct a uniform distribution over
                neurons in a CM.
        """
        super().__init__()
        num_inputs = input_filter.shape[0]
        num_inputs *= num_cms_per_mac_in_input
        num_inputs *= num_neurons_per_cm_in_input

        if len(input_filter) == 0:
            raise ValueError(
                'MAC input connection list cannot be empty! ' + 
                'This is most likely due to a bad set of layer ' + 
                'configurations, especially the mac_grid_num_rows and ' + 
                'mac_grid_num_cols properties.'
            )

        self.device = device
        self.input_num_cms = num_cms_per_mac_in_input
        self.input_num_neurons = num_neurons_per_cm_in_input
        self.input_num_macs = input_filter.shape[0]
        self.prev_layer_num_cms_per_mac = prev_layer_num_cms_per_mac

        self.layer_index = layer_index
        self.mac_index = mac_index

        self.activation_threshold_min = (
            self.input_num_macs * activation_threshold_min
        )

        self.activation_threshold_max = (
            self.input_num_macs * activation_threshold_max
        )

        self.sigmoid_lambda = sigmoid_lambda
        self.sigmoid_phi = sigmoid_phi
        self.sigmoid_chi = sigmoid_chi
        self.min_familiarity = min_familiarity

        self.permanence_steps = permanence_steps
        self.permanence_convexity = permanence_convexity

        self.weights = torch.nn.Parameter(
            torch.rand(
                (num_cms, num_inputs, num_neurons),
                dtype=torch.float32
            ), requires_grad=False
        )

        self.stored_codes = set()
        self.input_filter = input_filter.to(self.device)
        self.training = True

        self.is_active = True


    def get_input_filter(self) -> torch.Tensor:
        """
        Returns the input filter for the MAC.
        """
        return self.input_filter


    def train(self, mode: bool = True) -> None:
        self.training = mode


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Passes data through a MAC.

        Args:
            x: torch.Tensor of size (
                batch_size,
                num_macs_in_prev_layer,
                prev_layer_num_cms_per_mac,
                prev_layer_num_neurons_per_cm
            ) with dtype torch.float32

        Returns:
            torch.Tensor of size (
                batch_size,
                num_cms_per_mac,
                num_neurons_per_cm
            ) with dtype torch.float32
        """
        with torch.no_grad():
            # compute the number of incoming active MACs
            # for each sample in the batch
            active_input_macs = torch.div(
                torch.sum(
                    x, dim=(1, 2, 3)
                ),
                self.prev_layer_num_cms_per_mac
            )

            # find out if the MAC should be active or not
            # for each sample in the batch
            self.is_active = torch.logical_and(
                torch.le(active_input_macs, self.activation_threshold_max),
                torch.ge(active_input_macs, self.activation_threshold_min)
            ).float()

            # flatten x, maintaining only the batch dim.
            x = x.view(x.shape[0], -1)

            # normalize the input signal
            x = torch.div(x, torch.sum(x, -1, keepdim=True))
            x = torch.nan_to_num(x)

            # get the activations for all neurons
            # in all CMs in this MAC
            x = torch.matmul(x, self.weights)

            # swap dims to make sure batch dimension
            # is the leftmost dimension
            x = x.permute((1, 0, *list(range(2, len(x.shape)))))

            # get the max value from each CM
            familiarities = torch.max(x, -1)[0]

            if self.training:
                # compute the average familiarity across the MAC
                average_familiarity = torch.mean(familiarities, dim=1)

                # compute eta for the softmax
                eta = torch.max(
                    torch.div(
                        average_familiarity - self.min_familiarity,
                        1.0 - self.min_familiarity
                    ), torch.zeros_like(
                        average_familiarity,
                        dtype=torch.float32
                    )
                ) * self.sigmoid_chi

                # compute the logits for sampling the active neuron
                # in each CM
                cm_logits = torch.log(
                    torch.div(
                        eta.unsqueeze(1).unsqueeze(2).repeat(1, *x.shape[1:]),
                        1.0 + torch.exp(
                            -1.0 * self.sigmoid_lambda * x + self.sigmoid_phi
                        )
                    ) + 1e-5
                )

                # sample from categorial dist using processed inputs as logits
                active_neurons = Categorical(
                    logits=cm_logits
                ).sample().unsqueeze(-1)
            else:
                active_neurons = torch.argmax(x, 2, keepdim=True)

            output = torch.zeros(x.shape, dtype=torch.float32, device=self.device)
            output.scatter_(
                2, active_neurons,
                torch.ones(x.shape, dtype=torch.float32, device=self.device)
            )

            output = torch.mul(
                output, self.is_active.unsqueeze(1).unsqueeze(2)
            )

            if self.training:
                for code in active_neurons.flatten(start_dim=1).cpu().numpy():
                    self.stored_codes.add(tuple(code))

            return output


class SparseyLayerOld(torch.nn.Module):
    """
    SparseyLayer: class representing layers in the Sparsey model.

    Attributes:
        num_macs: int containing the number of macs in the layer.
        receptive_field_radius: float containing the radius of the 
            receptive field for the MAC.
        mac_positions: list[Tuple[int, int]] containing the positions
            of each MAC in the layer on the grid.
        input_list: list[list[int]] cotaining the indices of the 
            MACs in the previous layer within the receptive field of
            each MAC in this layer.
        mac_list: list[MAC] containing all the MACs in this layer.
        sigmoid_lambda (float): parameter for the familiarity computation.
        sigmoid_phi (float): parameter for the familiarity computation.
        activation_thresholds (list[list[Or[int, float]]]): a list
            of lists containing activation thresholds for each MAC in
            the Sparsey layer.
    """
    def __init__(self, autosize_grid: bool, grid_layout: str,
        num_macs: int, num_cms_per_mac: int, num_neurons_per_cm: int,
        mac_grid_num_rows: int, mac_grid_num_cols: int,
        mac_receptive_field_size: float,
        prev_layer_num_cms_per_mac: int,
        prev_layer_num_neurons_per_cm: int,
        prev_layer_mac_grid_num_rows: int,
        prev_layer_mac_grid_num_cols: int,
        prev_layer_num_macs: int, prev_layer_grid_layout: str,
        layer_index: int,
        sigmoid_phi: float, sigmoid_lambda: float,
        saturation_threshold: float,
        permanence_steps: float, permanence_convexity: float,
        activation_threshold_min: float,
        activation_threshold_max: float,
        min_familiarity: float, sigmoid_chi: float,
        device: torch.device):
        """
        Initializes the SparseyLayer object.

        Args:

        """
        super().__init__()

        self.device = device
        self.is_grid_autosized = autosize_grid
        self.num_macs = num_macs
        self.receptive_field_radius = mac_receptive_field_size
        self.grid_size = (
            mac_grid_num_rows,
            mac_grid_num_cols
        )

        self.prev_layer_grid_size = (
            prev_layer_mac_grid_num_rows,
            prev_layer_mac_grid_num_cols
        )

        self.prev_layer_output_shape = (
            prev_layer_mac_grid_num_rows,
            prev_layer_mac_grid_num_cols,
            prev_layer_num_cms_per_mac,
            prev_layer_num_neurons_per_cm
        )

        # save layer-level permanence value;
        # check if we actually need to do this
        self.permanence_steps = permanence_steps
        self.permanence_convexity = permanence_convexity
        self.activation_threshold_min = activation_threshold_min
        self.activation_threshold_max = activation_threshold_max

        self.mac_positions = self.compute_mac_positions(
            num_macs, mac_grid_num_rows, mac_grid_num_cols,
            grid_layout
        )

        prev_layer_mac_positions = self.compute_mac_positions(
            prev_layer_num_macs, prev_layer_mac_grid_num_rows,
            prev_layer_mac_grid_num_cols, prev_layer_grid_layout
        )

        self.input_connections = self.find_connected_macs_in_prev_layer(
            self.mac_positions, prev_layer_mac_positions
        )

        self.input_connections = [
            input_connection.to(device) for input_connection
            in self.input_connections
        ]

        self.mac_list = [
            MAC(
                num_cms_per_mac, num_neurons_per_cm,
                self.input_connections[i], prev_layer_num_cms_per_mac,
                prev_layer_num_neurons_per_cm,
                # push mac_index down into MAC
                layer_index, i,
                sigmoid_lambda, sigmoid_phi,
                # pass layer permanence value to individual MACs
                # this might need adjusting so it can be set
                # on a per-MAC basis
                permanence_steps, permanence_convexity, 
                activation_threshold_min,
                activation_threshold_max,
                sigmoid_chi, min_familiarity,
                self.device, prev_layer_num_cms_per_mac,
                prev_layer_num_neurons_per_cm
            ) for i in range(num_macs)
        ]

        self.mac_list = torch.nn.ModuleList(self.mac_list)

        self.saturation_threshold = saturation_threshold

        ####Edit out when we have a better mechanism for tracking layers
        self.layer_index = layer_index


    def get_macs(self) -> list[MAC]:
        """
        Returns the MACs making up the layer.

        Returns:
            (list[MAC]): the MACs making up the layer.
        """
        return self.mac_list


    def compute_mac_positions(
        self, num_macs: int, mac_grid_num_rows: int,
        mac_grid_num_cols: int,
        grid_layout: str) -> List[Tuple[float, float]]:
        """
        Computes the positions of each MAC in this layer.

        Args:
            num_macs: int representing the number of macs in the layer.
            mac_grid_num_rows: int representing the number of rows
                in the grid for this layer.
            mac_grid_num_cols: int representing the number of columns
                in the grid for this layer.   
            grid_layout: the type of grid layout (rectangular or hexagonal)
                for the layer.   

        Returns:
            (list[Tuple(int, int)]): the positions of all MACs in the layer.      
        """
        mac_positions = []
        global_col_offset = 0.5 if grid_layout == 'hex' else 0

        grid_col_spacing = 0.0

        if mac_grid_num_rows == 1:
            row_locations = [0.5]
        else:
            grid_row_spacing = 1 / (mac_grid_num_rows - 1)

            row_locations = [
                i * grid_row_spacing
                for i in range(mac_grid_num_rows)
            ]

        if mac_grid_num_cols == 1:
            col_locations = [0.5]
        else:
            grid_col_spacing = 1 / (mac_grid_num_cols - 1)

            col_locations = [
                i * grid_col_spacing
                for i in range(mac_grid_num_cols)
            ]

        for i in range(num_macs):
            mac_positions.append(
                (
                    row_locations[i // mac_grid_num_cols],
                    col_locations[i % mac_grid_num_cols] + (
                        global_col_offset * (
                            (i % mac_grid_num_rows) % 2
                        ) * grid_col_spacing
                    )
                 )
            )

        return mac_positions


    def _compute_distance(self,
        position_1: Tuple[float, float],
        position_2: Tuple[float, float]) -> float:
        """
        Computes the Euclidean distance between two positions.

        Args:
            position_1 (Tuple[int, int]): x and y coordinates of the
                first point.
        """
        return (
            abs(position_1[0] - position_2[0]) ** 2 +
            abs(position_1[1] - position_2[1]) ** 2
        ) ** 0.5


    def find_connected_macs_in_prev_layer(
        self, mac_positions: list[Tuple[float, float]],
        prev_layer_mac_positions: list[Tuple[float, float]]
    ) -> list[torch.Tensor]:
        """
        Finds the list of connected MACs in the previous layer
        for each MAC in the current layer.

        Args:
            mac_positions (list[Tuple[int, int]]): list
                of positions of MACs in the current layer.
            prev_layer_mac_positions (list[Tuple[int, int]]):
                list of positions of MACs in the previous layer.

        Returns:
            (list[torch.Tesnor]): list of tensors containing the indices
                of connected MACs from the previous layer for each
                MAC in the current layer.
        """
        connections = []

        for mac_position in mac_positions:
            mac_connections = []

            for (
                index, prev_layer_mac_position
            ) in enumerate(prev_layer_mac_positions):
                if self._compute_distance(
                    mac_position,
                    prev_layer_mac_position
                ) <= self.receptive_field_radius:
                    mac_connections.append(index)

            connections.append(mac_connections)

        return [torch.Tensor(conn).long() for conn in connections]


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Passes data through a Sparsey layer.

        Args:
            x: torch.Tensor of size (
                batch_size,
                prev_layer_mac_grid_num_rows,
                prev_layer_mac_grid_num_cols,
                prev_layer_num_cms_per_mac,
                prev_layer_num_neurons_per_cm
            ) of dtype torch.float32

        Returns:
            torch.Tensor of size (
                batch_size,
                mac_grid_num_rows,
                mac_grid_num_cols,
                num_cms_per_mac,
                num_neurons_per_cm
            ) of dtype torch.float32
        """
        if tuple(x.shape[1:]) != self.prev_layer_output_shape:
            raise ValueError(
                'Input shape is incorrect! '
                f'Expected shape {self.prev_layer_output_shape} but received '
                f'{tuple(x.shape[1:])} instead.'    
            )

        x = x.view(x.shape[0], -1, *x.shape[3:])

        # apply input filter to select only the
        # input signals (neurons) that this MAC
        # cares about.
        mac_outputs = [
            mac(
                torch.index_select(x, 1, input_filter)
            ) for mac, input_filter in zip(
                self.mac_list, self.input_connections
            )
        ]

        out = torch.stack(mac_outputs, dim=1)
        out = out.reshape(
            (out.shape[0], *self.grid_size, *out.shape[2:])
        )

        return out
