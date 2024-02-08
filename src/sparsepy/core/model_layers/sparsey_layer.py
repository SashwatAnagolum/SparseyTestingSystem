# -*- coding: utf-8 -*-

"""
Sparsey Layer: code for building and using individual layers
    in a Sparsey model.
"""


from typing import List, Tuple

import torch
from torch.distributions.categorical import Categorical


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
                 layer_index: int,
                 sigmoid_lambda=28.0, sigmoid_phi=5.0,
                 persistence=1.0,
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
        """
        super().__init__()
        num_inputs = input_filter.shape[0]
        num_inputs *= num_cms_per_mac_in_input
        num_inputs *= num_neurons_per_cm_in_input

        self.layer_index = layer_index

        if len(input_filter) == 0:
            raise ValueError(
                'MAC input connection list cannot be empty! ' + 
                'This is most likely due to a bad set of layer ' + 
                'configurations, especially the mac_grid_num_rows and ' + 
                'mac_grid_num_cols properties.'
            )

        self.input_num_cms = num_cms_per_mac_in_input
        self.input_num_neurons = num_neurons_per_cm_in_input
        self.input_min_macs = input_filter.shape[0]

        self.sigmoid_lambda = sigmoid_lambda
        self.sigmoid_phi = sigmoid_phi

        self.persistence = persistence

        self.weights = torch.nn.Parameter(
            torch.zeros(
                (num_cms, num_inputs, num_neurons),
                dtype=torch.float32
            ), requires_grad=False
        )

        self.stored_codes = set()

        self.input_filter = input_filter
        self.training = True
        
        self.is_active = True


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
                average_familiarity = torch.mean(familiarities)

                # compute the logits for sampling the active neuron
                # in each CM
                cm_logits = torch.log(
                    torch.div(
                        1.0,
                        1.0 + torch.exp(
                            -1.0 * self.sigmoid_lambda * x + self.sigmoid_phi
                        )
                    )
                )

                # sample from categorial dist using processed inputs as logits
                active_neurons = Categorical(
                    logits=cm_logits
                ).sample().unsqueeze(-1)
            else:
                active_neurons = torch.argmax(x, 2, keepdim=True)

            output = torch.zeros(x.shape, dtype=torch.float32)
            output.scatter_(
                2, active_neurons,
                torch.ones(x.shape, dtype=torch.float32)
            )

            self.is_active = (output.count_nonzero().item() > 0)

            if self.training:
                if tuple(
                    [i for i in active_neurons.flatten().numpy()]
                ) not in self.stored_codes:
                    self.stored_codes.add(
                        tuple(
                            [i for i in active_neurons.flatten().numpy()]
                        )
                    )

            return output


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
    """
    def __init__(self, num_macs: int, num_cms_per_mac: int,
        num_neurons_per_cm: int, mac_grid_num_rows: int,
        mac_grid_num_cols: int, mac_receptive_field_radius: float,
        prev_layer_num_cms_per_mac: int,
        prev_layer_num_neurons_per_cm: int,
        prev_layer_mac_grid_num_rows: int,
        prev_layer_mac_grid_num_cols: int,
        prev_layer_num_macs: int,
        layer_index: int,
        sigmoid_phi: float, sigmoid_lambda: float,
        saturation_threshold: float,
        persistence: float):
        """
        Initializes the SparseyLayer object.

        Args:

        """
        super().__init__()

        self.num_macs = num_macs
        self.receptive_field_radius = mac_receptive_field_radius

        # save layer-level persistence value; check if we actually need to do this
        self.persistence = persistence

        self.mac_positions = self.compute_mac_positions(
            num_macs, mac_grid_num_rows, mac_grid_num_cols
        )

        prev_layer_mac_positions = self.compute_mac_positions(
            prev_layer_num_macs, prev_layer_mac_grid_num_rows,
            prev_layer_mac_grid_num_cols
        )

        self.input_connections = self.find_connected_macs_in_prev_layer(
            self.mac_positions, prev_layer_mac_positions
        )

        self.mac_list = [
            MAC(
                num_cms_per_mac, num_neurons_per_cm,
                self.input_connections[i], prev_layer_num_cms_per_mac,
                prev_layer_num_neurons_per_cm, 
                layer_index, # ORDER MATTERS and this section needs to match the constructor signature exactly
                sigmoid_lambda, sigmoid_phi,
                persistence # pass layer persistence value to individual MACs--this might need adjusting so it can be set on a per-MAC basis
            ) for i in range(num_macs)
        ]

        self.mac_list = torch.nn.ModuleList(self.mac_list)

        self.saturation_threshold = saturation_threshold


        ####Edit out when we have a better mechanism for tracking layers
        self.layer_index = layer_index

    def compute_mac_positions(
        self, num_macs: int, mac_grid_num_rows: int,
        mac_grid_num_cols: int) -> List[Tuple[float, float]]:
        """
        Computes the positions of each MAC in this layer.

        Args:
            num_macs: int representing the number of macs in the layer.
            mac_grid_num_rows: int representing the number of rows
                in the grid for this layer.
            mac_grid_num_cols: int representing the number of columns
                in the grid for this layer.      

        Returns:
            A list[Tuple(int, int)]      
        """
        mac_positions = []

        if mac_grid_num_rows == 1:
            row_locations = [0.5]
        else:
            row_locations = [
                i * (1 / (mac_grid_num_rows - 1))
                for i in range(mac_grid_num_rows)
            ]

        if mac_grid_num_cols == 1:
            col_locations = [0.5]
        else:
            col_locations = [
                i * (1 / (mac_grid_num_cols - 1))
                for i in range(mac_grid_num_cols)
            ]

        for i in range(num_macs):
            mac_positions.append(
                (
                    row_locations[i // mac_grid_num_cols],
                    col_locations[i % mac_grid_num_cols]
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
                num_macs_in_prev_layer,
                prev_layer_num_cms_per_mac,
                prev_layer_num_neurons_per_cm
            ) of dtype torch.float32

        Returns:
            torch.Tensor of size (
                batch_size,
                num_macs,
                num_cms_per_mac,
                num_neurons_per_cm
            ) of dtype torch.float32
        """
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

        return torch.stack(mac_outputs, dim=1)
