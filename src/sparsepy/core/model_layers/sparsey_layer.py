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
                 num_neurons_per_cm_in_input: int) -> None:
        """
        Initializes the MAC object.

        Args:
            num_cms: int repesenting the number of CMs the MAC should contain.
            num_neurons: int representing the number of neurons per CM.
            input_filter: 1d torch.Tensor of dtype torch.long
                containing the indices of the MACs in the previous layer
                that are connected to this MAC.
            num_cms_per_mac_in_input: the number of CMs per mac in the input
            num_neurons_per_cm_in_input: the number of neurons per CM in
                the input.
        """
        super().__init__()
        num_inputs = input_filter.shape[0]
        num_inputs *= num_cms_per_mac_in_input
        num_inputs *= num_neurons_per_cm_in_input

        self.input_num_cms = num_cms_per_mac_in_input
        self.input_num_neurons = num_neurons_per_cm_in_input
        self.input_min_macs = torch.max(input_filter).item()

        self.weights = torch.randint(
            0, 2, (num_cms, num_inputs, num_neurons),
            dtype=torch.float32
        )

        self.input_filter = input_filter


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
        if (
            (len(x.shape) != 4) or 
            (tuple(x.shape[2:4]) != (
                self.input_num_cms, self.input_num_neurons
                )
            ) or 
            (x.shape[1] <= self.input_min_macs)
        ):
            raise ValueError(f'Incorrect input size: {x.shape}!')

        with torch.no_grad():
            # apply input filter to select only the
            # input signals (neurons) that this MAC
            # cares about.
            x = torch.index_select(x, 1, self.input_filter)

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

            # compute the average familiarity across the MAC
            average_familiarity = torch.mean(familiarities)

            # compute temperature for softmax
            softmax_temp = torch.div(1.0, average_familiarity + 1e-4) - 1

            # scale the logits for the softmax using the
            # average familiarity (higher familiarity => lower temperature,
            # and lower familiarity => higher temperature)
            x = torch.div(x, softmax_temp)

            # sample from categorial dist using processed inputs as logits
            active_neurons = Categorical(logits=x).sample().unsqueeze(-1)

            output = torch.zeros(x.shape, dtype=torch.float32)
            output.scatter_(
                2, active_neurons,
                torch.ones(x.shape, dtype=torch.float32)
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
    """
    def __init__(self, num_macs: int, num_cms_per_mac: int,
        num_neurons_per_mac: int, mac_grid_num_rows: int,
        mac_grid_num_cols: int, mac_receptive_field_radius: float,
        prev_layer_cms_per_mac: int,
        prev_layer_neurons_per_cm: int,
        prev_layer_mac_positions: list[Tuple[float, float]]):
        """
        Initializes the SparseyLayer object.

        Args:

        """
        super().__init__()

        self.num_macs = num_macs
        self.receptive_field_radius = mac_receptive_field_radius

        self.mac_positions = self.compute_mac_positions(
            num_macs, mac_grid_num_rows, mac_grid_num_cols
        )

        self.input_connections = self.find_connected_macs_in_prev_layer(
            self.mac_positions, prev_layer_mac_positions
        )

        self.input_connections = [
            torch.Tensor(connections).long()
            for connections in self.input_connections
        ]

        self.mac_list = [
            MAC(
                num_cms_per_mac, num_neurons_per_mac,
                self.input_connections[i], prev_layer_cms_per_mac,
                prev_layer_neurons_per_cm
            ) for i in range(num_macs)
        ]


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
    ) -> list[list[int]]:
        """
        Finds the list of connected MACs in the previous layer
        for each MAC in the current layer.

        Args:
            mac_positions (list[Tuple[int, int]]): list
                of positions of MACs in the current layer.
            prev_layer_mac_positions (list[Tuple[int, int]]):
                list of positions of MACs in the previous layer.

        Returns:
            (list[list[int]]): list of lists containing the indices
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
        mac_outputs = [
            mac(x) for mac in self.mac_list
        ]

        return torch.stack(mac_outputs, dim=1)


if __name__ == "__main__":
    layer = SparseyLayer(
        5, 10, 10, 2, 3, 0.5, 5, 5,
        [
            (0.0, 0.0), (0.0, 0.5), (0.0, 1.0),
            (0.5, 0.0), (0.5, 0.5), (0.5, 1.0),
            (1.0, 0.0), (1.0, 0.5), (1.0, 1.0),
        ]
    )

    out = layer(
        torch.randint(0, 2, (16, 9, 5, 5), dtype=torch.float32)
    )

    print(out.shape)