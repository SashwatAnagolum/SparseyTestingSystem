import torch
import torch.nn as nn
import torch.nn.functional as F

class SparseyLayer(nn.Module):
    def __init__(self, num_features, num_cms, num_units_per_cm):
        super().__init__()
        # Assuming num_features is the size of the input field (F1)
        # num_cms is the number of Competitive Modules (CMs) in the coding field (F2)
        # num_units_per_cm is the number of units (neurons) in each CM

        # Define the input field (F1)
        self.input_field = nn.Linear(num_features, num_cms * num_units_per_cm)
        self.max_pooling = nn.MaxPool1d(num_units_per_cm, stride=num_units_per_cm)
        self.avg_pooling = nn.MaxPool1d(num_units_per_cm, stride=num_units_per_cm)

        # Define coding field (F2) as a set of WTA Competitive Modules
        # Here, we use a simple linear layer to represent this; in practice, this will be more complex
        self.coding_field = nn.Linear(num_cms * num_units_per_cm, num_cms * num_units_per_cm)

        # Additional parameters and layers can be defined here

        self.num_cms = num_cms
        self.num_units_per_cm = num_units_per_cm

    def forward(self, x):
        num_active_inputs = torch.sum(x, 1, keepdim=True)

        # compute input signals to all neurons in each CM
        cm_neuron_inputs = self.input_field(x)
        normalized_cm_inputs = cm_neuron_inputs / num_active_inputs

        max_value_per_cm = torch.max(normalized_cm_inputs, 1)[0]

        g_score = torch.mean(max_value_per_cm)

        output = torch.zeros(self.num_cms * self.num_units_per_cm).float()

        print(normalized_cm_inputs)

        for i in range(self.num_cms):
            raw_activations = normalized_cm_inputs[:, i * self.num_units_per_cm:(i + 1) * self.num_units_per_cm]
            activations_max = max_value_per_cm[i]

            print(raw_activations, i)

            raw_activations -= activations_max
            exp_raw_activations = torch.exp(raw_activations)

            dist = torch.distributions.categorical.Categorical(logits=exp_raw_activations)

            chosen_active_neuron = dist.sample()

            print(chosen_active_neuron)

            output[:, i * self.num_units_per_cm + chosen_active_neuron] = 1.0

        return output

    # Additional methods for the Sparsey model can be defined here

# Example instantiation and usage
if __name__ == "__main__":
    num_features = 5  # Example feature size for input
    num_cms = 4  # Number of Competitive Modules in F2
    num_units_per_cm = 3  # Number of units per CM

    sparsey_model = SparseyLayer(num_features, num_cms, num_units_per_cm)
    # Example input tensor
    input_tensor = torch.rand(5, num_features)
    output = sparsey_model(input_tensor)
    print(output)

