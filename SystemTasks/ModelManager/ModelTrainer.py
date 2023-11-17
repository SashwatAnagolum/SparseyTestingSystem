# ModelTrainer.py
import torch
import torch.optim as optim
# Import the necessary model classes
from ModelArchitectures.Sparsey import Sparsey
# Add other model imports as needed

class ModelTrainer:
    def __init__(self, model_name, dataloader, recipe):
        # Instantiate the model based on the model_name
        if model_name == "Sparsey":
            # Set the parameters for Sparsey
            num_features = 64  # Example, adjust as needed
            num_cms = 4        # Example, adjust as needed
            num_units_per_cm = 3  # Example, adjust as needed

            self.model = Sparsey(num_features, num_cms, num_units_per_cm)
        else:
            raise ValueError(f"Unknown model architecture: {model_name}")

        self.dataloader = dataloader
        self.recipe = recipe
        self.optimizer = optim.Adam(self.model.parameters(), lr=recipe.learning_rate)

    def train(self):
        self.model.train()  # Set the model to training mode
        for epoch in range(self.recipe.epochs):
            for inputs, targets in self.dataloader:
                # Forward pass
                outputs = self.model(inputs)
                
                # Compute loss using the model's loss function
                loss = self.model.loss_function(outputs, targets)

                # Backpropagation
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            print(f"Epoch {epoch + 1}/{self.recipe.epochs} completed.")