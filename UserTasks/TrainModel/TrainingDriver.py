from SystemTasks.ModelManager.ModelTrainer import ModelTrainer
from UserTasks.TrainModel.TrainingRecipe import TrainingRecipe
from SystemTasks.DatasetManager.STSDataset import SparseyDataset
from torch.utils.data import DataLoader

class TrainingDriver:
    def __init__(self):
        self.recipe = None
        self.model_architecture = None
        self.dataset_path = 'SystemTasks/DatasetManager/Data/DemoDataset.txt'  # Hardcoded path
        self.dataloader = None

    def create_training_recipe(self):
        # Prompt user for training parameters
        learning_rate = float(input("Enter learning rate: "))
        epochs = int(input("Enter number of epochs: "))
        batch_size = int(input("Enter batch size: "))

        # Create and set the training recipe
        self.recipe = TrainingRecipe(learning_rate, epochs, batch_size)

    def set_model_architecture(self, architecture_name):
        # Simply store the architecture name
        self.model_architecture = architecture_name

    def prepare_data(self):
        # Load the dataset from the hardcoded path
        dataset = SparseyDataset(self.dataset_path)
        # Create a DataLoader
        self.dataloader = DataLoader(dataset, batch_size=self.recipe.batch_size, shuffle=True)

    def train_model(self):
        if self.recipe is None or self.model_architecture is None:
            raise ValueError("Training recipe or model architecture not set")

        # Prepare data
        self.prepare_data()

        # Initialize the model based on the architecture
        # Here, self.model_architecture() should return an instance of a model
        model_instance = self.model_architecture

        # Create a ModelTrainer instance and start training
        trainer = ModelTrainer(model_instance, self.dataloader, self.recipe)
        trainer.train()