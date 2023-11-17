# Import the STSManager class (Adjust the import path as necessary based on your package structure)
from UserTasks.ManageSTSBuild.STSManager import STSManager
from UserTasks.TrainModel.TrainingDriver import TrainingDriver

def coreSelection():
    # Create an STSManager instance
    manager = STSManager()

    # Get the list of cores
    cores = manager.coreLookup()

    # Print the menu options
    print("Select from available cores:")
    for index, core in enumerate(cores):
        print(f"{index}: {core}")

    # Read in the user's response and validate
    while True:
        try:
            selected_index = int(input("Enter the number corresponding to your choice: "))
            if 0 <= selected_index < len(cores):
                return cores[selected_index]
            else:
                print("Invalid selection. Please try again.")
        except ValueError:
            print("Invalid input. Please enter a digit.")


def chooseRunHPO():
    # Implement RunHPO logic here
    print("Running HPO...")

def chooseTrainModelInstance():
    # Existing logic for selecting the model architecture
    print("Select the Model Architecture:")
    active_core = coreSelection()
    print(f"You have selected: {active_core}")

    # Create a TrainingDriver instance
    driver = TrainingDriver()

    # Create the training recipe through user input
    driver.create_training_recipe()

    # Pass the selected model architecture to the Training Driver
    driver.set_model_architecture(active_core)

    # Initiate the training process
    driver.train_model()


def chooseEvaluateModelInstance():
    # Implement Evaluate Model Instance logic here
    print("Evaluating Model Instance...")

def chooseVisualizeResults():
    # Implement Visualize Results logic here
    print("Visualizing Results...")

def chooseChangeModelCore(current_core):
    # Implement Change Model Core logic here
    print("Changing Model Core...")

    print(f"Current core: {current_core}")
    current_core = coreSelection()
    print(f"You have selected: {current_core}")

def printMainMenu():
    while True:
        print("\nMain Menu:")
        print("0. Exit")
        print("1. RunHPO")
        print("2. Train Model Instance")
        print("3. Evaluate Model Instance")
        print("4. Visualize Results")

        try:
            choice = int(input("Enter the number corresponding to your choice: "))
            if 0 <= choice <= 4:
                return choice
            else:
                print("Invalid selection. Please try again.")
        except ValueError:
            print("Invalid input. Please enter a digit.")

def main():
    # Welcome message
    print("Welcome to SparseyTestingSystem.")

    # Main menu to assign high level function or quit
    while True:
        high_level = printMainMenu()
        if high_level == 0:
            print("Exiting SparseyTestingSystem.")
            break
        elif high_level == 1:
            chooseRunHPO()
        elif high_level == 2:
            chooseTrainModelInstance()
        elif high_level == 3:
            chooseEvaluateModelInstance()
        elif high_level == 4:
            chooseVisualizeResults()
    return

if __name__ == "__main__":
    main()