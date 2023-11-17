import os

class ModelLoader:
    def detectCores(self):
        # Path to the ModelArchitectures directory
        architectures_directory = "ModelArchitectures"

        # List all items in the ModelArchitectures directory
        items = os.listdir(architectures_directory)

        # Filter out directories (which are assumed to be packages)
        core_packages = [item for item in items if os.path.isdir(os.path.join(architectures_directory, item))]

        return core_packages