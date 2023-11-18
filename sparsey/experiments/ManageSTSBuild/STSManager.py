from SystemTasks.ModelManager.ModelLoader import ModelLoader

class STSManager:
    def coreLookup(self):
        # Create an instance of ModelLoader
        model_loader = ModelLoader()

        # Use the detectCores method from the ModelLoader instance
        return model_loader.detectCores()