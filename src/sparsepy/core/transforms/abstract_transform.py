import torch.nn as nn

class AbstractTransform(nn.Module):
    """
    An abstract class representing a transformation. All transformations
    should inherit from this class and implement the forward method.
    """
    def __init__(self):
        super().__init__()

    def forward(self, sample):
        """
        Defines the computation performed at every call. Should be overridden by all subclasses.
        
        :param sample: The input data to be transformed.
        :return: Transformed data.
        """
        raise NotImplementedError("Subclasses should implement this method")