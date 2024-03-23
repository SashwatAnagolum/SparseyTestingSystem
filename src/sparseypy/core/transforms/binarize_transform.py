import torch
from .abstract_transform import AbstractTransform

class BinarizeTransform(AbstractTransform):
    """
    A transform to first convert an image to grayscale and then
    binarize it based on a threshold.
    """
    def __init__(self, binarize_threshold=0.5):
        """
        Initialize the BinarizeTransform.

        :param threshold: The threshold for binarization. Default is 0.5.
        """
        super().__init__()
        self.threshold = binarize_threshold

    def forward(self, sample):
        """
        Apply grayscale conversion and binarization to the input sample.

        :param sample: The input data to be transformed, assumed to be a PyTorch tensor.
        :return: Binarized grayscale data.
        """
        # Check if the image has more than one channel (i.e., is not already grayscale)
        if sample.dim() == 3 and sample.size(0) > 1:  # Assuming [Channels, Height, Width]
            # Convert to grayscale by averaging across the color channels
            sample = torch.mean(sample, dim=0, keepdim=True)

        # Binarize the grayscale image using the specified threshold
        return (sample > self.threshold).float()
