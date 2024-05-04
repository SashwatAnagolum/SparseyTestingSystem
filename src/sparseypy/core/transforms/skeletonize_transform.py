import torch
from skimage import feature, morphology
from .abstract_transform import AbstractTransform

class SkeletonizeTransform(AbstractTransform):
    """
    A transform to apply Canny edge detection followed by skeletonization.
    """
    def __init__(self, sigma=3.0):
        """
        Initialize the CannySkeletonizeTransform.

        :param sigma: The standard deviation of the Gaussian filter used in Canny edge detector.
        """
        super().__init__()
        self.sigma = sigma

    def forward(self, sample):
        """
        Apply Canny edge detection and skeletonization to the input sample.

        :param sample: The input data to be transformed, assumed to be a PyTorch tensor.
        :return: Skeletonized edge data.
        """
        # Ensure the input is a CPU numpy array
        # may not work correctly on MPS since Torch doesn't have an equivalent sensing method
        if sample.is_cuda:
            sample = sample.cpu()
        sample_np = sample.squeeze().numpy()  # Assuming the input is grayscale [1, Height, Width]

        # Apply Canny edge detection
        edges = feature.canny(sample_np, sigma=self.sigma)

        # Apply skeletonization
        skeleton = morphology.skeletonize(edges).astype(float)

        # Convert back to PyTorch tensor
        return torch.from_numpy(skeleton).unsqueeze(0)

