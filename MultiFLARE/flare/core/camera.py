# Code: https://github.com/fraunhoferhhi/neural-deferred-shading/tree/main
# Modified/Adapted by: Shrisha Bharadwaj, Kelian Baert

from torch import Tensor

class Camera:
    """ Camera in OpenCV format.
        
    Args:
        K (Tensor): Camera matrix with intrinsic parameters (3x3)
        R (Tensor): Rotation matrix (3x3)
        t (Tensor): translation vector (3)
        device (torch.device): Device where the matrices are stored
    """

    def __init__(self, K: Tensor, R: Tensor, t: Tensor, device="cpu"):
        self.K = K.to(device)
        self.R = R.to(device)
        self.t = t.to(device)
        self.device = device

    def to(self, device="cpu"):
        self.K = self.K.to(device)
        self.R = self.R.to(device)
        self.t = self.t.to(device)
        self.device = device
        return self

    @property
    def center(self):
        return -self.R.t() @ self.t
