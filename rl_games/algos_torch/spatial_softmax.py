import torch
import torch.nn as nn
import torch.nn.functional as F


# Adopted from https://gist.github.com/kevinzakka/dd9fa5177cda13593524f4d71eb38ad5
class SpatialSoftArgmax(nn.Module):
    """Spatial softmax as defined in [1].

    Concretely, the spatial softmax of each feature
    map is used to compute a weighted mean of the pixel
    locations, effectively performing a soft arg-max
    over the feature dimension.

    References:
        [1]: End-to-End Training of Deep Visuomotor Policies,
        https://arxiv.org/abs/1504.00702
    """

    def __init__(self, normalize=False):
        """Constructor.

        Args:
            normalize (bool): Whether to use normalized
                image coordinates, i.e. coordinates in
                the range `[-1, 1]`.
        """
        super().__init__()

        self.normalize = normalize

    def _coord_grid(self, h, w, device):
        if self.normalize:
            return torch.stack(
                torch.meshgrid(
                    torch.linspace(-1, 1, w, device=device),
                    torch.linspace(-1, 1, h, device=device),
                )
            )
        return torch.stack(
            torch.meshgrid(
                torch.arange(0, w, device=device),
                torch.arange(0, h, device=device),
            )
        )

    def forward(self, x):
        assert x.ndim == 4, "Expecting a tensor of shape (B, C, H, W)."

        # compute a spatial softmax over the input:
        # given an input of shape (B, C, H, W),
        # reshape it to (B*C, H*W) then apply
        # the softmax operator over the last dimension
        b, c, h, w = x.shape
        softmax = F.softmax(x.reshape(-1, h * w), dim=-1)

        # create a meshgrid of pixel coordinates
        # both in the x and y axes
        xc, yc = self._coord_grid(h, w, x.device)

        # element-wise multiply the x and y coordinates
        # with the softmax, then sum over the h*w dimension
        # this effectively computes the weighted mean of x
        # and y locations
        x_mean = (softmax * xc.flatten()).sum(dim=1, keepdims=True)
        y_mean = (softmax * yc.flatten()).sum(dim=1, keepdims=True)

        # concatenate and reshape the result
        # to (B, C*2) where for every feature
        # we have the expected x and y pixel
        # locations
        return torch.cat([x_mean, y_mean], dim=1).view(-1, c * 2)
      
      
if __name__ == "__main__":
    b, c, h, w = 32, 64, 12, 12
    x = torch.zeros(b, c, h, w)
    true_max = torch.randint(0, 10, size=(b, c, 2))
    for i in range(b):
      for j in range(c):
          x[i, j, true_max[i, j, 0], true_max[i, j, 1]] = 1000
    soft_max = SpatialSoftArgmax()(x).reshape(b, c, 2)
    assert torch.allclose(true_max.float(), soft_max)