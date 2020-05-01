import torch
import torch.nn as nn
import torch.nn.functional as F
# import torch
# from attention.attention import *
# sensor = GlimpseNetwork(4, 4, [128, 64, 256], 3)
# img = torch.rand(2, 3, 64, 64)
# l = torch.rand(2, 2) * 2 - 1
# sensor(img, l)

class GlimpseSensor(nn.Module):
    def __init__(self, initial_patch_size, num_patches, scaling_factor):
        """
        Args
        ------
        - initial_patch_size: smallest patch size
        - num_patches: number of patches
        - scaling_factor: patch size increase factor

        """
        super(GlimpseSensor, self).__init__()
        self.initial_patch_size = initial_patch_size
        self.num_patches = num_patches
        self.scaling_factor = scaling_factor

    def forward(self, x, l):
        """
        Returns multi-resolution patches of the input image centered at l

        Args
        ------
        - x: 4d image tensor of shape (B, C, H, W)
        - l: 2d tensor of shape (B, 2)
        """

        glimpses = torch.empty(x.size(0), self.num_patches, x.size(1), self.initial_patch_size, self.initial_patch_size)

        for i in range(self.num_patches):
            size = self.initial_patch_size * self.scaling_factor**i
            glimpses[:, i, :, :, :] = self.extract_patch(x, size, l, self.initial_patch_size)
        return glimpses

    def extract_patch(self, x, size, l, downsampled_size):
        """
        Extracts a patch of specified resolution from the image 
        centered at location l

        Args
        ------
        - x: 4d image tensor of shape (B, C, H, W)
        - size: patch size
        - l: 2d tensor of shape (B, 2); first coord corresponds to H dim
        - downsampled_size: size of image after downsampling
        """
        padded = F.pad(x, (size, size, size, size), "constant", 0)
        l = self.denormalize(x.size(2), l, size)
        left_corners = l - (size//2)

        batch_size = x.size(0)
        patches = torch.empty(batch_size, x.size(1), downsampled_size, downsampled_size)
        for i in range(batch_size):
            orig_patch = padded[i, :, left_corners[i, 0].item():left_corners[i, 0].item() + size, left_corners[i, 1].item():left_corners[i, 1].item() + size].unsqueeze(0)
            patches[i] = F.interpolate(orig_patch, (downsampled_size, downsampled_size))[0]
        return patches

    def denormalize(self, side_len, coords, padding):
        """
        Converts coordinates to the proper range

        Args
        ------
        - side_len: size of one side of the image
        - coords: tensor of shape (2, )
        - padding: amount of padding added
        """
        return (0.5 * ((coords + 1.0) * side_len) + padding).long()

class GlimpseNetwork(nn.Module):
    def __init__(self, initial_patch_size, num_patches, hidden_dims, in_channels, scaling_factor=2):
        """
        Args
        ------
        - initial_patch_size: smallest patch size
        - num_patches: number of patches
        - hidden_dims: List of hidden dimensions sizes ordered as retina, location, and glimpse layer dimensions
        - scaling_factor: patch size increase factor
        """
        super(GlimpseNetwork, self).__init__()
        self.retina = GlimpseSensor(initial_patch_size, num_patches, scaling_factor)
        self.retina_layer = nn.Sequential(nn.Linear(in_channels*num_patches*initial_patch_size**2 , hidden_dims[0]), nn.LeakyReLU(0.2, inplace=True))
        self.location_layer = nn.Sequential(nn.Linear(2, hidden_dims[1]), nn.LeakyReLU(0.2, inplace=True))
        self.glimpse_layer = nn.Linear(hidden_dims[0] + hidden_dims[1], hidden_dims[2])

    def forward(self, x_t, l_tm1):
        batch_size = x_t.size(0)
        patches = self.retina(x_t, l_tm1)
        patches = patches.reshape(batch_size, -1)
        g_t = self.glimpse_layer(torch.cat([self.retina_layer(patches), self.location_layer(l_tm1)], dim=1))
        return g_t


