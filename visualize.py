
import torch
import torchvision.datasets as datasets
import os

from tqdm import tqdm
from torch import nn
from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader

from torchvision.utils import make_grid
import matplotlib.pyplot as plt

from pathlib import Path

# <===- Stat Libs Import -===>
from scipy.stats import norm
import numpy as np

# <===- Model Import -===>
from model import VariationalAutoEncoder

output_dir = Path("reconstructions")
output_dir.mkdir(exist_ok=True)
plot_dir = Path("plots")
plot_dir.mkdir(exist_ok=True)


# <===- Hyperparameters Defined -===>
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INPUT_DIM = 784
H_dim = 200
Z_DIM = 20
NUM_EPOCHS = 10
BATCH_SIZE = 32
LR_RATE = 3e-4

# Loading Dataset
dataset = datasets.MNIST(root="dataset/", train=True, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
example_data, _ = next(iter(train_loader))
in_shape = example_data.shape[1:]
print(in_shape)

import numpy as np
from scipy.stats import norm

def inference(num_examples=1):
    """
    Generates (num_examples) of a particular digit
    """
    model = VariationalAutoEncoder(in_channels=1, latent_dim=20).to(DEVICE)
    model.load_state_dict(torch.load("model.pth", map_location=DEVICE))
    model.eval()

    sample_images = [None] * 10

    # Grab one MNIST image of the desired digit, reshape to [1,1,28,28]

    for digit in range(10):
        for x, y in dataset:
            if y == digit:
                sample_images[digit] = x.unsqueeze(0).to(DEVICE)
                break

    # Reconstruct it via `generate()`
    for digit in range(10):
        with torch.no_grad():
            recon = model.generate(sample_images[digit])   # returns [1,1,28,28]
        save_image(recon, output_dir / f"recon_{digit}.png")    

    # Now draw N samples from the prior with `sample()` and save them
    with torch.no_grad():
        samples = model.sample(num_examples, DEVICE)  # [N,1,28,28]
    for i, s in enumerate(samples):
        save_image(s, output_dir / f"recon_sample_{i}.png")

def plot_2d_manifold(
    model: torch.nn.Module,
    n: int,
    device: torch.device,
    figsize=(8, 8),
    percentile_bounds=(0.05, 0.95)
):
    """
    Visualize a 2D VAE latent manifold by decoding an n×n grid of z-values.
    - model:            Trained VAE with `decode(z: Tensor) -> Tensor`.
    - n:                Grid size (will produce n*n images).
    - device:           torch.device('cuda') or torch.device('cpu').
    - figsize:          Matplotlib figure size.
    - percentile_bounds: Tuple of (low, high) percentiles in (0,1) to span in each z-dimension.
                        e.g. (0.05, 0.95) avoids extreme tails of N(0,1).
    """

    # <==========- Step 1: make an evenly sampled grid of (n*n, 2) -==========>
    # NOTE: the 2 is from the latent being size 2

    low, high = percentile_bounds
    u = np.linspace(low, high, n)
    z_vals_1d = norm.ppf(u)

    # Build an (n*n, 2) grid of latent points
    zz = np.stack(np.meshgrid(z_vals_1d, z_vals_1d), axis=-1)   # shape (n, n, 2)
    z_grid = torch.from_numpy(zz.reshape(-1, 2)).float().to(device)  # (n*n, 2)

    # <===========- Step 2: decode the grid -============>

    model.to(device).eval()
    with torch.no_grad():
        decoded = model.decode(z_grid)   # Tensor [n*n, C, H, W]
    
    # <=========- Step 3: Tile images n×n, and plot -============>

    grid = make_grid(decoded, nrow=n, padding=2, normalize=False, value_range=(-1, 1))
    # moves the channel dimension to last, yielding (H, W, C), and convert for np
    grid_np = grid.permute(1, 2, 0).cpu().numpy()

    plt.figure(figsize=figsize)
    plt.imshow(grid_np, interpolation='nearest', cmap='gray' if grid_np.ndim==2 else None)
    plt.title("Learned 2D Latent Manifold")
    plt.axis('off')
    plt.savefig(plot_dir / f'2d_manifold_{n}x{n}.png')
    plt.show()


def main():
    # inference()
    model = VariationalAutoEncoder(in_channels=1, latent_dim=2)
    model.load_state_dict(torch.load("models/model_2lat.pth", map_location=DEVICE))

    plot_2d_manifold(model, n=30, device=torch.device('cpu'), figsize=(10, 10), percentile_bounds=(0.05, 0.95))


if __name__ == "__main__":
    main()