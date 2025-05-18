import torch
import torchvision.datasets as datasets
import os

from tqdm import tqdm
from torch import nn
from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader

from pathlib import Path

# <===- Model Import -===>
from model import VariationalAutoEncoder

# Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INPUT_DIM = 784
H_dim = 200
Z_DIM = 20
NUM_EPOCHS = 10
BATCH_SIZE = 32
LEARNING_RATE = 3e-4
LATENT_DIM = 2

# Loading Dataset
dataset = datasets.MNIST(root="dataset/", train=True, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# example_data, _ = next(iter(train_loader))
# in_shape = example_data.shape[1:]
# print(in_shape)

output_dir = Path("reconstructions")
output_dir.mkdir(exist_ok=True)

def train_vae(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epochs: int,
    kld_weight: float
) -> dict:
    
    """
    Train a Variational Autoencoder (VAE).
    - model:       VAE instance
    - dataloader:  DataLoader yielding (images, labels) tuples
    - optimizer:   Optimizer for model.parameters()
    - device:      torch.device('cuda') or torch.device('cpu')
    - epochs:      Number of epochs to train
    - kld_weight:  Weight for the KL term (often batch_size / dataset_size)
    - Returns history: Dict with keys 'loss', 'reconstruction', 'kld' mapping to lists
                 of epoch-averaged values.
    """

    history = {'loss': [], 'reconstruction': [], 'kld': []}
    model.to(device)

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        running_recon = 0.0
        running_kld = 0.0

        loop = tqdm(enumerate(dataloader), desc=f"Epoch {epoch}/{epochs}", leave=False)
        for i, (images, _) in loop:
            images = images.to(device)

            optimizer.zero_grad()

            recons, originals, mu, log_var = model(images)

            loss_dict = model.loss_function(recons, originals, mu, log_var, M_N=kld_weight)
            loss = loss_dict['loss']

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_recon += loss_dict['Reconstruction_Loss'].item()
            running_kld   += loss_dict['KLD'].item()

            loop.set_postfix({
                'loss': running_loss/len(dataloader),
                'recon': running_recon/len(dataloader),
                'kld': running_kld/len(dataloader)
            })

        # record epoch averages
        history['loss'].append(running_loss / len(dataloader))
        history['reconstruction'].append(running_recon / len(dataloader))
        history['kld'].append(running_kld / len(dataloader))

    return history

def train():
    model = VariationalAutoEncoder(in_channels=1, latent_dim=LATENT_DIM)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    kld_weight = 128 / len(dataset)  # M_N

    history = train_vae(
        model=model,
        dataloader=train_loader,
        optimizer=optimizer,
        device=device,
        epochs=NUM_EPOCHS,
        kld_weight=kld_weight
    )

    # Save the model
    path = os.path.join("models", f"model_{LATENT_DIM}lat.pth")
    torch.save(model.state_dict(), path)

def main():
    train()

if __name__ == "__main__":
    main()