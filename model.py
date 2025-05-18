"""
EECS 445 - Introduction to Machine Learning
Winter 2025 - HW4 - q4_models.py
Implement the Autoencoder class
"""
import torch
import torch.nn as nn
from typing import List
from torch.nn import functional as F

class VariationalAutoEncoder(nn.Module):
    def __init__(self, 
                 in_channels: int,
                 latent_dim: int, 
                 hidden_dims: List = None,
                **kwargs):
        """
        Note: modified to work in black/white images
        - in_shape: shape of the input image (height, width)
        - latent_dim: size of the latent representation
        """

        super(VariationalAutoEncoder, self).__init__()

        self.latent_dim = latent_dim


        # NOTE: There are 5 hidden layers 
        if not hidden_dims:
            hidden_dims = [32, 64, 128, 256, 512]

        # <===========- ENCODER -===========>
        modules = []

        # Build the encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels=in_channels,
                              out_channels=h_dim,
                              kernel_size=3,
                              stride=2,
                              padding = 1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU()
                    )
                )
            in_channels = h_dim

        # Assign the encoder and fc layers
        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1] * 1, latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1] * 1, latent_dim)


        # <==========- DECODER -===========>
        modules = []

        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * 1)

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        in_channels=hidden_dims[i],
                        out_channels=hidden_dims[i + 1],
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]), 
                    nn.LeakyReLU())
            )


        self.decoder = nn.Sequential(*modules)

        # <==========- FINAL LAYER -===========>
        modules = []

        modules.append(
            nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels=hidden_dims[-1],
                    out_channels=hidden_dims[-1],
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1),
                nn.BatchNorm2d(hidden_dims[-1]),
                nn.LeakyReLU(),
                nn.Conv2d(in_channels=hidden_dims[-1],
                          out_channels=1,
                          kernel_size=5,
                          padding=0),
                nn.Tanh(),

            )
        )

        self.final_layer = nn.Sequential(*modules)
        
    def encode(self, input: torch.Tensor) -> torch.Tensor:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        - param input: (Tensor) Input tensor to encoder [N x C x H x W]
        - return: (Tensor) List of latent codes
        """
       
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Maps the given latent codes
        onto the image space.
        - param z: (Tensor) [B x D]
        - return: (Tensor) [B x C x H x W]
        """
        
        result = self.decoder_input(z)
        result = result.view(-1, 512, 1, 1)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        - param mu: (Tensor) Mean of the latent Gaussian [B x D]
        - param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        - return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the autoencoder which encodes and decodes the input image
        - x: input image
        - Returns: [self.decode(z), input, mu, log_var]
        """
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return [self.decode(z), input, mu, log_var]
    
    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        """
        Computes the VAE loss function.
        KL(N(mu, sigma), N(0, 1)) = log 1/sigma + (sigma^2 + mu^2)/2 - 1/2
        """
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]

        kld_weight = kwargs['M_N'] # Account for the minibatch samples from the dataset
        recons_loss =F.mse_loss(recons, input)


        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

        loss = recons_loss + kld_weight * kld_loss
        return {'loss': loss, 'Reconstruction_Loss':recons_loss.detach(), 'KLD':-kld_loss.detach()}

    def sample(self,
               num_samples: int,
               current_device: int, 
               **kwargs) -> torch.Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        - param num_samples: (Int) Number of samples
        - param current_device: (Int) Device to run the model
        - return: (Tensor)
        """
        z = torch.randn(num_samples,
                        self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def generate(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Given an input image x, returns the reconstructed image
        - param x: (Tensor) [B x C x H x W]
        - return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]

def test():
    """
    Test function to make sure the dimension of the layers match up
    """
    # Simulated MNIST input shape
    batch_size = 32
    in_channels = 1 
    in_height = 28
    in_width = 28
    latent_dim = 20

    model = VariationalAutoEncoder(in_channels=in_channels, latent_dim=latent_dim)
    dummy_input = torch.randn(batch_size, in_channels, in_height, in_width)
    reconstructed, original, mu, log_var = model(dummy_input)

    print("Encoder Layers:")
    for block_idx, block in enumerate(model.encoder, start=1):
        print(f" Block {block_idx}:")
        for layer in block:
            name = layer.__class__.__name__
            if isinstance(layer, nn.Conv2d):
                print(f"    • {name:16} in={layer.in_channels:3}  out={layer.out_channels:3}  "
                    f"kernel={layer.kernel_size}  stride={layer.stride}  pad={layer.padding}")
            elif isinstance(layer, nn.BatchNorm2d):
                print(f"    • {name:16} num_features={layer.num_features:3}")
            elif isinstance(layer, nn.LeakyReLU):
                print(f"    • {name:16}")

    print("Decoder Layers:")
    for block_idx, block in enumerate(model.decoder, start=1):
        print(f" Block {block_idx}:")
        for layer in block:
            name = layer.__class__.__name__
            if isinstance(layer, nn.ConvTranspose2d):
                print(f"    • {name:16} in={layer.in_channels:3}  out={layer.out_channels:3}  "
                    f"kernel={layer.kernel_size}  stride={layer.stride}  "
                    f"pad={layer.padding}  out_pad={layer.output_padding}")
            elif isinstance(layer, nn.BatchNorm2d):
                print(f"    • {name:16} num_features={layer.num_features:3}")
            elif isinstance(layer, nn.LeakyReLU):
                print(f"    • {name:16}")

    print("\nFinal Layer:")
    for block_idx, block in enumerate(model.final_layer, start=1):
        print(f" Block {block_idx}:")
        for layer in block:
            name = layer.__class__.__name__
            if isinstance(layer, nn.ConvTranspose2d):
                print(f"    • {name:16} in={layer.in_channels:3}  out={layer.out_channels:3}  "
                    f"kernel={layer.kernel_size}  stride={layer.stride}  "
                    f"pad={layer.padding}  out_pad={layer.output_padding}")
            elif isinstance(layer, nn.Conv2d):
                print(f"    • {name:16} in={layer.in_channels:3}  out={layer.out_channels:3}  "
                    f"kernel={layer.kernel_size}  pad={layer.padding}")
            elif isinstance(layer, nn.BatchNorm2d):
                print(f"    • {name:16} num_features={layer.num_features:3}")
            elif isinstance(layer, nn.LeakyReLU) or isinstance(layer, nn.Tanh):
                print(f"    • {name:16}")

if __name__ == "__main__":
    test()
