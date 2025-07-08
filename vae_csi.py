import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Simulate CSI data (complex channel matrices)
def generate_csi_data(num_samples, num_antennas):
    # Real and imaginary parts as separate channels
    real_part = np.random.randn(num_samples, num_antennas, num_antennas)
    imag_part = np.random.randn(num_samples, num_antennas, num_antennas)
    return np.stack([real_part, imag_part], axis=1)  # Shape: (samples, 2, antennas, antennas)

# VAE Model
class CSI_VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(CSI_VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(2, 8, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        
        # Calculate flattened size after convolutions
        conv_out_size = input_dim // 4  # After two stride=2 convolutions
        self.fc_mu = nn.Linear(16 * conv_out_size * conv_out_size, latent_dim)
        self.fc_logvar = nn.Linear(16 * conv_out_size * conv_out_size, latent_dim)
        
        self.decoder_fc = nn.Linear(latent_dim, 16 * conv_out_size * conv_out_size)
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (16, conv_out_size, conv_out_size)),
            nn.ConvTranspose2d(16, 8, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 2, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Tanh()  # Output in range [-1,1] matching normalized input
        )
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std
    
    def forward(self, x):
        # Encoder
        enc = self.encoder(x)
        mu, logvar = self.fc_mu(enc), self.fc_logvar(enc)
        z = self.reparameterize(mu, logvar)
        
        # Decoder
        dec_in = self.decoder_fc(z)
        reconstructed = self.decoder(dec_in)
        return reconstructed, mu, logvar

# Training function
def train_vae(model, data, epochs=100, batch_size=32, lr=0.001):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        # Mini-batch training
        for i in range(0, len(data), batch_size):
            batch = data[i:i+batch_size]
            batch = torch.tensor(batch, dtype=torch.float32)
            
            # Forward pass
            reconstructed, mu, logvar = model(batch)
            
            # Loss = reconstruction loss + KL divergence
            recon_loss = criterion(reconstructed, batch)
            kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            loss = recon_loss + kl_div
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        if (epoch+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(data):.4f}')

# Main execution
if __name__ == "__main__":
    # Parameters
    num_samples = 1000
    num_antennas = 8  # For 8x8 MIMO system
    latent_dim = 16
    
    # Generate data
    data = generate_csi_data(num_samples, num_antennas)
    
    # Initialize model
    model = CSI_VAE(num_antennas, latent_dim)
    
    # Train model
    train_vae(model, data)
    
    print("VAE training completed!")
