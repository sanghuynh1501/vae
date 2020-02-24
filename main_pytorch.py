import torch
import os
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms

import numpy as np
import matplotlib.pyplot as plt

from feature_loss import Feature_Loss

lr = 0.01
origin_dim = 784
intermediate_dim = 512
epochs = 50
batch_size = 128
latent_dim = 2
log_interval = 100
use_cuda = False
device = torch.device("cuda" if use_cuda else "cpu")

pm = Feature_Loss(origin_dim, intermediate_dim, 10)
pm.load_state_dict(torch.load("feature.pt"))

# VAE model = encoder + decoder
# build encoder model
class Encoder(torch.nn.Module):
    def __init__(self, origin_dim, intermediate_dim, latent_dim):
        super(Encoder, self).__init__()
        self.linear = torch.nn.Linear(origin_dim, intermediate_dim)
        self.z_mean = torch.nn.Linear(intermediate_dim, latent_dim)
        self.z_log_var = torch.nn.Linear(intermediate_dim, latent_dim)

    def forward(self, x):
        x = F.relu(self.linear(x))
        z_mean = self.z_mean(x)
        z_log_var = self.z_log_var(x)
        return z_mean, z_log_var

# build decoder model
class Decoder(torch.nn.Module):
    def __init__(self, origin_dim, intermediate_dim, latent_dim):
        super(Decoder, self).__init__()
        self.linear = torch.nn.Linear(latent_dim, intermediate_dim)
        self.origin = torch.nn.Linear(intermediate_dim, origin_dim)

    def forward(self, x):
        x = F.relu(self.linear(x))
        origin = torch.sigmoid(self.origin(x))
        return origin

class VAE(torch.nn.Module):
    def __init__(self, origin_dim, intermediate_dim, latent_dim):
        super(VAE, self).__init__()
        self.encoder = Encoder(origin_dim, intermediate_dim, latent_dim)
        self.decoder = Decoder(origin_dim, intermediate_dim, latent_dim)

    def sampling(self, z_mean, z_log_var):
        batch = z_mean.shape[0]
        dim = z_mean.shape[1]
        epsilon = torch.randn(batch, dim)
        return z_mean + torch.exp(0.5 * z_log_var) * epsilon

    def forward(self, x):
        z_mean, z_log_var = self.encoder(x)
        z = self.sampling(z_mean, z_log_var)
        output = self.decoder(z)
        kl_loss = 1 + z_log_var - z_mean ** 2 - torch.exp(z_log_var)
        return output, -0.5 * torch.sum(kl_loss, axis=-1)

def dfc_loss(inputs, outputs, kl_loss):
    pm.eval()
    h1_list = pm(inputs)
    h2_list = pm(outputs)
    
    rc_loss = 0.0
    
    for h1, h2, weight in zip(h1_list, h2_list, [1.0, 1.0]):
        h1 = torch.flatten(h1, 1)
        h2 = torch.flatten(h2, 1)
        rc_loss = rc_loss + weight * torch.sum((h1 - h2) ** 2, axis=-1)
        
    return torch.mean(rc_loss + kl_loss)

def mse_loss(inputs, outputs, kl_loss):
    mse = nn.MSELoss()
    reconstruction_loss = mse(inputs, outputs)
    reconstruction_loss *= origin_dim
    return torch.mean(reconstruction_loss + kl_loss)


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        data = data.reshape(-1, 28 * 28)
        optimizer.zero_grad()
        output, kl_loss = model(data)
        loss = mse_loss(output, data, kl_loss)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            data = data.reshape(-1, 28 * 28)
            output, kl_loss = model(data)
            test_loss += mse_loss(output, data, kl_loss).item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}\n'.format(test_loss))


def plot_results(models,
                 data,
                 batch_size=128,
                 model_name="vae_mnist"):

    encoder, decoder = models
    x_test, y_test = data
    os.makedirs(model_name, exist_ok=True)

    filename = os.path.join(model_name, "vae_mean.png")
    # display a 2D plot of the digit classes in the latent space
    z_mean, _ = encoder(x_test)

    plt.figure(figsize=(12, 10))
    plt.scatter(z_mean[:, 0], z_mean[:, 1], c=y_test)
    plt.colorbar()
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.savefig(filename)
    plt.show()

    filename = os.path.join(model_name, "digits_over_latent.png")
    # display a 30x30 2D manifold of digits
    n = 30
    digit_size = 28
    figure = np.zeros((digit_size * n, digit_size * n))
    # linearly spaced coordinates corresponding to the 2D plot
    # of digit classes in the latent space
    grid_x = np.linspace(-4, 4, n)
    grid_y = np.linspace(-4, 4, n)[::-1]

    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = np.array([[xi, yi]])
            x_decoded = decoder(torch.from_numpy(z_sample).float())
            digit = x_decoded[0].reshape(digit_size, digit_size)
            figure[i * digit_size: (i + 1) * digit_size,
                   j * digit_size: (j + 1) * digit_size] = digit

    plt.figure(figsize=(10, 10))
    start_range = digit_size // 2
    end_range = (n - 1) * digit_size + start_range + 1
    pixel_range = np.arange(start_range, end_range, digit_size)
    sample_range_x = np.round(grid_x, 1)
    sample_range_y = np.round(grid_y, 1)
    plt.xticks(pixel_range, sample_range_x)
    plt.yticks(pixel_range, sample_range_y)
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.imshow(figure, cmap='Greys_r')
    plt.savefig(filename)
    plt.show()


train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=True, download=True,
                      transform=transforms.Compose([
                          transforms.ToTensor()
                      ])),
        batch_size=batch_size, shuffle=True)

test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=False,
                      transform=transforms.Compose([
                          transforms.ToTensor()
                      ])),
        batch_size=batch_size, shuffle=True)


model = VAE(origin_dim, intermediate_dim, latent_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)
for epoch in range(1, epochs + 1):
    train(model, device, train_loader, optimizer, epoch)
    test(model, device, test_loader)


test_data = torch.zeros(1, 28 * 28)
test_label = np.array([])
dem = 0
for data, target in test_loader:
    data = data.reshape(-1, 28 * 28)
    if len(test_data) == 1:
        test_data = data
        test_label = target.numpy()
    else:
        test_data = torch.cat((test_data, data), axis=0)
        test_label = np.concatenate((test_label, target.numpy()), axis=0)
    if dem > 10:
        break

with torch.no_grad():
    plot_results([model.encoder, model.decoder], [test_data, test_label])