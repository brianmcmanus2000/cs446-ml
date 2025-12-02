import torch
import torch.nn as nn
import torch.optim as optim


class LinearAE(nn.Module):
    def __init__(self, d_input: int, d_hidden: int):
        super().__init__()
        self.encoder = nn.Linear(d_input, d_hidden, bias=False)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return z @ self.encoder.weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))


def autoencode(data: torch.Tensor):
    x = data  # do NOT cast to float32
    x_mean = x.mean(dim=0, keepdim=True)
    x_centered = x - x_mean


    N, d_input = x_centered.shape
    d_hidden = 2  

    model = LinearAE(d_input, d_hidden).to(dtype=x.dtype)
    optimizer = optim.Adam(model.parameters(), lr=1e-2)
    criterion = nn.MSELoss()

    num_epochs = 500
    model.train()
    for _ in range(num_epochs):
        optimizer.zero_grad()
        recon = model(x_centered)
        loss = criterion(recon, x_centered)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        ae_components = model.encode(x_centered)

    return ae_components
