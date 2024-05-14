import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import imageio
import os

# Parameters
alpha = 0.01  # Thermal diffusivity
N_u = 100  # Number of initial and boundary points
N_f = 10000  # Number of collocation points for PDE

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Domain bounds
lb = np.array([0.0, 0.0])
ub = np.array([1.0, 1.0])

# Initial and boundary data
X_u_train = np.vstack([np.zeros((50, 2)), np.ones((50, 2))])
X_u_train[:, 1] = np.linspace(0, 1, 100)
u_train = np.sin(np.pi * X_u_train[:, 1])  # Initial condition: u(x,0) = sin(pi*x)

# Convert to tensors
X_u_train = torch.tensor(X_u_train, dtype=torch.float32, requires_grad=True).to(device)
u_train = torch.tensor(u_train, dtype=torch.float32).view(-1, 1).to(device)

# Collocation points
X_f_train = lb + (ub - lb) * np.random.rand(N_f, 2)
X_f_train = torch.tensor(X_f_train, dtype=torch.float32, requires_grad=True).to(device)

# Neural Network
class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 20),
            nn.Tanh(),
            nn.Linear(20, 20),
            nn.Tanh(),
            nn.Linear(20, 1)
        )
    
    def forward(self, x):
        return self.net(x)
    
    def net_f(self, x):
        """ The physics informed part of the loss, representing the PDE """
        u = self.forward(x)
        u_t = torch.autograd.grad(u.sum(), x, create_graph=True)[0][:, 1:2]
        u_x = torch.autograd.grad(u.sum(), x, create_graph=True)[0][:, 0:1]
        u_xx = torch.autograd.grad(u_x.sum(), x, create_graph=True)[0][:, 0:1]
        return u_t - alpha * u_xx

# Initialize model
model = PINN().to(device)

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Directory to save figures
save_dir = "heat_eqn_frames"
os.makedirs(save_dir, exist_ok=True)

def save_plot(epoch):
    model.eval()
    with torch.no_grad():
        x = np.linspace(0, 1, 100)
        y = np.linspace(0, 1, 100)
        X, Y = np.meshgrid(x, y)
        grid = np.vstack([X.ravel(), Y.ravel()]).T
        grid_tensor = torch.tensor(grid, dtype=torch.float32, device=device)
        U_pred = model(grid_tensor).cpu().numpy().reshape(100, 100)
    plt.figure(figsize=(10, 8))
    cp = plt.pcolormesh(X, Y, U_pred, cmap='hot')
    plt.colorbar(cp)
    plt.title(f'Epoch {epoch}')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig(f"{save_dir}/frame_{epoch:04d}.png")
    plt.close()

# Training loop with loss collection
def train(model, X_u_train, u_train, X_f_train, epochs, save_every):
    losses = []
    for epoch in range(epochs):
        optimizer.zero_grad()
        u_pred = model(X_u_train)
        f_pred = model.net_f(X_f_train)
        loss_u = nn.MSELoss()(u_pred, u_train)
        loss_f = torch.mean(f_pred ** 2)
        loss = loss_u + loss_f
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        
        if epoch % save_every == 0:
            print(f'Epoch {epoch}, Loss: {loss.item()}')
            save_plot(epoch)
    return losses

losses = train(model, X_u_train, u_train, X_f_train, 1000, 100)

# Plot training loss
plt.figure(figsize=(10, 5))
plt.plot(losses, label='Training Loss')
plt.title('Training Loss vs. Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

# Generate GIF
images = []
for filename in sorted(os.listdir(save_dir)):
    if filename.endswith('.png'):
        images.append(imageio.imread(f"{save_dir}/{filename}"))
imageio.mimsave('heat_equation_training.gif', images, fps=2)
