Repository: ujjwal895/ocl-thesis-framework
Files analyzed: 3

Estimated tokens: 984

Directory structure:
└── ujjwal895-ocl-thesis-framework/
    ├── agents/
    │   └── dgr_agent.py
    ├── config/
    │   └── dgr_config.yaml
    └── models/
        └── generator.py


================================================
FILE: agents/dgr_agent.py
================================================

import torch
import torch.nn.functional as F
from agents.base import BaseAgent
from models.generator import CVAE

class DGRAgent(BaseAgent):
    def __init__(self, model, args):
        super().__init__(model, args)
        self.args = args
        self.device = args.device if hasattr(args, "device") else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.generator = CVAE(args).to(self.device)
        self.gen_optimizer = torch.optim.Adam(self.generator.parameters(), lr=args.gen_lr)
        self.generator.trained = False

    def observe(self, data):
        x, y = data
        x, y = x.to(self.device), y.to(self.device)

        loss = self.observe_batch(x, y)
        self.train_generator(x, y)

        if self.generator.trained:
            gen_x, gen_y = self.generator.sample(self.args.replay_batch_size)
            gen_loss = self.observe_batch(gen_x, gen_y)
            loss += gen_loss

        return loss

    def observe_batch(self, x, y):
        self.model.train()
        output = self.model(x)
        loss = self.criterion(output, y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss

    def train_generator(self, x, y):
        self.generator.train()
        self.gen_optimizer.zero_grad()
        loss = self.generator.loss_function(x, y)
        loss.backward()
        self.gen_optimizer.step()
        self.generator.trained = True



================================================
FILE: config/dgr_config.yaml
================================================

agent: DGR
dataset: splitMNIST
model: MLP
batch_size: 64
input_dim: 784
latent_dim: 100
num_classes: 10
gen_lr: 0.0002
replay_batch_size: 64
optimizer: SGD
learning_rate: 0.1
epochs: 1



================================================
FILE: models/generator.py
================================================

import torch
import torch.nn as nn
import torch.nn.functional as F

class CVAE(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.latent_dim = args.latent_dim
        self.input_dim = args.input_dim
        self.num_classes = args.num_classes
        self.trained = False

        self.fc1 = nn.Linear(self.input_dim + self.num_classes, 400)
        self.fc21 = nn.Linear(400, self.latent_dim)
        self.fc22 = nn.Linear(400, self.latent_dim)

        self.fc3 = nn.Linear(self.latent_dim + self.num_classes, 400)
        self.fc4 = nn.Linear(400, self.input_dim)

    def encode(self, x, y):
        inputs = torch.cat([x, F.one_hot(y, self.num_classes).float()], dim=1)
        h1 = F.relu(self.fc1(inputs))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, y):
        inputs = torch.cat([z, F.one_hot(y, self.num_classes).float()], dim=1)
        h3 = F.relu(self.fc3(inputs))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x, y):
        mu, logvar = self.encode(x, y)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, y), mu, logvar

    def loss_function(self, x, y):
        recon_x, mu, logvar = self.forward(x, y)
        BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE + KLD

    def sample(self, batch_size):
        device = next(self.parameters()).device
        z = torch.randn(batch_size, self.latent_dim).to(device)
        y = torch.randint(0, self.num_classes, (batch_size,), device=device)
        gen_x = self.decode(z, y)
        return gen_x, y

