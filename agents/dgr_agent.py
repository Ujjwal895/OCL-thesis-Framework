
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
