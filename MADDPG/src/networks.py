import os

import torch


class Critic(torch.nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, agent_name, checkpoint_directory):
        super(Critic, self).__init__()
        self.agent_name = agent_name
        
        self.linear1 = torch.nn.Linear(input_size, hidden_size1)
        self.linear2 = torch.nn.Linear(hidden_size1, hidden_size2)
        self.linear3 = torch.nn.Linear(hidden_size2, 1)
        
        if torch.backends.mps.is_available():
            self.DEVICE = torch.device(device="mps")
        elif torch.cuda.is_available():
            self.DEVICE = torch.device(device="cuda")
        else:
            self.DEVICE = torch.device(device="cpu")
        
        self.to(self.DEVICE)
        
        self.checkpoint_path = os.path.join(checkpoint_directory, agent_name+'.pt')
    
    def forward(self, x):
        x = torch.nn.functional.relu(self.linear1(x))
        x = torch.nn.functional.relu(self.linear2(x))
        x = self.linear3(x)
        
        return x
    
    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_path)
    
    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_path))


class Actor(torch.nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size, agent_name, checkpoint_directory):
        super(Actor, self).__init__()
        self.agent_name = agent_name
        
        self.linear1 = torch.nn.Linear(input_size, hidden_size1)
        self.linear2 = torch.nn.Linear(hidden_size1, hidden_size2)
        self.mu_linear3 = torch.nn.Linear(hidden_size2, output_size)
        self.sigma_linear3 = torch.nn.Linear(hidden_size2, output_size)
        
        if torch.backends.mps.is_available():
            self.DEVICE = torch.device(device="mps")
        elif torch.cuda.is_available():
            self.DEVICE = torch.device(device="cuda")
        else:
            self.DEVICE = torch.device(device="cpu")
        
        self.to(self.DEVICE)
        
        self.checkpoint_path = os.path.join(checkpoint_directory, agent_name+'.pt')
    
    def forward(self, x):
        x = torch.nn.functional.relu(self.linear1(x))
        x = torch.nn.functional.relu(self.linear2(x))
        mu = torch.nn.functional.sigmoid(self.mu_linear3(x))
        sigma = torch.nn.functional.softplus(self.sigma_linear3(x))
        
        return mu, sigma
    
    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_path)
    
    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_path))