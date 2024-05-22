import numpy as np
import torch

from src.networks import Actor, Critic


class Agent:
    def __init__(self, actor_input_shape, critic_input_shape, n_actions, agent_name,
                 actor_lr, critic_lr, hidden_size1, hidden_size2,
                 discount_factor, tau, checkpoint_directory):
        
        self.discount_factor = discount_factor
        self.tau = tau
        self.n_actions = n_actions
        self.agent_name = agent_name
        
        self.optimizers = {}
        
        self.actor = Actor(actor_input_shape, hidden_size1, hidden_size2, n_actions, agent_name, checkpoint_directory)
        self.critic = Critic(critic_input_shape, hidden_size1, hidden_size2, agent_name+'_critic', checkpoint_directory)
        
        self.optimizers['actor'] = torch.optim.Adam(self.actor.parameters(), lr = actor_lr)
        self.optimizers['critic'] = torch.optim.Adam(self.critic.parameters(), lr= critic_lr)
        
        self.target_actor = Actor(actor_input_shape, hidden_size1, hidden_size2, n_actions, agent_name+'_target', checkpoint_directory)
        self.target_critic = Critic(critic_input_shape, hidden_size1, hidden_size2, agent_name+'_critic_target', checkpoint_directory)
        
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())
    
    def synchronize(self):
        
        target_actor_state_dict = self.target_actor.state_dict()
        actor_state_dict = self.actor.state_dict()
        
        for key in actor_state_dict:
            target_actor_state_dict[key] = actor_state_dict[key]*self.tau + target_actor_state_dict[key]*(1-self.tau)
        
        self.target_actor.load_state_dict(target_actor_state_dict)
        
        
        target_critic_state_dict = self.target_critic.state_dict()
        critic_state_dict = self.critic.state_dict()
        
        for key in critic_state_dict:
            target_critic_state_dict[key] = critic_state_dict[key]*self.tau + target_critic_state_dict[key]*(1-self.tau)
        
        self.target_critic.load_state_dict(target_critic_state_dict)
    
    def choose_action(self, agent_observation, lower_bound, upper_bound, stochasticity = False, evaluate= False):
        agent_observation = torch.tensor(np.array([agent_observation]), dtype=torch.float32).to(self.actor.DEVICE)
        mu, sigma = self.actor.forward(agent_observation)
        
        if stochasticity:
            eps = 1e-6
            action_dist = torch.distributions.normal.Normal(mu, sigma+eps)
            actions = action_dist.sample()
        else:
            actions = mu
        
        # Adding Noise
        if not evaluate:
            noise = torch.rand(actions.shape).to(self.actor.DEVICE)
            actions = actions+ noise*(1-int(evaluate))
        
        actions = torch.clamp(actions, float(lower_bound), float(upper_bound))
        
        return actions.squeeze(0).detach().cpu().numpy()
    
    def load_models(self):
        self.actor.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.critic.load_checkpoint()
        self.target_critic.load_checkpoint()
    
    def save_models(self):
        self.actor.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.critic.save_checkpoint()
        self.target_critic.save_checkpoint()