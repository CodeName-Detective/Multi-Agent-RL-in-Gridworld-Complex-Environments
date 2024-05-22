import random
from collections import deque

import numpy as np
import torch


class MultiAgent_ReplayBuffer:
    ### Modify
    def __init__(self, buffer_size, agents, sample_size):
        self.buffer_size = buffer_size  
        self.current_buffer_length = 0
        self.sample_size = sample_size
        
        self.agents = agents
        
        self.actor_buffer_transistion_schema = ['observation', 'action', 'new_observation']
        self.critic_buffer_transistion_schema = ['env_observation', 'env_new_observation', 'reward', 'terminal_status_buffer']
        
        self.agent_buffers = {}
        
        for agent in agents:
            self.agent_buffers[agent] = {}
            for field in self.actor_buffer_transistion_schema:
                self.agent_buffers[agent][field] = deque(maxlen=self.buffer_size)
        
        self.critic_buffers = {}
        for agent in agents:
            self.critic_buffers[agent] = {}
            for field in self.critic_buffer_transistion_schema:
                self.critic_buffers[agent][field] = deque(maxlen=self.buffer_size)
                
                
    def store_transistion(self, observation, action, new_observation, reward, termination_status, truncation_status):
        
        env_observation = self.observations_to_env_observations_vector(observation)
        new_env_observation = self.observations_to_env_observations_vector(new_observation)
        
        
        for agent in self.agents:
            # Store in Actor Buffer
            self.agent_buffers[agent]['observation'].append(observation[agent])
            self.agent_buffers[agent]['action'].append(action[agent])
            self.agent_buffers[agent]['new_observation'].append(new_observation[agent])
            
            # Store in Critic Buffer
            self.critic_buffers[agent]['env_observation'].append(env_observation)
            self.critic_buffers[agent]['env_new_observation'].append(new_env_observation)
            self.critic_buffers[agent]['reward'].append(reward[agent])
            self.critic_buffers[agent]['terminal_status_buffer'].append(termination_status[agent] or truncation_status[agent])
        
        if self.current_buffer_length != self.buffer_size:
            self.current_buffer_length += 1
    
    def observations_to_env_observations_vector(self, observation):
        state_observation = np.array([])
        for agent in self.agents:
            state_observation = np.concatenate([state_observation, observation[agent]])
        
        return state_observation
    
    def sample_buffer(self):
        
        actor_sample = {}
        
        critic_sample = {}
        
        sample_batch = random.sample(range(self.current_buffer_length), self.sample_size)
        
        for agent in self.agents:
            # Sample Actor Buffer
            actor_sample[agent] = {}
            actor_sample[agent]['observation'] = torch.tensor(np.array(self.agent_buffers[agent]['observation'])[sample_batch], dtype=torch.float32)
            actor_sample[agent]['action'] = torch.tensor(np.array(self.agent_buffers[agent]['action'])[sample_batch], dtype=torch.float32)
            actor_sample[agent]['new_observation'] = torch.tensor(np.array(self.agent_buffers[agent]['new_observation'])[sample_batch], dtype=torch.float32)
            
            # Sample Critic Buffer
            critic_sample[agent] = {}
            critic_sample[agent]['env_observation'] = torch.tensor(np.array(self.critic_buffers[agent]['env_observation'])[sample_batch], dtype=torch.float32)
            critic_sample[agent]['env_new_observation'] = torch.tensor(np.array(self.critic_buffers[agent]['env_new_observation'])[sample_batch], dtype=torch.float32)
            critic_sample[agent]['reward'] = torch.tensor(np.array(self.critic_buffers[agent]['reward'])[sample_batch], dtype=torch.float32)
            critic_sample[agent]['terminal_status_buffer'] = torch.tensor(np.array(self.critic_buffers[agent]['terminal_status_buffer'])[sample_batch], dtype=torch.float32)
        
        return actor_sample, critic_sample 
    
    def ready_to_be_sampled(self):
        if self.current_buffer_length >= self.sample_size:
            return True
        else:
            return False