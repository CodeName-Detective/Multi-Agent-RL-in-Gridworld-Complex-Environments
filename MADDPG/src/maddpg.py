import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm.notebook import tqdm_notebook as tqdm

from src.agents import Agent
from src.reply_buffer import MultiAgent_ReplayBuffer


class MA_DDPG:
    def __init__(self, env, lr_actor, lr_critic, hidden_size1, hidden_size2, environment_name, 
                 buffer_size, batch_size, discount_factor=0.99, tau=0.01, checkpoint_directory='models/'):
        
        self.env = env
        
        self.actions_space = {}
        self.observation_spaces = {}
        self.environment_name = environment_name
        
        self.checkpoint_directory = checkpoint_directory + environment_name
        
        for agent in self.env.possible_agents:
            self.actions_space[agent] = self.env.action_spaces[agent].shape[0]
            self.observation_spaces[agent] = self.env.observation_spaces[agent].shape[0]
        
        self.critic_input_shape= sum(self.observation_spaces.values()) + sum(self.actions_space.values())# For Critic Input
        
        
        self.agents = {}
        
        self.reward_across_episodes = []
        self.avg_reward_across_last_100_episodes = []
        self.avg_reward_per_episode = []
        self.best_avg_reward_per_100_episodes = []
        self.steps_per_episode = [] 
        
        self.each_agent_reward_per_episode = {}
        self.each_agent_avg_reward_per_100_episodes = {}
        
        for agent in self.env.possible_agents:
            self.agents[agent] = Agent(self.observation_spaces[agent], self.critic_input_shape, self.actions_space[agent], agent, 
                                       lr_actor, lr_critic, hidden_size1, hidden_size2, discount_factor, tau, self.checkpoint_directory)
            
            self.each_agent_reward_per_episode[agent] = []
            self.each_agent_avg_reward_per_100_episodes[agent] = []
        
        self.replay_buffer = MultiAgent_ReplayBuffer(buffer_size, self.env.possible_agents, batch_size)
        
        self.writer = SummaryWriter(log_dir=f'logs/{environment_name}')
        
    
    def save_checkpoint(self):
        for agent in self.env.possible_agents:
            self.agents[agent].save_models()
    
    def load_checkpoint(self):
        for agent in self.env.possible_agents:
            self.agent[agent].load_models()
            
    def choose_action(self, observation, evaluate):
        actions = {}
        
        for agent in self.env.possible_agents:
            actions[agent] = self.agents[agent].choose_action(observation[agent], self.env.action_spaces[agent].low_repr, self.env.action_spaces[agent].high_repr, evaluate=evaluate)
        
        return actions
    
    def evaluation(self, episodes):
        reward_across_episode = []
        for _ in range(episodes):
            observations, _ = self.env.reset()
            done = [False]*len(self.env.agents)
            reward_per_episode = 0
            while not any(done):
                actions = self.choose_action(observations, evaluate=True)
                new_observations, rewards, termination_status, truncations_status, _ = self.env.step(actions)
                done = []
                
                for agent in self.env.possible_agents:
                    done.append(termination_status[agent] or truncations_status[agent])
                
                reward_per_episode += sum(rewards.values())
                observations = new_observations
            reward_across_episode.append(reward_per_episode)
        return reward_across_episode
                
    def learn(self):
        if not self.replay_buffer.ready_to_be_sampled():
            return
        
        actor_sample, critic_sample = self.replay_buffer.sample_buffer()
        
        
        # For Agent
        all_agents_actions = {}
        all_agents_new_actions = {}
        
        old_actions_critic = []
        
        actions_critic = []
        new_actions_critic = []
        
        
        for agent in self.env.possible_agents:
            
            with torch.no_grad():
                # Current Actions taken as per current policy
                observation_batch = actor_sample[agent]['observation'].to(self.agents[agent].actor.DEVICE)
                action_batch = self.agents[agent].actor.forward(observation_batch)
                all_agents_actions[agent] = action_batch
                
                actions_critic.append(action_batch)
                
                
                # New Actions Taken as per current target policy
                new_observation_batch = actor_sample[agent]['new_observation'].to(self.agents[agent].actor.DEVICE)
                new_action_batch = self.agents[agent].target_actor.forward(new_observation_batch)
                all_agents_new_actions[agent] = new_action_batch
                
                new_actions_critic.append(new_action_batch)
                
                
                # Action Sampled as per old policy
                old_actions_critic.append(actor_sample[agent]['action']) 
        
        actions_critic = torch.cat(old_actions_critic, dim=1)
        new_actions_critic = torch.cat(old_actions_critic, dim=1)
        
        old_actions_critic = torch.cat(old_actions_critic, dim=1)
        
        # For Critic
        
        for agent in self.env.possible_agents:
            with torch.no_grad():
                new_critic_value = self.agents[agent].target_critic.forward(torch.cat([critic_sample[agent]['env_new_observation'], new_actions_critic], dim=1).to(self.agents[agent].critic.DEVICE)).flatten()
                target = critic_sample[agent]['reward'].to(self.agents[agent].critic.DEVICE) + self.agents[agent].discount_factor* new_critic_value * (1 - critic_sample[agent]['terminal_status_buffer'].to(self.agents[agent].critic.DEVICE))
            
            critic_value = self.agents[agent].critic.forward(torch.cat([critic_sample[agent]['env_observation'], old_actions_critic], dim=1).to(self.agents[agent].critic.DEVICE)).flatten()
            critic_loss = torch.nn.functional.mse_loss(target, critic_value)
            self.agents[agent].optimizers['critic'].zero_grad()
            critic_loss.backward()
            
            torch.nn.utils.clip_grad_norm_(self.agents[agent].critic.parameters(), 10.0)
            self.agents[agent].optimizers['critic'].step()
            
            actor_loss = self.agents[agent].critic.forward(torch.cat([critic_sample[agent]['env_observation'], actions_critic], dim=1).to(self.agents[agent].critic.DEVICE)).flatten() #Actions for the current states according to the
            # regular actor network as opposing to the actual actions we took(in the replay buffer)
            actor_loss = -torch.mean(actor_loss)
            self.agents[agent].optimizers['actor'].zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.agents[agent].actor.parameters(), 10.0)
            self.agents[agent].optimizers['actor'].step()
            
            self.agents[agent].synchronize()
    
    
    def train(self, episodes):
        
        self.best_score = -float('inf')
        total_steps = 0
        prog_bar = tqdm(range(episodes), desc='Training Episode:', disable=False)
        for episode in prog_bar:
            observations, _ = self.env.reset()
            done = [False]*len(self.env.agents)
            
            reward_per_episode = 0
            each_agent_reward_per_episode = {}
            
            for agent in self.env.possible_agents:
                each_agent_reward_per_episode[agent] = 0
                
            episode_step = 0
            
            while not any(done):
                actions = self.choose_action(observations, evaluate=False)
                new_observations, rewards, termination_status, truncations_status, _ = self.env.step(actions)
                
                done = []
                
                for agent in self.env.possible_agents:
                    done.append(termination_status[agent] or truncations_status[agent])
                
                self.replay_buffer.store_transistion(observations, actions, new_observations, rewards,
                                                     termination_status, truncations_status)
                
                
                if total_steps % 100 == 0:
                    self.learn() 
                
                reward_per_episode += sum(rewards.values())
                
                for agent in self.env.possible_agents:
                    each_agent_reward_per_episode[agent] += rewards[agent]
                
                observations = new_observations
                
                
                total_steps += 1
                episode_step += 1
            
            
            avg_score = np.mean(self.reward_across_episodes[-100:])
            
            self.reward_across_episodes.append(reward_per_episode)
            
            for agent in self.env.possible_agents:
                self.each_agent_reward_per_episode[agent].append(each_agent_reward_per_episode[agent])
                self.each_agent_avg_reward_per_100_episodes[agent].append(np.mean(self.each_agent_reward_per_episode[agent][-100:]))
            
            self.avg_reward_across_last_100_episodes.append(avg_score)
            self.avg_reward_per_episode.append(reward_per_episode/episode_step)
            self.best_avg_reward_per_100_episodes.append(self.best_score)
            self.steps_per_episode.append(episode_step)
            
            if avg_score > self.best_score:
                self.save_checkpoint()
                self.best_score = avg_score
            
            self.writer.add_scalar("reward/reward_per_episode", reward_per_episode, episode)
            self.writer.add_scalar("reward/avg_reward_per_episode", reward_per_episode/episode_step, episode)
            self.writer.add_scalar("reward/avg_reward_last_100_episodes", avg_score, episode)
            self.writer.add_scalar("reward/best_avg_reward_for_100_episodes", self.best_score, episode)
            self.writer.add_scalar("steps/episode_step", episode_step, episode)
            
            if episode % 500 == 0 and episode > 0:
                prog_bar.set_postfix_str(
                    f"Episode: {episode} - Average_Score:{avg_score}, Best_Score:{self.best_score}"
                )
        self.writer.flush()
        self.writer.close()