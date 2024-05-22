# Exploring Multi-Agent Reinforcement Learning in Cooperative Gridworld and Complex Multi-Agent Environments
This project delves into the realm of Multi-Agent Reinforcement Learning (MARL), focusing on cooperative gridworlds and adversarial environments. It includes several cutting-edge algorithms such as Decentralized Q-learning, Correlated Q-Learning, Multi-Agent Deep Double Q-Network (MAD-DQN), and REINFORCE. In the cooperative gridworld, themed around Batman's Gotham City, agents like Batman and Robin collaborate to achieve goals while navigating through obstacles and rewards. The adversarial environment explores interactions between a primary agent and its opponent, employing techniques like Multi-Agent Deep Deterministic Policy Gradient (MADDPG) with Centralized Learning and Decentralized Execution (CLDE) for effective coordination. This project showcases advancements in MARL algorithms and their applications in diverse scenarios.

## Grid World
A Batman-themed gridworld environment with 81 states representing different locations in Gotham City, Batman(Agent-1) and Robin(Agent-2) must navigate the streets to reach Catwoman(Goal) while avoiding villains like the Joker and Bane which are obstacles. They move up, down, left, or right, with the environment rewarding strategic movements and punishing delays or encounters with villains. The environment enforces boundary checks to keep the agents within the grid, imposes penalties for staying in the same position, and prevents Agent-1  and Agent-2 from occupying the same grid cell simultaneously. This cooperative setup ensures that the agents work together to rescue Catwoman and stay away from villains.

### Characters
- **Batman** – Agent 1
- **Robin** – Agent 2
- **Selena** – Goal State
- **Batmobile** – Reward
- **Scarecrow** – Obstacle
- **Alfred** – Obstacle
- **Redbird** – Reward
- **Arkham** – Obstacle
- **Joker** – Obstacle
- **Court of Owls** – Obstacle
- **Bane** – Obstacle

### Rewards and Obstacles
| Character     | Reward/Penalty                |
|---------------|-------------------------------|
| **Selena**    | 20                            |
| **Joker**     | -10                           |
| **Court of Owls** | -7                         |
| **Bane**      | -5                            |
| **Scarecrow** | -3                            |
| **Alfred**    | 10                            |
| **Arkham**    | -1 + Reset the agent to the original state |
| **Batmobile** | 10 (for Batman)               |
| **Redbird**   | 10 (for Robin)                |


This cooperative environment encourages teamwork between Batman and Robin as they navigate through Gotham City, aiming to rescue Catwoman and achieve victory.

### Algorithms

#### Decentralized Q-learning
This method is a pivotal technique in multi-agent reinforcement learning, enabling each agent to independently refine its policy based on interactions with the environment. This approach enhances scalability and adaptability for both cooperative and competitive scenarios without necessitating direct communication among agents. Each agent maintains and updates its own Q-table, effectively balancing exploration and exploitation through individual rewards. 

This method offers significant advantages for complex systems, such as robotics or distributed network management. However, it also presents several challenges. The non-stationarity issue arises from simultaneous learning by multiple agents, there is a lack of strong theoretical convergence guarantees, and difficulties in credit assignment can impede the achievement of optimal collective outcomes.

#### Correlated Q-Learning*
By employing joint action-value functions, our agents can harmonize actions and cultivate cooperation effectively. This methodology adapts traditional Q-Learning by incorporating computations for correlated equilibrium, where agents sample joint actions from the equilibrium and adjust their strategies to uphold correlations, leading to the attainment of collective objectives. Despite the promising prospects for enhancing cooperation, the selection of the most suitable equilibrium presents challenges due to the extensive range of possible correlated equilibria. Hence, our approach emphasizes the necessity for meticulous evaluation and robust analytical methods to facilitate optimal decision-making.


#### Multi-Agent Deep Double Q-Network (MAD-DQN) 

MAD-DQN is a cutting-edge advancement in multi-agent reinforcement learning, designed to empower multiple agents with decentralized policies while considering the actions of their counterparts. This method significantly enhances exploration efficiency and fosters collaboration across a spectrum of scenarios.

MAD-DQN leverages a decentralized architecture and integrates double Q-learning, enabling each agent to manage both target and local Q-networks independently for action evaluation. This approach mitigates overestimation challenges effectively.

Key techniques such as experience replay and target network updates play pivotal roles in enhancing learning efficiency and ensuring stability throughout the training process. By storing past interactions and periodically refining target networks, agents engage in informed learning, resulting in reliable Q-value estimations and improved policy development.

In essence, MAD-DQN provides a robust and adaptable framework essential for navigating complex multi-agent environments with precision and effectiveness.

#### REINFORCE
The Policy Gradient Theorem holds significant importance in Multi-Agent Reinforcement Learning (MARL), where agents autonomously refine their policies to maximize rewards. The REINFORCE algorithm, grounded in this theorem, leverages Monte Carlo return estimates derived from complete episodic trajectories to iteratively improve policies. However, it encounters challenges due to high variance, leading to training instability.

To address this issue, baselines such as the state-value function are incorporated to mitigate gradient variance while ensuring effective policy optimization. Actor-Critic algorithms, exemplified by Advantage Actor-Critic (A2C) and Proximal Policy Optimization (PPO), operate by concurrently training a parameterized policy (actor) alongside a value function (critic). This dual approach enhances training stability and convergence by balancing bias and variance, thus facilitating efficient policy learning in dynamic multi-agent environments.



## Multiparticle Environemnt - Simple Adversery

The Simple Adversary Environment is designed to facilitate the exploration of adversarial interactions in reinforcement learning, focusing on the dynamics between a primary agent and its opponent. In this environment, the primary agent is tasked with achieving a specific goal or reaching a designated state, while the opposing agent's mission is to thwart this success. Both agents employ reinforcement learning techniques, adjusting their actions based on the rewards and penalties they receive, thereby continuously evaluating and refining their strategies.

![Simple Adversary Environment](images/mpe_simple_adversary.gif)

### Algorithms

#### Multi-Agent Deep Deterministic Policy Gradient 


This project implements the Multi-Agent Deep Deterministic Policy Gradient (MADDPG) algorithm with a knowledge-sharing framework based on Centralized Learning and Decentralized Execution (CLDE). The algorithm is designed to facilitate effective coordination and decision-making among multiple agents in a shared environment.

##### Centralized Learning and Decentralized Execution

The MADDPG framework embodies a delicate balance between centralized learning and decentralized execution. During the learning phase, the critic networks play a pivotal role by providing a global perspective on the actions and values across all agents. This centralized learning facilitates the development of effective collaborative strategies.

During the execution phase, each agent relies solely on its own actor network to make decisions, ensuring autonomy and real-time adaptability. This decentralized execution allows agents to operate independently, leveraging the refined policies developed during the learning phase to navigate complex multi-agent environments effectively.

![Centralized Learning and Decentralized Execution](images/centralized_learning_and_decentralized_execution.png)


