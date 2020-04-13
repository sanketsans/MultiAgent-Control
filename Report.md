[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135623-e770e354-7d12-11e8-998d-29fc74429ca2.gif "Trained Agent"
[image2]: https://user-images.githubusercontent.com/10624937/42135622-e55fb586-7d12-11e8-8a54-3c31da15a90a.gif "Soccer"


#  Collaboration and Competition

### Introduction

For this project, I worked with the [Tennis](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis) environment.

![Trained Agent][image1]

## Environment details

In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1.  If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01.  Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation.  Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping. 

The task is episodic, and in order to solve the environment, your agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

- After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
- This yields a single **score** for each episode.

The environment is considered solved, when the average (over 100 episodes) of those **scores** is at least +0.5.

## Implementation details

The project was solved using an *off-policy method* called **Multi Agent Deep Deterministic Policy Gradient (MADDPG)** algorithm.

**MADDPG** is an extension of DDPG algorithm for multiple agents in a continous space. It is described in this paper - 
[Continuous control with deep reinforcement learning](https://arxiv.org/abs/1509.02971). 

## Details on DDPG
DDPG is also an off policy method. It is a kind of actor-critic method where actor network is responsible for generating actions
and critic evaluates the accuaracy of those action. It can also be described as a "kind of supervised learning" method, where 
crtic is considered as the labels using which network / agent is improved. 
The algorithm for DDPG is 

![DDPG algorithm from Spinning Up website](./images/DDPG.svg)

I have also tried to simplify the algorithm using this diagram :

<p align="center">
  <img width="460" height="400" src="https://github.com/sanketsans/MultiAgent-Control/blob/master/images/ddpg_explained.jpeg">
</p>

## Multi Agent Deep Deterministic Gradient Policy (MADDPG)

For this project I have used a variant of DDPG called **Multi Agent Deep Deterministic Policy Gradient (MADDPG)** which is  
described in the paper [Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments](https://arxiv.org/abs/1706.02275)

we accomplish our goal by adopting the framework of centralized training withdecentralized execution. Thus, we allow
the policies to use extra information to ease training, so long as this information is not used at test time. 
Thus, we propose a simple extension of actor-critic policy gradient methods where the critic is augmented with extra
information about the policies of other agents.

In short, this means that during the training, the Critics networks have access to the states and actions information
of both agents, while the Actors networks have only access to the information corresponding to their local agent
The agents are update asynchronously i.e, each agent computes its own gradient and updates it network asynchronously. 

## Code Implementation 

The projects consists of :

- model.py : It consists the neural network model of both actor and critic. Consider that in the critic network we change
the input parameters by adding number of actions to state size and multiplying by number of agents. 

- utilities.py : It consists of important utilities classes for training of the network - OUNoise, for adding the noise to
the action space to create better exploration. ReplayBuffer - to store the experiences to be used later to train the network. 

- Tennis_1.ipynb : The python notebook for the implementation. 
          - It first loads the unity evnrionment.
          - Then all the other dependencies and classes are imported. 
          - I create a single shared buffer which will be shared by all the agents in the network. It means all the agent 
          will store their experiences in this buffer memory.
          - DDPG agent : A single DDPG agent is created which has two neural network for actor and critic - one for training
          at each timestep and one for updating the weights(learning). The learn() function is used to compute the loss 
          function and update the network. 
          - MADDPG agent : For number of agents in the environment, DDPG agent is instantiated for each agent and all their
          weights are set to random. 
          - Training : For each action step, experience is stored in the shared buffer. And at each timestep, replay buffer 
          samples data for states, new_states, action for each agent(list) and total rewards and done(any). 
          From the sampled data, each agents computes its loss and updates the network. 
          
          
## Results : 

I first tried to load the list directly into the learn() method of DDPG agnet, which was giving me problems. Then later I changed
that list to tensors for each agent and computed the loss function for each agent separetly to update the network. 
Also, complicating the network was not really a good choice at first. I started with 1024 nodes of hidden layers for each actor
and critic network - which really slowed the training and also the loss was not really very significant. 

So, I dropped down to 200 and 100 nodes of hidden layers. 

Training hyperparameters :


```
SEED = 0                          # Random seed

NB_EPISODES = 5000                # Max nb of episodes
UPDATE_EVERY_NB_EPISODE = update every step       # Nb of episodes between learning process

BUFFER_SIZE = int(1e5)             # replay buffer size
BATCH_SIZE = 250                   # minibatch size

ACTOR_FC1_UNITS = 200              # Number of units for the layer 1 in the actor model
ACTOR_FC2_UNITS = 150              # Number of units for the layer 2 in the actor model
CRITIC_FCS1_UNITS = 200            # Number of units for the layer 1 in the critic model
CRITIC_FC2_UNITS = 150             # Number of units for the layer 2 in the critic model
NON_LIN = F.relu                   # Non linearity operator used in the model
LR_ACTOR = 1e-4                    # Learning rate of the actor 
LR_CRITIC = 5e-3   #2e-3           # Learning rate of the critic
WEIGHT_DECAY = 0                   # L2 weight decay

GAMMA = 0.995 #0.99                # Discount factor
TAU = 1e-3                         # For soft update of target parameters
CLIP_CRITIC_GRADIENT = False       # Clip gradient during Critic optimization

ADD_OU_NOISE = True                # Add Ornstein-Uhlenbeck noise
MU = 0.                            # Ornstein-Uhlenbeck noise parameter
THETA = 0.15                       # Ornstein-Uhlenbeck noise parameter
SIGMA = 0.2                        # Ornstein-Uhlenbeck noise parameter
```

Actor Neural Network Model : Input node (state_size)
                             Output node (action_size)
                             
Critic Neural Network Model : Input node ((state_size+action_size) * num_agents)
                             Output node (1 or q-value)   
                             
Given the architecture above : 

![Results]()

**These results meets the project's expectation as the agent is able to receive an average reward (over 100 episodes)
of at least +0.5 in 1892 episodes** 

## Ideas for future work 

to further improve our Multi-Agents project would be to implement [Prioritized experience replay](https://arxiv.org/abs/1511.05952)

> Experience replay lets online reinforcement learning agents remember and reuse experiences from the past. In 
prior work, experience transitions were uniformly sampled from a replay memory. However, this approach simply replays
transitions at the same frequency that they were originally experienced, regardless of their significance. In this paper 
we develop a framework for prioritizing experience, so as to replay important transitions more frequently, and therefore 
learn more efficiently. We use prioritized experience replay in Deep Q-Networks (DQN), a reinforcement learning algorithm 
that achieved human-level performance across many Atari games. DQN with prioritized experience replay achieves a new 
state-of-the-art, outperforming DQN with uniform replay on 41 out of 49 games.
