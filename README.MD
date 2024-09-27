# Gymnasium: Classic Control
Solutions for classic control environments of Gymnasium package.

## Tabular method

Since all environments have continuous state space, it needs to be converted into discrete space. It is done by equally
dividing state space into interval for each feature. Tabular methods were used only for environments with discrete
action space, therefore, no discretisation was needed there.

### Cart pole
Q-learning where state is converted into discrete values. Behavioral policy is $\epsilon$-greedy with constant
epsilon. Can hold the pole up for full 500 steps.

### Mountain car
Q-learning with optimistic value function and $\epsilon$ greedy behaviour policy.
Observations are discretised to allow tabular Q-learning. 
The agent can be trained to reach goal everytime and it takes ~ 120 steps. 

### Acrobot
N-step Sarsa with multiple n to compare performance. The worst performing was N=1 with average of 280 steps.
Other agent were roughly on 200 steps with slightly better results with increasing N. None of them is by far 
the optimal policy - one would need better exploration of states near the goal as they are not visited enough.
Also, observation space has 6 features which means the discrete representation has either
very rough granularity or is very large and needs massive amount of time to explore fully.

## Approximated
### Mountain car - polynomial
The simplest approximated approach would be polynomial function but I was unable to find 
reliable hyperparameters. Therefore, more complex solutions is needed.

### Mountain car - Radial Basis Function
The RBF is sort of a intermediate step to neural networks.
Agent need ~100 episodes to learn how to get to the top. It reaches the top in ~140 steps.

### Neural networks

#### DQN
Deep Q Network is method based on Q learning where Q table is replaced by neural network. 
It means agent doesn't need to explore as much as in tabular method because tabular 
agent needs to visit every state multiple time while DQN agent approximates from similar
states. There are many variations, I used the one with two identical NN policy and target called
double DQN which reduces oscillation and Dueling DDQN which further impores performance.

Special note to MountainCar environment: it is highly unlikely to solve (although possible if lucky) with simple 
$\epsilon$-greedy exploration. We use similar trick as in tabular solution of this environment. In there, all
action-state pairs have initial value of one which, in combination with -1 reward for each step, makes unvisited states
more appealing than those visited. The same trick is applied here - neural network is trained to output positive 
constant number for any input while greedily following policy. Keep in mind that, unlike in tabular method, this
doesn't guarantee the agent to find solution as the policy is updated as a whole and negative reward in one state can
affect value of other state, even remote ones.

#### Actor-Critic
TBD

## Continuous action space

In this case, use of tabular methods is still possible by applying discretisation to action space as well, but it
would yield poor results in general. Thus, only approximation via neural networks is considered here.

### DDPG
DDPG is gradient descent algorithm like Actor-Critic method. Gradient descent is a family of algorithms that
approximate policy directly, not via utility function like Q methods. This fact makes it applicable in environments 
with both discrete and continuous action spaces.
