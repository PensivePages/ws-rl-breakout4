import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
import numpy as np
from collections import defaultdict #for Q ie: state store
from collections import namestuple #for transistions store ie: memory, buffer, ReplayMemory

def conv(in_channels, out_channels, kernel_size, stride=2, padding=1, batch_norm=True, init_zero_weights=False):
    """Creates a convolutional layer, with optional batch normalization.
    """
    layers = []
    conv_layer = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
    if init_zero_weights:
        conv_layer.weight.data = torch.randn(out_channels, in_channels, kernel_size, kernel_size) * 0.001
    layers.append(conv_layer)

    if batch_norm:
        layers.append(nn.BatchNorm2d(out_channels))
    return nn.Sequential(*layers)

class Meta_Controller():
    # Meta-controller chooses a goal
    def __init__(self):
        super(Meta_Controller, self).__init__()
        # conv(channels_in, channels_out, kernel, stride, padding)
        self.conv1 = conv(3, 32, 8, 4, 0) #all 3 convs set to paper figure 5b Q1 example
        self.conv2 = conv(32, 64, 4, 2, 0)
        self.conv3 = conv(64, 64, 3, 1, 0)
        self.MC_conv_out = self.conv3.size(0) * self.conv3.size(1) * self.conv3.size(2) * self.conv3.size(3)
        self.projection = nn.Linear(MC_conv_out, MC_hidden_size) #set to h=512 via paper figure 5b
        self.output = nn.Linear(MC_hidden_size, MC_number_of_goals)
        self.nonLinearity = torch.nn.ReLU()
        self.softmax = nn.Softmax() #not sure if used

    def forward(self):
        x = self.nonLinearity(self.conv1(x))
        x = self.nonLinearity(self.conv2(x))
        x = self.nonLinearity(self.conv3(x))
        x = x.view(1, -1) # flatten
        x = self.nonLinearity(self.projection(x))
        x = self.output(x)
        #x = self.softmax(self.output(x)) #not sure if used
        return x

class Controller(nn.Module):
    def __init__(self):
        super(Controller, self).__init__()
        # conv(channels_in, channels_out, kernel, stride, padding)
        self.conv1 = conv(3, 32, 8, 4, 0) #all 3 convs set to paper figure 5b
        self.conv2 = conv(32, 64, 4, 2, 0)
        self.conv3 = conv(64, 64, 3, 1, 0)
        self.C_conv_out = self.conv3.size(0) * self.conv3.size(1) * self.conv3.size(2) * self.conv3.size(3)
        self.projection = nn.Linear(C_conv_out, C_hidden_size) #set to h=512 via paper figure 5b
        self.output = nn.Linear(C_hidden_size, C_number_of_actions)
        self.nonLinearity = torch.nn.ReLU()
        self.softmax = nn.Softmax() #not sure if used

    def forward(self):
        x = self.nonLinearity(self.conv1(x))
        x = self.nonLinearity(self.conv2(x))
        x = self.nonLinearity(self.conv3(x))
        x = x.view(1, -1) # flatten
        x = self.nonLinearity(self.projection(x))
        x = self.output(x)
        x = self.softmax(self.output(x)) #not sure if used
        return x

class Critic_NN(nn.Module):
    def __init__(self, batch_size):
        super(Critic_NN, self).__init__()
        self.conv1 = conv(nc, ndf, 4, 2, 1)
        self.conv2 = conv(ndf, ndf * 2, 4, 2, 1)
        self.conv3 = conv(ndf * 2, ndf * 4, 4, 2, 1)
        self.projection = nn.Linear(conv_out, hidden_size)
        self.output = nn.Linear(hidden_size, output_size)
        self.nonLinearity = torch.nn.ReLU()
        self.softmax = nn.Softmax() #not sure if used

    def forward(self):
        x = self.nonLinearity(self.conv1(x))
        x = self.nonLinearity(self.conv2(x))
        x = self.nonLinearity(self.conv3(x))
        x = x.view(1, -1) # flatten
        x = self.nonLinearity(self.projection(x))
        x = self.output(x)
        #return self.softmax(self.output(x)) #not sure if used
        return x

class Critic():
    def step(self, action):
        next_state, reward, done = env(action)
        return next_state, reward, done

    def reward(self):
        next_state, reward, done = self.step(action) #need to check extrinsic reward also
        #if extrinsic reward: do stuff
        # get intrinsic reward
        if goal in next_state: return reward, 1, next_state, done #intrinsic_reward = 1
        else: return reward, 0, next_state, done #intrinsic_reward = 0

#memory code from AISC RL Workshop
class ReplayMemory(object):
    '''
        A simple memory for storing episodes where each episodes 
        is a names tuple with (state, action, next_state, reward, done)
    '''

    def __init__(self, capacity):
        '''
          Initialize memory of size capacity
          Input: Capacity : int 
                      size of the memory

          output: initialized ReplayMemory object
        '''
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, meta, *args): # meta=True or False to pick storage (MC/C)
        '''
          input *args : list  *args is list for transition 
          [state, action, next_state, reward, done] and add
          transition to memory.
          Returns : None
        '''
        if len(self.memory) < self.capacity:
            self.memory.append(None) #give it space in memory to set
        #self.memory[self.position] = Transition(*args) #original
        if meta:
            self.memory[self.position] = D2(*args)
        else:
            self.memory[self.position] = D1(*args)
        self.position = (self.position + 1) % self.capacity 
        #^ mod makes it loop around once it passed 10k

    def sample(self, batch_size, meta): # meta=True or False to pick storage (MC/C)
        '''
          Randomly sample transitions from memory
          Input batch_size : int
                  numer of transition to sample
          Output:  namedtuple
                    Namedtupe with each field contains a list of data points

        '''
        batch = random.sample(self.memory, batch_size)
        #return Transition(*zip(*batch)) #original
        if meta:
            return D2(*zip(*batch))
        else:
            return D1(*zip(*batch))


    def __len__(self):
        '''
            returns current size of memory
        '''
        return len(self.memory)

MC_pred = Meta-Controller()
MC_targ = Meta-Controller()
C_pred = Controller()
C_targ = Controller()
#critic_nn = Critic_NN() #not implemented
critic = Critic()

class Agent():

    def __init__(self, env, MC_buffer, C_buffer,\
                 learning_rate, gamma,\
                 exploration_param2, exploration_param1,
                 batch, epochs):
        self.env = env
        self.learning_rate = learning_rate
        self.MC_buffer = MC_buffer
        self.C_buffer = C_buffer
        self.gamma2 = gamma2 # "discount rate
        self.ep2 = exploration_param2 #MC
        self.ep1 = exploration_param1 #C
        self.batch = batch #input
        self.done = False
        self.terminated = False
        self.epochs = epochs
        self.epoch = 0
        self.Q1 = None
        self.Q2 = None

    def _init_model(self, env):
        self.Q1 = defaultdict(lambda: np.zeros(env.action_space.n))
        self.Q2 = defaultdict(lambda: np.zeros(env.action_space.n))
        #^ this way, if the state doesn't exist create it with n zeros = number of actions

    def train(self):
        if self.Q == None:
            _init_model(self, env)
        bQ1 = 0 #count for batch to break for training
        bQ2 = 0 #count for batch to break for training
        while epoch < epochs or not done or not terminated:
            Q2_prediction = MC_pred(self.batch) # set Q values for goal at given state
            #Q2_expectation = MC_targ(self.next_batch) # set Q values for goal at given state
            i_goal = np.argmax(Q2_prediction[state]) # e-greedy only atm
            goal = goals[i_goal] # select goal image from list of goals by index
            #NOTE: "exploration probability e2" with starting value 1 works with the greedy pick
            # ^so, if random(0 to 1) < e1: choose random action
            # ^likewise, if random(0 to 1) < e2: choose random goal
            self.batch = torch.cat((self.batch, goal), dim=1)
            intrinsic_reward = 0
            while intrinsic_reward == 0:
                Q1_prediction = C_pred(self.batch) # set Q values for action at given state and goal
                #Q1_expectation = C_targ(self.next_batch) # set Q values for action at given state and goal
                i_action = np.argmax(Q1_prediction[state]) # e-greedy only atm
                action = actions[i_action] # select action from list of actions by index
                extrinsic_reward, intrinsic_reward, next_state, done = critic(action)
                # store controller transition
                C_buffer.push(False, state, action, goal, intrinsic_reward, next_state)
                MC_buffer.push(True, state, goal, extrinsic_reward, next_state)
                #Q1_expectation = intrinsic_reward + (gamma^t-t_prime) * max(Q1_expectation)
                #loss = mse(Q1_prediction, Q1_expectation.detatch())
                bQ1 += 1
            # store meta-controller transision
            #MC_buffer.push(True, state, goal, extrinsic_reward, next_state)
            #Q2_expectation = extrinsic_reward + (gamma^t-t_prime) * max(Q2_expectation)
            #loss = mse(Q2_prediction, Q2_expectation.detatch())
            epoch += 1
            bQ2 += 1

    def update_Q1(self):
        Q1_prediction = C_pred(self.batch)
        Q1_expectation = C_targ(self.next_batch)
        Q1_expectation = intrinsic_reward + (gamma^t-t_prime) * max(Q1_expectation)
        Q1_loss = mse(Q1_prediction, Q1_expectation.detatch())

    def update_Q2(self):
        Q2_prediction = MC_pred(self.batch)
        Q2_expectation = MC_targ(self.next_batch)
        Q2_expectation = extrinsic_reward + (gamma^t-t_prime) * max(Q2_expectation)
        Q2_loss = mse(Q2_prediction, Q2_expectation.detatch())
            
            

#---params---
learning_rate =  0.00025
gamma = .99
exploration_param2 = 1 #start at 1 go to 0.1

exploration_param1 = 1 #start at 1 go to 0.1
num_of_goals = object_detector.goals #pseudo
num_of_actions = env.actions #pseudo

#general dims
nc = 3
ngf = 64

# meta-controller dims
MC_conv_out = MC_batch_size * MC_C * MC_W * MC_H
MC_hidden_size = 512
MC_number_of_goals = num_of_goals

# controller dims
C_conv_out = C_batch_size * C_C * C_W * C_H
C_hidden_size = 512
C_num_of_actions = num_of_actions

# critic dims
input_size_critic = conv_out_size
hidden_size_critic = conv_out_size
output_size_critic = 1 #num_of_rewards?

#---memory---
# meta-controller
D2 = namestuple('D2',
                    ('state', 'goal', 'extrinsic_reward', 'next_state'))
# controller
D1 = namestuple('D1',
                    ('state', 'action', 'goal', 'intrinsic_reward', 'next_state'))

MC_buffer = ReplayMemory(1000000)
C_buffer = ReplayMemory(50000)
