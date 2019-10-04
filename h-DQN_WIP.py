import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import cv2
import gym
#colab install for universe
#!git clone https://github.com/openai/universe.git
#!cd universe
#!pip install -e .
#!pip install 'gym[atari]'
#!pip install universe
import universe # register the universe environments
import atari_py as ap #for list
from collections import defaultdict #for Q ie: state Q values store
from collections import namedtuple #for transistions store ie: memory, buffer, ReplayMemory

# list of the games
game_list = ap.list_games()
print(sorted(game_list))

# custom weights initialization
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

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

class Meta_Controller(nn.Module):
    # Meta-controller predicts the value of a goal
    def __init__(self, MC_conv_out, MC_hidden_size, MC_number_of_goals, ngpu):
        super(Meta_Controller, self).__init__()
        self.ngpu = ngpu
        # conv(channels_in, channels_out, kernel, stride, padding)
        self.conv1 = conv(1, 32, 8, 4, 0) #all 3 convs set to paper figure 5b Q1 example
        self.conv2 = conv(32, 64, 4, 2, 0)
        self.conv3 = conv(64, 64, 3, 1, 0)
        self.projection = nn.Linear(MC_conv_out, MC_hidden_size) #set to h=512 via paper figure 5b
        self.output = nn.Linear(MC_hidden_size, MC_number_of_goals)
        self.nonLinearity = torch.nn.ReLU()

    def forward(self, x):
        print("inside Meta_Controller.forward")
        x = self.nonLinearity(self.conv1(x))
        x = self.nonLinearity(self.conv2(x))
        x = self.nonLinearity(self.conv3(x))
        print("Meta_Controller final conv out: ", x.size())
        x = x.view(4, -1) # flatten
        x = self.nonLinearity(self.projection(x))
        x = self.output(x)
        return x

class Controller(nn.Module):
    # Controller predicts the value of a action
    def __init__(self, C_conv_out, C_hidden_size, C_number_of_actions, ngpu):
        super(Controller, self).__init__()
        self.ngpu = ngpu
        # conv(channels_in, channels_out, kernel, stride, padding)
        self.conv1 = conv(1, 32, 8, 4, 0) #all 3 convs set to paper figure 5b
        self.conv2 = conv(32, 64, 4, 2, 0)
        self.conv3 = conv(64, 64, 3, 1, 0)
        self.projection = nn.Linear(C_conv_out, C_hidden_size) #set to h=512 via paper figure 5b
        self.output = nn.Linear(C_hidden_size, C_number_of_actions)
        self.nonLinearity = torch.nn.ReLU()

    def forward(self, x):
        print("inside Controller.forward")
        x = self.nonLinearity(self.conv1(x))
        x = self.nonLinearity(self.conv2(x))
        x = self.nonLinearity(self.conv3(x))
        print("Controller final conv out: ", x.size())
        x = x.view(4, -1) # flatten
        x = self.nonLinearity(self.projection(x))
        x = self.output(x)
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

    def forward(self, x):
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

class Agent():

    def __init__(self, env, MC_buffer, C_buffer,\
                 learning_rate, gamma,\
                 exploration_param2, exploration_param1,\
                 max_goals_to_try, batch_size, extrinsic_tries_before_eval):
        self.new_session = True
        self.env = env
        self.learning_rate = learning_rate
        self.MC_buffer = MC_buffer
        self.C_buffer = C_buffer
        self.gamma = gamma # "discount rate"
        self.ep2 = exploration_param2 #MC, meta-controller
        self.ep1 = exploration_param1 #C, controller
        #self.state = None #input
        self.done = False
        self.terminated = False
        self.max_goals_to_try = max_goals_to_try
        self.num_goals_tried = 0
        self.Q1 = None
        self.Q2 = None
        self.batch_size = batch_size
        self.intrinsic_reward = 0
        self.intrinsic_success_count = 0
        self.intrinsic_total = 0
        self.extrinsic_success_count = 0
        self.total_sessions = 0
        self.eval_tries = extrinsic_tries_before_eval
        self.done = False
        self.goals = []
        self.actions = list(range(self.env.action_space.n))

    def change_pixels(self, state, center_pixels: list, padding: int): #state tensor size (N, C, W, H)
        staring_row = center_pixels[0] - padding
        starting_column = center_pixels[1] - padding
        side = padding * 2 + 1
        i, j = staring_row, starting_column
        while i < side:
            while j < side:
                state[0][0][i][j] = 127
                j += 1
            i += 1
        return state
    
    def create_goals(self, state):
        #goal_template = torch.ones_like(state)
        # goal 1 (key): change pixels: center: row: 30, column: 9, padding: 1?
        #key = change_pixels(self, goal_template, [30, 9], 2)
        key = lambda state: self.change_pixels(state, [30, 9], 2)
        # goal 2 (middle-ladder): change pixels: center: row: 35, column: 41, padding: _?
        #middle_ladder = change_pixels(self, goal_template, [35, 41], 2)
        middle_ladder = lambda state: self.change_pixels(state, [35, 41], 2)
        # goal 3 (left-ladder): change pixels: center: row: 59, column: 17, padding: _?
        #left_ladder = change_pixels(self, goal_template, [59, 17], 2)
        left_ladder = lambda state: self.change_pixels(state, [59, 17], 2)
        # goal 4 (right-ladder): change pixels: center: row: 59, column: 71, padding: _?
        #right_ladder = change_pixels(self, goal_template, [59, 71], 2)
        right_ladder = lambda state: self.change_pixels(state, [59, 71], 2)
        # goal 5 (left-door): change pixels: center: row: 10, column: 10, padding: _?
        #left_door = change_pixels(self, goal_template, [10, 10], 2)
        left_door = lambda state: self.change_pixels(state, [10, 10], 2)
        # goal 6 (right-door): change pixels: center: row: 10, column: 76, padding: _?
        #right_door = change_pixels(self, goal_template, [10, 76], 2)
        right_door = lambda state: self.change_pixels(state, [10, 76], 2)
        self.goals = [key, middle_ladder, left_ladder, right_ladder, left_door, right_door]

    def init_Q(self, env):
        print("inside init_Q")
        self.Q1 = defaultdict(lambda: np.zeros(env.action_space.n))
        #^ this way, if the state doesn't exist create it with n zeros = number of actions
        self.Q2 = defaultdict(lambda: np.zeros(len(goals))) #pseudo

    def init_session(self):
        print("inside init_session")
        # reset state to init
        state = env.reset() #pseudo, need to know the command
        # reset time_step to init
        time_step = 0 #not in use atm
        # create goals
        if not self.goals:
            self.create_goals(state)
        return state, time_step

    # preprocess() from: http://www.pinchofintelligence.com/openai-gym-part-3-playing-space-invaders-deep-reinforcement-learning/
    def preprocess(self, observation):
        observation = cv2.cvtColor(cv2.resize(observation, (84, 110)), cv2.COLOR_BGR2GRAY)
        observation = observation[26:110,:]
        ret, observation = cv2.threshold(observation,1,255,cv2.THRESH_BINARY)
        return np.reshape(observation,(1,84,84)) #np.reshape(observation,(84,84,1)) #original

    def format_state(self, state, next_state=None, test=False): #so format_state() shows what the preprocessing changes once
        state = self.preprocess(state)
        state = torch.from_numpy(state).unsqueeze(dim=0).type('torch.FloatTensor')
        # init starting image state with 3 previous void images states
        if self.new_session:
            ph_state = torch.zeros_like(state)
            state = torch.cat((state, ph_state), dim=0)
            state = torch.cat((state, ph_state), dim=0)
            state = torch.cat((state, ph_state), dim=0)
            #some test code from preprocess() source
            action0 = 0  # do nothing
            observation0, reward0, terminal, info = env.step(action0)
            print("reward0: ", reward0)
            print("terminal: ", terminal)
            print("info: ", info)
            print("Before processing: " + str(np.array(observation0).shape))
            plt.imshow(np.array(observation0))
            plt.show()
            observation0 = self.preprocess(observation0)
            print("After processing: " + str(np.array(observation0).shape))
            plt.imshow(np.array(np.squeeze(observation0)))
            plt.show()
            #brain.setInitState(observation0) #test model not used
            #brain.currentState = np.squeeze(brain.currentState) #test model not used
        else:
            state = self.update_state(self, state, next_state)
        if test:
            print("state size (shape): ", state.size())
            #print("state: ", state)
            batch = state
            bottom_label = list(range(0, 83))
            for image in batch:
                for channel in image: #only 1 channel since gray scale
                    for i, grid in enumerate(channel):
                        print("Grid %s: "%(i), grid)
                        print("label  :        ", bottom_label)
                    break #break after 1
        
        return state.to(device)
    
    def update_state(self, state, next_state):
        state[1:] = state[:-1]
        state[0] = next_state
        return state

    def place_goal_in_state(state, goal):
        state[0] *= goal
        return state

    def critic(self, action, goal):
        print("inside critic")
        # execute action in env
        next_state, extrinsic_reward, done, info = self.env.step(action) #pseudo
        # get intrinsic reward
        if goal == next_state: return extrinsic_reward, 1, next_state, done #intrinsic_reward = 1
        else: return extrinsic_reward, 0, next_state, done #intrinsic_reward = 0

    def select_direction(self, choices, exploration_param): #select goal, or action, greedy vs explore
        print("inside select_direction")
        random = np.random.uniform(size=1)[0]
        if random <= exploration_param:
            return np.random.choice(np.arange(len(choices)))
        else: #greedy
            return np.argmax(choices)

    def train(self):
        print("inside train")
        if self.Q1 == None:
            self.init_Q(env)
        bQn = 0 #count for batch to break for training
        while self.num_goals_tried < self.max_goals_to_try:
            if self.new_session == True:
                # reset session
                state, time_step = self.init_session()
                state = self.format_state(state)
                self.new_session = False
                first = False
            Q2_prediction = MC_pred(state) # set Q values for goal at given state
            i_goal = self.select_direction(Q2_prediction, self.ep2) # select goal intex
            goal = self.goals[i_goal] # select goal function
            state_goal = goal(state) # put goal on current state image
            while self.intrinsic_reward == 0 and not self.done:
                Q1_prediction = C_pred(state_goal) # set Q values for action at given state and goal
                i_action = self.select_direction(Q1_prediction, self.ep1) # select action intex
                action = self.actions[i_action] # select action from list of actions by index
                extrinsic_reward, self.intrinsic_reward, next_state, self.done = self.critic(action, goal) #get rewards and state
                next_state = self.format_state(state, next_state)
                next_state = goal(state) # put goal on current state image
                # store transitions
                C_buffer.push(False, state_goal, action, self.intrinsic_reward, next_state) #store controller transition
                MC_buffer.push(True, state, goal, extrinsic_reward, next_state) #store meta-controller transition
                # update state
                state = next_state
                # manage batch NN update cycle
                bQn += 1
                if bQn%batch_size == 0: #update every batch_size
                    states, next_states, rewards = self.get_random_batch(C_buffer) #pseudo
                    self.update_Q1(states, next_states, rewards)
                    states, next_states, rewards = self.get_random_batch(MC_buffer) #pseudo
                    self.update_Q2(states, next_states, rewards)
                # track intrinsic reward success and total
                if self.intrinsic_reward > 0:
                    self.intrinsic_success_count += 1
                self.intrinsic_total += 1
                # track extrinsic reward success
                if extrinsic_reward > 0:
                    self.extrinsic_success_count += 1
            # track number of goals tried so far
            self.num_goals_tried += 1
            # when done reset session
            if self.done:
                # track total sessions
                self.total_sessions += 1
                # new session
                self.new_session = True
            #--update the exploration params--
            if self.num_goals_tried%self.eval_tries == 0:
                # how often intrinsic goal reached
                self.ep1 = 1 - (self.intrinsic_success_count/self.intrinsic_total)
                if self.ep1 < .1:
                    self.ep1 = .1 #cap minimum exploration at .1
                # how often intrinsic goals lead to extrinsic goal
                #pass
                # how often extrinsic goal was reached
                self.ep2 = 1 - (self.extrinsic_success_count/self.total_sessions)
                if self.ep2 < .1:
                    self.ep2 = .1 #cap minimum exploration at .1

    def update_Q1(self, states, next_states, rewards):
        print("inside update_Q1")
        C_pred.zero_grad()
        prediction = C_pred(states)
        expectation = C_targ(next_states) #should I consider if the next_state changed the goal? how?
        #^ changing goal may be a argument to use: pred = random_batch[:-1] and tar = random_batch[1:]
        #   ^then use just the 'states' and that goal
        #       ^because the shifted state would = next_state and next_goal
        expectation = rewards + (gamma * np.amax(expectation)) #pseudo #notice, as long as (gamma^t_prime-t) is relative to the previous state it's gamma^1 = gamma
        loss = F.nn.MSELoss(prediction, expectation.detatch())
        loss.backward()
        optimizerC.step()

    def update_Q2(self, states, next_states, rewards):
        print("inside update_Q2")
        MC_pred.zero_grad()
        prediction = MC_pred(states)
        expectation = MC_targ(next_states)
        expectation = rewards + (gamma * np.amax(expectation)) #pseudo #notice, as long as (gamma^t_prime-t) is relative to the previous state it's gamma^1 = gamma
        loss = F.nn.MSELoss(prediction, expectation.detatch())
        loss.backward()
        optimizerMC.step()

    def get_random_batch(self, transitions):
        print("inside get_random_batch")
        random_batch = np.array(transitions)[np.random.choice(np.arange(len(transitions)), self.batch_size)] #pseudo, but close?
        states = random_batch[:, 'state'] #pseudo
        next_states = random_batch[:, 'next_state'] #pseudo
        rewards = random_batch[:, 'reward'] #pseudo
        return states, next_states, rewards, goals

    def update_df(self):
        pass #may implement if I decide to do gamma^t'- t, where t' = current and t = (future or past) 
            
#--environment--
env = gym.make('MontezumaRevenge-v4')
env.reset()
#env.render()

#---memory---
# meta-controller
D2 = namedtuple('D2',
                    ('state', 'goal', 'reward', 'next_state')) #reward = extrinsic reward
# controller
D1 = namedtuple('D1',
                    ('state', 'action', 'goal', 'reward', 'next_state')) #reward = intrinsic reward

#---params---

# meta-controller dims
MC_conv_out = 1 * 64 * 7 * 7 #guess for now
MC_hidden_size = 512
MC_number_of_goals = 6 #num_of_goals

# controller dims
C_conv_out = 1 * 64 * 7 * 7 #guess for now
C_hidden_size = 512
C_number_of_actions = env.action_space.n #num_of_actions

# # critic dims #not implemented
# input_size_critic = conv_out_size
# hidden_size_critic = conv_out_size
# output_size_critic = 1 #num_of_rewards?

# Agent vars
env = env
MC_buffer = ReplayMemory(1000000)
C_buffer = ReplayMemory(50000)
learning_rate =  0.00025
gamma = .99
exploration_param2 = 1 #start at 1 go to 0.1
exploration_param1 = 1 #start at 1 go to 0.1
num_of_goals = 6 #randomly picked 6 ftm #object_detector.goals.n #pseudo
num_of_actions = env.action_space.n
max_goals_to_try = 10000
batch_size = 512
extrinsic_tries_before_eval = batch_size #setting for batch_size ftm

# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1

# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

# Assign models
MC_pred = Meta_Controller(MC_conv_out, MC_hidden_size, MC_number_of_goals, ngpu).to(device)
MC_pred.apply(weights_init)
MC_targ = Meta_Controller(MC_conv_out, MC_hidden_size, MC_number_of_goals, ngpu).to(device)
MC_targ.apply(weights_init)
C_pred = Controller(C_conv_out, C_hidden_size, C_number_of_actions, ngpu).to(device)
C_pred.apply(weights_init)
C_targ = Controller(C_conv_out, C_hidden_size, C_number_of_actions, ngpu).to(device)
C_targ.apply(weights_init)

# Handle multi-gpu if desired
model = None #not Implemented
if (device.type == 'cuda') and (ngpu > 1):
    model = nn.DataParallel(model, list(range(ngpu)))

# Setup Adam optimizers for MC_pred and C_pred, excluded _targ in attempt to hold those constant
optimizerMC = optim.Adam(MC_pred.parameters(), lr=learning_rate, betas=(0.5, 0.999))
optimizerC = optim.Adam(C_pred.parameters(), lr=learning_rate, betas=(0.5, 0.999))

#--Agent call--
agent = Agent(env, #environment
      MC_buffer, #MC_buffer
      C_buffer, #C_buffer
      learning_rate, #learning_rate
      gamma, #gamma
      exploration_param1, #exploration_param1
      exploration_param2, #exploration_param2
      max_goals_to_try, #max_goals_to_try
      batch_size, #batch_size
      extrinsic_tries_before_eval) #extrinsic_tries_before_eval

agent.train()
