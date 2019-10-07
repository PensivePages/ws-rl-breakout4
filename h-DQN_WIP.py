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
        #print("inside Meta_Controller.forward")
        x = self.nonLinearity(self.conv1(x))
        x = self.nonLinearity(self.conv2(x))
        x = self.nonLinearity(self.conv3(x))
        x = x.view(1, -1) # flatten
        x = self.nonLinearity(self.projection(x))
        x = self.output(x)
        return x[0]

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
        #print("inside Controller.forward")
        x = self.nonLinearity(self.conv1(x))
        x = self.nonLinearity(self.conv2(x))
        x = self.nonLinearity(self.conv3(x))
        x = x.view(1, -1) # flatten
        x = self.nonLinearity(self.projection(x))
        x = self.output(x)
        return x[0]

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
                 max_goals_to_try, batch_size, extrinsic_tries_before_eval, conv_stack='stacked'): #conv_stack = stacked' for 4, 1, 84, 84, or 'side_by_side' for 1, 1, 84, 336
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
        self.intrinsic_total = .0001
        self.extrinsic_success_count = 0
        self.total_sessions = .00001
        self.eval_tries = extrinsic_tries_before_eval
        self.done = False
        self.goals = []
        self.actions = list(range(self.env.action_space.n))
        self.num_actions = 0
        self.session_success = False
        self.session_success_list = []
        self.lives = 0
        self.max_steps = 5000
        self.max_score = 400
        self.score = 0
        # reset memory
        C_buffer.__init__(1000000)
        MC_buffer.__init__(50000)

    def change_pixels(self, state, center_pixel: list, padding: int): #state tensor size (N, C, W, H)
        # print("state.size() goal set: ", state.size())
        # for i, row in enumerate(state[0][0]):
        #   print("row %s: "%(i), row)
        staring_row = center_pixel[0] - padding
        starting_column = center_pixel[1] - padding
        side = padding * 2 + 1
        i = staring_row
        while i < staring_row + side:
            j = starting_column
            while j < starting_column + side:
                if i == center_pixel[0] and j == center_pixel[1]:
                    pass
                else:
                    state[0][0][i][j] = 16
                j += 1
            i += 1
        #print("state goal set: ", state[0][0])
        # print("state.size() goal set: ", state.size())
        # for i, row in enumerate(state[0][0]):
        #   print("row %s: "%(i), row)
        # raise NotImplementedError
        return state, center_pixel
    
    def create_goals(self, state):
        #goal_template = torch.ones_like(state)
        # goal 1 (key): change pixels: center: row: 30, column: 9, padding: 1?
        key = lambda state: self.change_pixels(state, [30, 7], 3)
        # goal 2 (middle-ladder): change pixels: center: row: 35, column: 41, padding: _?
        middle_ladder = lambda state: self.change_pixels(state, [25, 41], 3)
        # goal 3 (left-ladder): change pixels: center: row: 59, column: 17, padding: _?
        left_ladder = lambda state: self.change_pixels(state, [55, 13], 3)
        # goal 4 (right-ladder): change pixels: center: row: 59, column: 71, padding: _?
        right_ladder = lambda state: self.change_pixels(state, [59, 71], 3)
        # goal 5 (left-door): change pixels: center: row: 10, column: 10, padding: _?
        left_door = lambda state: self.change_pixels(state, [12, 14], 3)
        # goal 6 (right-door): change pixels: center: row: 10, column: 76, padding: _?
        right_door = lambda state: self.change_pixels(state, [12, 69], 3)
        self.goals = [key, middle_ladder, left_ladder, right_ladder, left_door, right_door]
        self.goals_names = ['key', 'middle_ladder', 'left_ladder', 'right_ladder', 'left_door', 'right_door']

    def init_Q(self, env):
        #print("inside init_Q")
        self.Q1 = defaultdict(lambda: np.zeros(env.action_space.n))
        #^ this way, if the state doesn't exist create it with n zeros = number of actions
        self.Q2 = defaultdict(lambda: np.zeros(len(goals))) #pseudo

    def init_session(self):
        #print("inside init_session")
        # reset state to init
        self.env = utils.wrap_env(gym.make('MontezumaRevenge-v4'))
        state = self.env.reset()
        self.done = False
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
        #ret, observation = cv2.threshold(observation,1,255,cv2.THRESH_BINARY)
        ret, observation = cv2.threshold(observation,20,255,cv2.THRESH_TRUNC)
        #ret, observation = cv2.threshold(observation,0,255,cv2.THRESH_TOZERO)
        return np.reshape(observation,(1,84,84)) #np.reshape(observation,(84,84,1)) #original

    def format_state(self, state, next_state=None, test=False, first=False): #so format_state() shows what the preprocessing changes once
        #print(state)
        # init starting image state with 3 previous void images states
        if self.new_session:
            state = self.preprocess(state)
            state = torch.from_numpy(state).unsqueeze(dim=0).type('torch.FloatTensor')
            ph_state = torch.zeros_like(state)
            #--for 4, 1, 84, 84--
            if conv_stack=='stacked':
                state = torch.cat((state, ph_state), dim=0)
                state = torch.cat((state, ph_state), dim=0)
                state = torch.cat((state, ph_state), dim=0)
            #--for 1, 1, 84, 336--
            else:
                state = torch.cat((state, ph_state), dim=-1)
                state = torch.cat((state, ph_state), dim=-1)
                state = torch.cat((state, ph_state), dim=-1)
            #some test code from preprocess() source
            if first:
                action0 = 0  # do nothing
                observation0, reward0, terminal, info = env.step(action0)
                print("action space: ", env.action_space)
                print("observation space: ", env.observation_space)
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
                print("new observation space: ", env.observation_space)
                #brain.setInitState(observation0) #test model not used
                #brain.currentState = np.squeeze(brain.currentState) #test model not used
        else: #advance state
            next_state = self.preprocess(next_state)
            next_state = torch.from_numpy(next_state).unsqueeze(dim=0).type('torch.FloatTensor')
            state = self.update_state(state, next_state)
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
            raise NotImplementedError
        
        return state.to(device)
    
    def update_state(self, state, next_state):
        #print("state: ", state)
        #print("next_state: ", next_state)
        #--for 4, 1, 84, 84--
        if conv_stack=='stacked':
            state[1:] = state[:-1]
            state[0] = next_state
        #--for 1, 1, 84, 336--
        else:
            state[:, :, :, 84:] = state[:, :, :, :252]
            state[:, :, :, :84] = next_state
        return state

    def critic(self, state, action, goal, goal_name):
        #print("inside critic")
        # execute action in env
        self.env.render(mode='ansi')
        next_state, extrinsic_reward, done, info = self.env.step(action) #pseudo
        self.max_steps += 1
        #print("info: ", info)
        # format next_state
        next_state = self.format_state(state, next_state)
        next_state, goal_pixel = goal(state) # put goal on current state image
        #check color and goal
        # if conv_stack=='stacked':
        #     observationCheck = next_state[0].view(84,84).numpy() # 4, 1, 84, 84
        # else:
        #     observationcheck = next_state[:, :, :, :84].view(84,84).numpy() # 1, 1, 84, 336
        # print("check goal (%s) color: "%(goal_name) + str(np.array(observationCheck).shape))
        # plt.imshow(np.array(np.squeeze(observationCheck)))
        # plt.show()
        # get intrinsic reward
        #print("actual goal pixel (from next_state inside critic): ", next_state[0][0][goal_pixel[0]][goal_pixel[1]])
        #if next_state[0][0][goal_pixel[0]][goal_pixel[1]] == 255:
        if next_state[0][0][goal_pixel[0]][goal_pixel[1]] == 20:
            #print("actual goal pixel (from next_state inside critic): ", next_state[0][0][goal_pixel[0]][goal_pixel[1]])
            #print("Got the (ง°ل͜°)ง %s!"%(goal_name))
            #--show goal images
            # if conv_stack=='stacked':
            #     observationNext = next_state[0].view(84,84).numpy() # 4, 1, 84, 84
            # else:
            #     observationNext = next_state[:, :, :, :84].view(84,84).numpy() # 1, 1, 84, 336
            # #print("w/ goal: " + str(np.array(observationNext).shape))
            # plt.imshow(np.array(np.squeeze(observationNext)))
            # plt.show()
            self.session_success_list.append(goal_name)
            if goal_name == 'key':
                print("!!!!!!!!!!!!!!!!!!!!Got the [̲̅$̲̅(̲̅ ͡° ͜ʖ ͡°̲̅)̲̅$̲̅]  %s  [̲̅$̲̅(̲̅ ͡° ͜ʖ ͡°̲̅)̲̅$̲̅] !!!!!!!!!!!!!!!!!!!!"%(goal_name))
            return extrinsic_reward, 1, next_state, done #intrinsic_reward = 1
        else:
            if info['ale.lives'] < self.lives:
                self.lives -= 1
                return extrinsic_reward, -10, next_state, done #intrinsic_reward = 0
            else:
                return extrinsic_reward, 0, next_state, done #intrinsic_reward = 0

    def select_direction(self, choices, exploration_param): #select goal, or action, greedy vs explore
        #print("inside select_direction")
        random = np.random.uniform(size=1)[0]
        if random <= exploration_param:
            return np.random.choice(np.arange(len(choices)))
        else: #greedy
            return np.argmax(choices.detach().numpy())

    def train(self):
        #print("inside train")
        first = True
        if self.Q1 == None:
            self.init_Q(env)
        bQn = 0 #count for batch to break for training
        while self.num_goals_tried < self.max_goals_to_try:
            if self.new_session == True:
                # reset session
                state, time_step = self.init_session()
                state = self.format_state(state, first=first, test=False)
                self.new_session = False
                first = False
                self.session_success = False
                self.session_success_list = []
                self.lives = 6
                self.max_steps = 5000
                self.score = 0
            Q2_prediction = MC_pred(state) # set Q values for goal at given state
            #print("Q2_prediction: ", Q2_prediction)
            i_goal = self.select_direction(Q2_prediction, self.ep2) # select goal intex
            goal_name = self.goals_names[i_goal]
            #print("goal is (☞ﾟヮﾟ)☞  %s  ☜(ﾟヮﾟ☜)"%(goal_name))
            goal = self.goals[i_goal] # select goal function
            state_goal, goal_pixel = goal(state) # put goal on current state image
            # show goal
            # if conv_stack=='stacked':
            #     observationShow = state_goal[0].view(84,84).numpy() # 4, 1, 84, 84
            # else:
            #     observationShow = state_goal[:, :, :, :84].view(84,84).numpy() # 1, 1, 84, 336
            # #print("Show new goal (%s): "%(goal_name) + str(np.array(observationShow).shape))
            # plt.imshow(np.array(np.squeeze(observationShow)))
            # plt.show()
            self.intrinsic_reward = 0
            while self.intrinsic_reward == 0 and not self.done or self.max_steps < 5000:
                Q1_prediction = C_pred(state_goal) # set Q values for action at given state and goal
                #print("Q1_prediction: ", Q1_prediction)
                i_action = self.select_direction(Q1_prediction, self.ep1) # select action intex
                action = self.actions[i_action] # select action from list of actions by index
                self.num_actions += 1
                #print("action is %s"%(i_action))
                extrinsic_reward, self.intrinsic_reward, next_state, self.done = self.critic(state, action, goal, goal_name) #get rewards and state
                self.score += (extrinsic_reward + self.intrinsic_reward)
                # store transitions
                C_buffer.push(False, state_goal, action, self.intrinsic_reward, next_state) #store controller transition
                MC_buffer.push(True, state, i_goal, extrinsic_reward, next_state) #store meta-controller transition
                # update state
                state = next_state
                # manage batch NN update cycle
                bQn += 1
                if bQn % 5000 == 0 and bQn != 0: #update every batch_size=512
                    #states, next_states, rewards = self.get_random_batch(C_buffer) #pseudo
                    sample = C_buffer.sample(self.batch_size, False)
                    self.update_Q1(sample.state, sample.action, sample.next_state, sample.reward)
                    #states, next_states, rewards = self.get_random_batch(MC_buffer) #pseudo
                    sample = MC_buffer.sample(self.batch_size, True)
                    self.update_Q2(sample.state, sample.goal, sample.next_state, sample.reward)
                # track intrinsic reward success and total
                if self.intrinsic_reward > 0:
                    #print("last goal achieved in %s actions!"%(self.num_actions))
                    self.num_actions = 0
                    self.intrinsic_success_count += 1
                    #print("self.intrinsic_success_count: ", self.intrinsic_success_count)
                    self.session_success = True
                    #utils.show_video()
                self.intrinsic_total += 1
                # track extrinsic reward success
                if extrinsic_reward > 0:
                    self.extrinsic_success_count += 1
            # track number of goals tried so far
            self.num_goals_tried += 1
            # when done reset session
            if self.done or self.max_steps == 5000:
                # track total sessions
                self.total_sessions += 1
                # new session
                self.new_session = True
                print("----done----")
                #print("self.intrinsic_success_count: ", self.intrinsic_success_count)
                #print("self.extrinsic_success_count: ", self.extrinsic_success_count)
                print("self.total_sessions: ", self.total_sessions)
                #print("Session ending score: ", self.score)
                if self.total_sessions % 50 == 0:
                    self.ep1 = self.ep1 - .15
                    print("New action exploration is %s"%(self.ep1))
                #self.ep1 = 1 - (self.score/self.max_score)
                if self.ep1 < .1:
                    self.ep1 = .1 #cap minimum exploration at .1
                #elif self.ep1 > .4:
                #    self.ep1 = .4 #cap max exploration at .5
                #print("New action exploration is %s"%(self.ep1))
                # reset counts
                self.intrinsic_success_count = 0
                self.intrinsic_total = .00001
                self.env.close()
                #utils.show_video()
                if self.session_success and self.total_sessions > 300:
                    print("Success in this video are: ", self.session_success_list)
                    utils.show_video()
                print("------------")
            #--update the exploration params--
            # if self.intrinsic_success_count%20 == 0:
            #     # how often intrinsic goal reached
            #     #print("intrinsic success ratio:  ", self.intrinsic_success_count/self.intrinsic_total)
            #     #self.ep1 = ((1 - (self.intrinsic_success_count/self.intrinsic_total)) + (1 - (self.extrinsic_success_count/self.total_sessions)))/2 #try tying it to the extrinsic goal so it can just repeat the same thing over and over and decide it's winning
            #     print("final score after 20 successes: ", self.score)
            #     self.ep1 = 1 - (self.score/self.max_score)
            #     if self.ep1 < .1:
            #         self.ep1 = .1 #cap minimum exploration at .1
            #     print("New action exploration is %s"%(self.ep1))
            #     # reset counts
            #     self.intrinsic_success_count = 0
            #     self.intrinsic_total = .00001
            if self.num_goals_tried%self.eval_tries == 0:
                # # how often intrinsic goal reached
                # self.ep1 = ((1 - (self.intrinsic_success_count/self.intrinsic_total)) + (.5 * self.ep2))/2 #try tying it to the extrinsic goal so it can just repeat the same thing over and over and decide it's winning
                # if self.ep1 < .1:
                #     self.ep1 = .1 #cap minimum exploration at .1
                # print("New action exploration is %s"%(self.ep1))
                # # how often intrinsic goals lead to extrinsic goal
                # #pass
                # how often extrinsic goal was reached
                self.ep2 = 1 - (self.extrinsic_success_count/self.total_sessions)
                if self.ep2 < .1:
                    self.ep2 = .1 #cap minimum exploration at .1
                print("New goal exploration is %s"%(self.ep2))
        env.close()
        utils.show_video()

    def update_Q1(self, states, actions, next_states, rewards):
        #print("inside update_Q1")
        #states = torch.stack(states, 0).view(512, 1, 84, 336) # works!
        #print("states: ", states.size())
        #next_states = torch.stack(next_states, 0).view(512, 1, 84, 336)
        #print("next_states: ", next_states.size())
        #rewards = torch.stack(rewards, 0)
        #rewards = torch.tensor(rewards)
        #print("rewards: ", rewards.size())
        preds = []
        expect = []
        C_pred.zero_grad()
        for i, _ in enumerate(states):
            #C_pred.zero_grad()
            prediction = C_pred(states[i].to(device))
            prediction = prediction[actions[i]]
            preds.append(prediction)
            expectation = C_targ(next_states[i]) #should I consider if the next_state changed the goal? how?
            expectation = rewards[i] + (self.gamma * torch.max(expectation, dim=0)[0]) #pseudo #notice, as long as (gamma^t_prime-t) is relative to the previous state it's gamma^1 = gamma
            expect.append(expectation)
            # loss = F.mse_loss(prediction, expectation.detach())
            # loss.backward()
            # optimizerC.step()
        #prediction = C_pred(states)
        #prediction = prediction[actions]
        #expectation = C_targ(next_states) #should I consider if the next_state changed the goal? how?
        #expectation = rewards + (self.gamma * torch.max(expectation, dim=0)[0]) #pseudo #notice, as long as (gamma^t_prime-t) is relative to the previous state it's gamma^1 = gamma
        prediction = torch.stack(preds, 0)
        expectation = torch.stack(expect, 0)
        loss = F.mse_loss(prediction, expectation.detach())
        #print("Q1 loss: ", loss)
        loss.backward()
        optimizerC.step()
        #print("end update Q1")
        #raise NotImplementedError

    def update_Q2(self, states, _goals, next_states, rewards):
        #print("inside update_Q2")
        preds = []
        expect = []
        C_pred.zero_grad()
        for i, _ in enumerate(states):
            #MC_pred.zero_grad()
            prediction = MC_pred(states[i].to(device))
            prediction = prediction[_goals[i]]
            preds.append(prediction)
            expectation = MC_targ(next_states[i])
            expectation = rewards[i] + (gamma * torch.max(expectation, dim=0)[0]) #pseudo #notice, as long as (gamma^t_prime-t) is relative to the previous state it's gamma^1 = gamma
            expect.append(expectation)
            # loss = F.mse_loss(prediction, expectation.detach())
            # loss.backward()
            # optimizerMC.step()
        prediction = torch.stack(preds, 0)
        expectation = torch.stack(expect, 0)
        loss = F.mse_loss(prediction, expectation.detach())
        #print("Q2 loss: ", loss)
        loss.backward()
        optimizerC.step()

    def get_random_batch(self, transitions):
        #print("inside get_random_batch")
        #print("np.array(transitions): ", np.array(transitions))
        print("len(transitions): ", len(transitions))
        print("len(np.random.choice(len(transitions),  self.batch_size)): ", len(np.random.choice(len(transitions),  self.batch_size)))
        for _ in transitions:
          print(_)
        random_batch = np.array(transitions)[np.random.choice(len(transitions),  self.batch_size)] #pseudo, but close?
        states = random_batch[:, 'state'] #pseudo
        next_states = random_batch[:, 'next_state'] #pseudo
        rewards = random_batch[:, 'reward'] #pseudo
        return states, next_states, rewards, goals

    def update_df(self):
        pass #may implement if I decide to do gamma^t'- t, where t' = current and t = (future or past)       
            
# environment
#env = gym.make('MontezumaRevenge-v4')
env = utils.wrap_env(gym.make('MontezumaRevenge-v4'))
env.reset()

#---memory---
# meta-controller
D2 = namedtuple('D2',
                    ('state', 'goal', 'reward', 'next_state')) #reward = extrinsic reward
# controller
D1 = namedtuple('D1',
                    ('state', 'action', 'reward', 'next_state')) #reward = intrinsic reward

#---params---
batch_size = 512
conv_stack = 'stacked' # 'stacked' or 'side_by_side' but anything other than 'stacker' will trigger 'side_by_side'

# meta-controller dims
if conv_stack=='stacked':
    MC_conv_out = 4 * 64 * 7 * 7 # for 4, 1, 84, 84
else:
    MC_conv_out = 1 * 64 * 7 * 38 # for 1, 1, 84, 336
MC_hidden_size = 512
MC_number_of_goals = 6 #num_of_goals

# controller dims
if conv_stack=='stacked':
    C_conv_out = 4 * 64 * 7 * 7 # for 4, 1, 84, 84
else:
    C_conv_out = 1 * 64 * 7 * 38 # for 1, 1, 84, 336
C_hidden_size = 512
C_number_of_actions = env.action_space.n #num_of_actions

# # critic dims #not implemented
# input_size_critic = conv_out_size
# hidden_size_critic = conv_out_size
# output_size_critic = 1 #num_of_rewards?

# Agent vars
env = env
MC_buffer = ReplayMemory(1000000)
#MC_buffer = ReplayMemory(50000)
C_buffer = ReplayMemory(50000)
#C_buffer = ReplayMemory(10000)
learning_rate =  0.00025
gamma = .99
exploration_param2 = 1 #start at 1 go to 0.1
exploration_param1 = 1.15 #start at 1 go to 0.1
num_of_goals = 6 #randomly picked 6 ftm #object_detector.goals.n #pseudo
num_of_actions = env.action_space.n
max_goals_to_try = 10000
#batch_size = 512
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
