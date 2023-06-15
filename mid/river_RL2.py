# 此程式100%原創 未參考chatGPT、其他網站
import pfrl
import torch
import torch.nn
import numpy as np
from copy import *
from random import *

"""
狀態空間8維 人 狼 羊 菜  |河|  人 狼 羊 菜
            
2. 過河問題

    start
        人 狼 羊 菜
        0  0  0  0

    finish
        人 狼 羊 菜
        1  1  1  1

    避免:
        只有 狼 羊
        [x 1 1 x] and 1的數量==2
        [x 0 0 x] and 0的數量==2
        只有 羊 菜
        [x x 1 1] and 1的數量==2
        [x x 0 0] and 0的數量==2
"""

state = [0,0,0,0]
finall = [1,1,1,1]

class env():
    def __init__(self):
        self.state = [0,0,0,0]
    def reset(self):
        self.state = [0,0,0,0]
    def sample(self):
        return randint(0,5)
    def isDone(self, state):
        return state==[1,1,1,1]
    def step(self, action):
        input("step")
        reward =0
        done = False
        reset = state==[1,1,1,1]
        return (valid_next_state(state),reward,done,reset)
        
        
    def random_act(self):
        return randint(0,11)
        return randint(0,11)
    
def swap(i):
    if i:return 0
    return 1

def get_valid_act(filter, arr, human=0):
    output = []
    for i in range(len(arr)-1):
        if arr[i+1] == filter: output.append(i+1)
    if human != 0 and arr[0]==filter: output.append(0)
    return output

def valid_next_state(state):
    # if state == [1,1,1,1]: return "Done"
    act_set = []
    # if 人 == 0
    if state[0] == 0:
        valid_act = get_valid_act(state[0], state)
        for i in valid_act:
            s = deepcopy(state)
            s[i] = swap(s[i])
            s[0] = swap(s[0])
            act_set.append(s)

    # if 人 == 1
    if state[0] == 1:
        valid_act = get_valid_act(state[0], state)
        for i in valid_act:
            s = deepcopy(state)
            s[i] = swap(s[i])
            s[0] = swap(s[0])
            act_set.append(s)

    state[0] = swap(state[0])
    act_set.append(state) 

    # remove invalid action
    for i in act_set:
        if i[1] == i[2] and len(get_valid_act(i[1], i, 1))==2:
            act_set.remove(i)
            
        elif i[2] == i[3] and len(get_valid_act(i[2], i, 1))==2:
            act_set.remove(i)
    return list(map(list, set(map(tuple, act_set))))

actmap = {}

def act2arr(act):
    return actmap[f"{act}"]

tmp_set = {}
index = 0

for i in range(16):
    binary = format(i, '04b')  # 將數字轉換為4位的二進位表示形式
    binary_list = [int(bit) for bit in binary]  # 將二進位字符串轉換為整數列表
    # print(type(binary_list))
    if binary_list==[1,1,1,1]:break
    for i in valid_next_state(binary_list):
        if f"{i}" not in tmp_set:
            tmp_set[f"{i}"] = index
            # print(f"tmp_set[{f'{i}'}] == {index}")
            actmap[f"{index}"] = i
            index += 1
    # print(binary_list)




class QFunction(torch.nn.Module):
    def __init__(self, obs_size, n_actions):
        super().__init__()
        self.l1 = torch.nn.Linear(obs_size, 128)
        self.l2 = torch.nn.Linear(128, 128)
        self.l3 = torch.nn.Linear(128, n_actions)

    def forward(self, x):
        h = x
        h = torch.nn.functional.relu(self.l1(h))
        h = torch.nn.functional.relu(self.l2(h))
        h = self.l3(h)
        return pfrl.action_value.DiscreteActionValue(h)


obs_size  = 4
n_actions = 11
q_func = QFunction(obs_size, n_actions)
optimizer = torch.optim.Adam(q_func.parameters(), eps=1e-3)
gamma = 0.9
gamma = 0.95
def mysample(): 
    return randint(0, 10)
explorer = pfrl.explorers.ExponentialDecayEpsilonGreedy(
    start_epsilon=1.0, end_epsilon=0.99, decay=0.001, random_action_func=mysample)

# explorer = pfrl.explorers.ConstantEpsilonGreedy(epsilon=0.1, random_action_func=mysample)
replay_buffer = pfrl.replay_buffers.ReplayBuffer(capacity=10 ** 6)
phi = lambda x: x.astype(np.float32, copy=False)
gpu = -1

agent = pfrl.agents.DQN(
    q_func,
    optimizer,
    replay_buffer,
    gamma,
    explorer,
    replay_start_size=500,
    update_interval=1,
    target_update_interval=10,
    phi=phi,
    gpu=gpu,
)

n_episodes = 1000000

End = False
while not(End):
# for i in range(n_episodes):
    # input(i)
    # print(i)
    # obs = env.reset(obs)
    obs = [0,0,0,0]
    obs = np.array(obs)
    record_path = f"{obs}"
    # input(obs)
    # print(obs, end=" --> ")
    R = 0
    reward = 0
    step = 0
    while True:
        action = agent.act(obs)
        action = act2arr(action)
        while action not in valid_next_state(obs.tolist()): 
            action = agent.act(obs)
            action = act2arr(action)
         
            
        step += 1
        if step==20: break
  

        obs = action
        obs = np.array(obs)
        record_path += f" {obs}"
        done = False
        reward -= step
        if f"{obs}" in record_path: reward -= 50
        if np.array_equal(obs, np.array([1, 1, 1, 1])):
            reward += 10000
            reward += (10000//step) * 5
            if step==7: 
                reward += 10000
                End = True
                print("\n"+record_path)
                print("得分",reward)
                print("長度",step)
                input("\nFinish")
                break
            done = True
            agent.observe(obs, reward, done, "") 
            print("\n"+record_path)
            print("得分",reward)
            print("長度",step)
            break
        info = i 
        agent.observe(obs, reward, done, info) 
    # print("\n\n\n")


        
# find:['[0, 0, 0, 0] -> ', '[1, 0, 1, 0] -> ', '[0, 0, 1, 0] -> ', '[1, 1, 1, 0] -> ', '[0, 1, 0, 0] -> ', '[1, 1, 0, 1] -> ', '[0, 1, 0, 1] -> ', '[1, 1, 1, 1]     
# find:['[0, 0, 0, 0] -> ', '[1, 0, 1, 0] -> ', '[0, 0, 1, 0] -> ', '[1, 0, 1, 1] -> ', '[0, 0, 0, 1] -> ', '[1, 1, 0, 1] -> ', '[0, 1, 0, 1] -> ', '[1, 1, 1, 1]  
