import pfrl
import torch
import torch.nn
import gym
import numpy
import os

# env = gym.make('LunarLander-v2')
env = gym.make('LunarLanderContinuous-v2')

obs_size = env.observation_space.low.size
n_actions = env.action_space.low.size

q_func = torch.nn.Sequential(
        pfrl.nn.ConcatObsAndAction(),
        torch.nn.Linear(obs_size + n_actions, 400),
        torch.nn.ReLU(),
        torch.nn.Linear(400, 200),
        torch.nn.ReLU(),
        torch.nn.Linear(200, 1),
    )
policy =torch.nn.Sequential(
        torch.nn.Linear(obs_size, 400),
        torch.nn.ReLU(),
        torch.nn.Linear(400, 200),
        torch.nn.ReLU(),
        torch.nn.Linear(200, n_actions), 
        torch.nn.Softmax(dim = 1),
        pfrl.policies.DeterministicHead(),
)

from torch import nn
from pfrl.policies import DeterministicHead
from pfrl.nn import BoundByTanh, ConcatObsAndAction
q_func = nn.Sequential(
    ConcatObsAndAction(),
    nn.Linear(obs_size + n_actions, 400),
    nn.ReLU(),
    nn.Linear(400, 300),
    nn.ReLU(),
    nn.Linear(300, 1),
)
policy = nn.Sequential(
    nn.Linear(obs_size, 400),
    nn.ReLU(),
    nn.Linear(400, 300),
    nn.ReLU(),
    nn.Linear(300, n_actions),
    BoundByTanh(low=env.action_space.low, high=env.action_space.high),
    DeterministicHead(),
)


optimizer_a = torch.optim.Adam(q_func.parameters(), eps=2e-4)
optimizer_c = torch.optim.Adam(q_func.parameters(), eps=3e-4)
opt_a = torch.optim.Adam(policy.parameters())
opt_c = torch.optim.Adam(q_func.parameters())
optimizer_a = opt_a
optimizer_c = opt_c

gamma = 0.964
explorer = pfrl.explorers.ExponentialDecayEpsilonGreedy(
    start_epsilon=0.99, end_epsilon=0.01, decay=0.996, random_action_func=env.action_space.sample)


explorer = pfrl.explorers.ConstantEpsilonGreedy(
    epsilon=0.1, random_action_func=env.action_space.sample)

replay_buffer = pfrl.replay_buffers.ReplayBuffer(capacity=10 ** 6)
phi = lambda x: x.astype(numpy.float32, copy=False)
gpu = -1

def explore_action_func():
    return env.action_space.sample()

agent = pfrl.agents.DDPG(policy = policy,
    q_func = q_func,
    actor_optimizer = optimizer_a,
    critic_optimizer = optimizer_c,
    replay_buffer = replay_buffer,
    gamma = gamma,
    explorer = explorer,
    replay_start_size=64,
    phi=phi,
    target_update_method = 'soft',
    target_update_interval = 1,
    update_interval=1,
    soft_update_tau = 1e-5,
    n_times_update = 1,
    gpu = gpu,
    minibatch_size = 64,
    burnin_action_func = explore_action_func
)

agent = pfrl.agents.DDPG(policy = policy,
    q_func = q_func,
    actor_optimizer = optimizer_a,
    critic_optimizer = optimizer_c,
    replay_buffer = replay_buffer,
    gamma = 0.99,
    explorer = explorer,
    replay_start_size=10000,
    phi=phi,
    target_update_method = 'soft',
    target_update_interval = 1,
    update_interval=1,
    soft_update_tau = 5e-3,
    n_times_update = 1,
    gpu = gpu,
    minibatch_size = 100,
    burnin_action_func = explore_action_func
)








n_episodes = 1000
max_episode_len = 1000
# max_episode_len = 2 ** 7 * 10
high = []
avrlen = []
stop = 10
stop2 = 10
run_epi = 0
limit = 300

# load = input('run_epi:')
load = 974
# load = 0

if load == "0" or load == "": print("start training ... ")
if not(os.path.isfile(f"./DDPG/agent-{limit}-{stop2}-{load}/model.pt")):
    print(f"epi\tR\tlen\thigh")
    for i in range(1, n_episodes + 1):
        obs = env.reset()
        R = 0  # return (sum of rewards) 回合獎勵
        t = 0  # time step
        while True:
            # env.render()
            action = agent.act(obs)
            obs, reward, done, _ = env.step(action)
            R += reward # reward 是 單步獎勵!
            t += 1
            reset = t == max_episode_len
            agent.observe(obs, reward, done, reset)
            if R >= limit:
                stop -= 1
                if stop == 0: run_epi = i
            if done or reset:break
        if run_epi != 0: break
        high.append(R)
        avrlen.append(t)
        if i % 5 == 0:
            # print('epi', i, '\tR', int(R), "\tlen", sum(avrlen)//len(avrlen), "  \thigh", int(max(high)))
            print(f"{i}\t{int(R)}\t{sum(avrlen)//len(avrlen)}\t{int(max(high))}")
            high=[]
            avrlen=[]
        # if i % 50 == 0:
        #     print('statistics:', agent.get_statistics())
    print('Finished.', run_epi)
    agent.save(f'./DDPG/agent-{limit}-{stop2}-{run_epi}')
else:
    print(f"load: {load}")
    agent.load(f'./DDPG/agent-{limit}-{stop2}-{load}')


avr_R  = []
avr_len= []
with agent.eval_mode():
    print(f"epi\tlen\tR")
    for i in range(10):
        obs = env.reset()
        R = 0
        t = 0
        while True:
            # env.render()
            action = agent.act(obs)
            obs, r, done, _ = env.step(action)
            R += r
            high.append(R)
            t += 1
            reset = t == 500000
            # agent.observe(obs, r, done, reset)
            if done or reset: break
        # if i%1 == 0: print(f"{i}\t{t}\t{int(R)}")
        avr_R  .append(R)
        avr_len.append(t)

print(f"avr_R  : {sum(avr_R)//len(avr_R)}")
print(f"avr_len: {sum(avr_len)//len(avr_len)}")# avr_R  = []
# avr_len= []
# with agent.eval_mode():
#     print(f"epi\tlen\tR")
#     for i in range(1000):
#         obs = env.reset()
#         R = 0
#         t = 0
#         while True:
#             # env.render()
#             action = agent.act(obs)
#             obs, r, done, _ = env.step(action)
#             R += r
#             high.append(R)
#             t += 1
#             reset = t == 500
#             # agent.observe(obs, r, done, reset)
#             if done or reset: break
#         if i%100 == 0: print(f"{i}\t{t}\t{int(R)}")
#         avr_R  .append(R)
#         avr_len.append(t)

# print(f"avr_R  : {sum(avr_R)//len(avr_R)}")
# print(f"avr_len: {sum(avr_len)//len(avr_len)}")

