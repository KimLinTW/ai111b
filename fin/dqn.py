import pfrl
import torch
import torch.nn
import gym
import numpy
import os

env = gym.make('LunarLander-v2')

class QFunction(torch.nn.Module):

    def __init__(self, obs_size, n_actions):
        super().__init__()
        self.l1 = torch.nn.Linear(obs_size, 400)
        self.l2 = torch.nn.Linear(400, 200)
        self.l3 = torch.nn.Linear(200, n_actions)

    def forward(self, x):
        h = x
        h = torch.nn.functional.relu(self.l1(h))
        h = torch.nn.functional.relu(self.l2(h))
        h = self.l3(h)
        return pfrl.action_value.DiscreteActionValue(h)


obs_size = env.observation_space.low.size
n_actions = env.action_space.n
q_func = QFunction(obs_size, n_actions)
optimizer = torch.optim.Adam(q_func.parameters(), eps=1e-3)
gamma = 0.99
explorer = pfrl.explorers.ExponentialDecayEpsilonGreedy(
    start_epsilon=1.0, end_epsilon=0.01, decay=0.996, random_action_func=env.action_space.sample)
replay_buffer = pfrl.replay_buffers.ReplayBuffer(capacity=10 ** 6)
phi = lambda x: x.astype(numpy.float32, copy=False)
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

n_episodes = 1000
max_episode_len = 500
high = []
avrlen = []
stop = 10
stop2 = 10
run_epi = 0
limit = 300


load = input('run_epi:')
if load == "0" or load == "": print("start training ... ")
if not(os.path.isfile(f"./DQN/agent-{limit}-{stop2}-{load}/model.pt")):
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
        if i % 10 == 0:
            # print('epi', i, '\tR', int(R), "\tlen", sum(avrlen)//len(avrlen), "  \thigh", int(max(high)))
            print(f"{i}\t{int(R)}\t{sum(avrlen)//len(avrlen)}\t{int(max(high))}")
            high=[]
            avrlen=[]
        # if i % 50 == 0:
        #     print('statistics:', agent.get_statistics())
    print('Finished.', run_epi)
    agent.save(f'./DQN/agent-{limit}-{stop2}-{run_epi}')

else:
    print(f"load: {load}")
    agent.load(f'./DQN/agent-{limit}-{stop2}-{load}')

avr_R  = []
avr_len= []
with agent.eval_mode():
    print(f"epi\tlen\tR")
    for i in range(100):
        obs = env.reset()
        R = 0
        t = 0
        while True:
            env.render()
            action = agent.act(obs)
            obs, r, done, _ = env.step(action)
            R += r
            high.append(R)
            t += 1
            reset = t == 500
            # agent.observe(obs, r, done, reset)
            if done or reset: break
        if i%10 == 0: print(f"{i}\t{t}\t{int(R)}")
        avr_R  .append(R)
        avr_len.append(t)

print(f"avr_R  : {sum(avr_R)//len(avr_R)}")
print(f"avr_len: {sum(avr_len)//len(avr_len)}")