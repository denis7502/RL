import gym
import numpy as np


class Env():
    def __init__(self, rend_mode="human", seed=1):
        self.env = gym.make('CartPole-v1', render_mode=rend_mode)
        self.obs = self.env.reset(seed=seed)[0]

        self.DISCRETE_OS_SIZE = [25, 25]
        self.real_observation_space = np.array(
            [self.env.observation_space.high[2], 3.5])
        self.discrete_os_win_size = (
            self.real_observation_space * 2 / self.DISCRETE_OS_SIZE)  # step-size

    def step(self, action):
        obs, reward, done, info, _ = self.env.step(action)
        self.obs = obs
        return self.obs, reward, done, info

    def get_discrete_state(self, state):
        trimmed_state = np.array([state[2], state[3]])
        discrete_state = (
            trimmed_state + self.real_observation_space) / self.discrete_os_win_size
        return tuple(discrete_state.astype(int))

    def reset(self, seed=np.random.randint(1, 500)):
        self.obs = self.env.reset(seed=seed)[0]
        return self.obs

    def render(self):
        self.env.render()

    def __exit__(self):
        self.env.close()
