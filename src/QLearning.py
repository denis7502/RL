from typing import List
import numpy as np
from . import agent
import pickle


class QAgent(agent.Agent):
    def __init__(self, os_size: List[int], action_space: int):
        super().__init__()
        self.q_table = np.random.uniform(
            low=0, high=1, size=(os_size + [action_space]))
        self.model = None

    def load_model(self, path: str):
        with open(path, 'rb') as handle:
            self.model = pickle.load(handle)

    def act(self, state):        
        action = np.argmax(self.model[state])
        
        return action
