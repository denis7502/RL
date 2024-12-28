class Agent:
    def __init__(self):
        pass
    
    def act(self, state):
        # State: self.cart_pos, self.cart_vel, self.ang, self.ang_vel
        
        if state[2] > 0: return 1 
        else: return 0
        
    def load_model(self, path):
        pass