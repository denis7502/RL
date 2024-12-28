import src.QLearning as QLearning, src.env as env

if __name__ == '__main__':
    envs = env.Env()
    agt = QLearning.QAgent(envs.DISCRETE_OS_SIZE, envs.env.action_space.n)
    agt.load_model("weight.pth")
    for _ in range(1000):
        done = False
        envs.reset()
        while not done:
            action = agt.act(envs.get_discrete_state(envs.obs))
            _, _, done, _ = envs.step(action)
            envs.render()