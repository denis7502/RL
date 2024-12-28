
import numpy as np
import pickle
import tqdm
import src.QLearning as QLearning, src.env as env

def train():

    dict_params = {
        "learning_rate": 0.05,
        "discount": 0.95,
        "episodes": 1_200_000,
        "log_frequency": 200,
        "epsilon": 0.15,
        "start_decay": 1}

    envr = env.Env('None')
    q = QLearning.QAgent(envr.DISCRETE_OS_SIZE, envr.env.action_space.n)

    for episode in tqdm.tqdm(range(dict_params["episodes"])):

        discrete_state = envr.get_discrete_state(envr.reset())
        done = False
        # One iteration of the environment
        while not done:
            # Using epsilon to introduce exploration
            if np.random.random() > dict_params["epsilon"]:
                action = np.argmax(q.q_table[discrete_state])
            else:
                action = np.random.randint(0, 2)
            new_state, reward, done, _ = envr.step(action)
            new_discrete_state = envr.get_discrete_state(new_state)

            # if render:
            #     envr.render()

                # Adjusting the values in our Q-table according to the Q-learning formula
            if not done:
                max_future_q = np.max(q.q_table[new_discrete_state])
                current_q = q.q_table[discrete_state + (action, )]
                new_q = (1 - dict_params["learning_rate"]) * current_q + dict_params["learning_rate"] * (
                    reward + dict_params["discount"] * max_future_q)
                q.q_table[discrete_state + (action, )] = new_q
                discrete_state = new_discrete_state
        # Decay epsilon
        if dict_params["episodes"] // 2 >= episode >= 1:
            dict_params["epsilon"] -= dict_params["epsilon"] / \
                (dict_params["episodes"] // 2 - 1)

    save_model(q.q_table)


def save_model(q_table):
    with open('weight.pth', 'wb') as handle:
        pickle.dump(q_table, handle, protocol=pickle.HIGHEST_PROTOCOL)


train()
