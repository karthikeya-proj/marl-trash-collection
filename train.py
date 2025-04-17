# train.py
from environment import ParkEnvironmentMultiAgent
from dqn_agent import DQNAgent
import torch

num_agents = 2
env = ParkEnvironmentMultiAgent(num_agents=num_agents)
agents = [DQNAgent(state_dim=3, action_dim=4) for _ in range(num_agents)]

num_episodes = 500

for episode in range(num_episodes):
    states = env.reset()
    done = False
    episode_reward = [0 for _ in range(num_agents)]

    while not done:
        env.render()
        actions = [agent.select_action(state, epsilon=0.1) for agent, state in zip(agents, states)]
        next_states, rewards, done, _ = env.step(actions)

        for i, agent in enumerate(agents):
            agent.store_transition((states[i], actions[i], rewards[i], next_states[i], done))
            agent.train_step()

        states = next_states
        for i in range(num_agents):
            episode_reward[i] += rewards[i]

    for agent in agents:
        agent.update_target()

    print(f"Episode {episode+1}: Total rewards {episode_reward}")
