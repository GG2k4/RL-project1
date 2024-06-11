# libraries
import numpy as np
import matplotlib.pyplot as plt

import tqdm


##########
# Environment - DO NOT MODIFY
##########


class WindyGridWorld:
    """
    This is a class for the Windy Grid World task in the Sutton and Barto book.
    Start state is (3, 0) and goal state is (3, 7). The wind is as follows:
    [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]
    The agent can take actions in the four directions. The wind pushes the agent up by the
    number of cells specified in the wind array.
    """

    def __init__(self):
        self.rows = 7
        self.cols = 10
        self.start = (3, 0)
        self.goal = (3, 7)
        self.wind = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]
        self.actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        self.num_actions = len(self.actions)

    def step(self, state, action):
        """
        Take a step in the environment.
        :param state: current state
        :param action: action to take
        :return: next state and reward
        """
        next_state = (state[0] + self.wind[state[1]] + action[0], state[1] + action[1])
        next_state = (
            max(0, min(next_state[0], self.rows - 1)),
            max(0, min(next_state[1], self.cols - 1)),
        )
        reward = -1
        return next_state, reward

    def reset(self):
        return self.start

    def is_terminal(self, state):
        return state == self.goal


#############
# Agent - Complete the methods - qLearning, sarsa, expected_sarsa
#############



class Agent:
    """
    This is a class for the agent in the Windy Grid World task.
    """

    def __init__(self, env, epsilon=0.1, alpha=0.5):
        self.env = env
        self.epsilon = epsilon
        self.alpha = alpha
        self.q_values = np.zeros((self.env.rows, self.env.cols, self.env.num_actions))
        self.policy = (
            np.ones((self.env.rows, self.env.cols, self.env.num_actions))
            / self.env.num_actions
        )

    def qLearning(self, state):
        """
        Q-Learning algorithm.
        :param state: current state
        :return: next state, reward, done
        """
        # Write your code here
        #########
        if np.random.random() < self.epsilon:
            action = np.random.choice(self.env.num_actions)
        else:
            action = self.get_action(state)
        next_state, reward = self.env.step(state, self.env.actions[action])
        next_action = self.get_action(next_state)
        self.q_values[state[0], state[1], action] += self.alpha * (reward + self.q_values[next_state[0], next_state[1], next_action] - self.q_values[state[0], state[1], action])
        return next_state, reward, self.env.is_terminal(next_state)

    def sarsa(self, state, action):
        """
        SARSA algorithm.
        :param state: current state
        :return: next state, reward, done
        """
        # Write your code here
        #########
        next_state, reward = self.env.step(state, self.env.actions[action])
        if np.random.random() < self.epsilon:
            next_action = np.random.choice(self.env.num_actions)
        else:
            next_action = self.get_action(next_state)
        self.q_values[state[0], state[1], action] += self.alpha * (reward + self.q_values[next_state[0], next_state[1], next_action] - self.q_values[state[0], state[1], action])
        return next_state, reward, self.env.is_terminal(next_state), next_action

    def expected_sarsa(self, state):
        """
        Expected SARSA algorithm.
        :param state: current state
        :return: next state, reward, done
        """
        # Write your code here
        #########
        if np.random.random() < self.epsilon:
            action = np.random.choice(self.env.num_actions)
        else:
            action = self.get_action(state)
        next_state, reward = self.env.step(state, self.env.actions[action])
        self.q_values[state[0], state[1], action] += self.alpha * (reward + np.dot(self.policy[next_state[0], next_state[1]], self.q_values[next_state[0], next_state[1]]) - self.q_values[state[0], state[1], action])
        return next_state, reward, self.env.is_terminal(next_state)

    def update_policy(self):
        """
        Update the policy based on the Q-values.
        """
        self.policy = np.eye(self.env.num_actions)[np.argmax(self.q_values, axis=2)]

    def get_action(self, state):
        """
        Get action based on the policy.
        :param state: current state
        :return: action
        """
        return np.random.choice(self.env.num_actions, p=self.policy[state[0], state[1]])


    def reset(self):
        self.q_values = np.zeros((self.env.rows, self.env.cols, self.env.num_actions))
        self.policy = (
            np.ones((self.env.rows, self.env.cols, self.env.num_actions))
            / self.env.num_actions
        )


##########
# Simulate an Episode - DO NOT MODIFY
##########


def simulate_episode(agent, env, algorithm):
    """
    Simulate a single episode.
    :param agent: agent
    :param env: environment
    :param algorithm: algorithm to use
    :return: total reward
    """
    state = env.reset()
    done = False
    total_reward = 0
    action = np.random.choice(env.num_actions)
    while not done:
        if algorithm == "q_learning":
            next_state, reward, done = agent.qLearning(state)
        elif algorithm == "sarsa":
            next_state, reward, done, action = agent.sarsa(state, action)
        elif algorithm == "expected_sarsa":
            next_state, reward, done = agent.expected_sarsa(state)
        state = next_state
        total_reward += reward
        agent.update_policy()
    return total_reward


##########
# Main - DO NOT MODIFY Except line 194. Change the name from windy_grid_world.png to your FullName_BITSID.png
##########

if __name__ == "__main__":
    # first n episodes
    n = 100

    # repetitions
    m = 10

    alphas = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    epsilon = 0.1

    results = {}

    for algorithm in ["sarsa", "q_learning", "expected_sarsa"]:
        avg_reward = []
        for alpha in alphas:
            avg_reward_mruns = []
            for _ in range(m):
                env = WindyGridWorld()
                agent = Agent(env, epsilon, alpha)
                total_reward = 0
                for i in range(n):
                    total_reward += simulate_episode(agent, env, algorithm)
                avg_reward_mruns.append(total_reward)
            avg_reward.append(np.mean(avg_reward_mruns) / n)
        results[algorithm] = avg_reward

    # Plot the results
    plt.figure(figsize=(10, 6))
    for algorithm in results:
        plt.plot(alphas, results[algorithm], label=algorithm)
    plt.xlabel("Alpha")
    plt.ylabel("Average Reward per Episode")
    plt.legend()
    plt.title("Average Reward per Episode vs alpha")
    plt.show()
    plt.savefig("./figures/GarvGupta_2022A7PS0207G.png")
