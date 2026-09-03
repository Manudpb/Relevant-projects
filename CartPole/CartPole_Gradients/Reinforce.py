import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

# Uncomment next line if encountering OMP error
#os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# --- Neural Network definitions --- #
# Policy network
class Reinforce(nn.Module):
  def __init__(self, input, output,net_size):
    super(Reinforce, self).__init__()
    mlp = []
    for i in range(len(net_size)):
      if i == 0:
        mlp.append(nn.Linear(input, net_size[i]))
        mlp.append(nn.ReLU())
      else:
        mlp.append(nn.Linear(net_size[i-1], net_size[i]))
        mlp.append(nn.ReLU())

    self.mlp_layers = nn.Sequential(*mlp)
    self.out = nn.Linear(net_size[-1], output)

  def forward(self, x):
    x = self.mlp_layers(x)
    x = self.out(x)
    return torch.softmax(x,1)


# Select an action based on the policy network
def select_action(pnet,state):
  state_tensor = torch.from_numpy(state).float().unsqueeze(0)
  action_probs = pnet(state_tensor)
  distribution = torch.distributions.Categorical(action_probs)
  action = distribution.sample()
  return action.item(), distribution.log_prob(action)

# Compute rewards of the episode
def compute_returns(rewards, gamma):
  returns = []
  for t in range(len(rewards)):
    R = 0
    for k in range(t, len(rewards)):
      R += (gamma ** (k - t)) * rewards[k]
    returns.append(R)
  returns_tensor = torch.tensor(returns)
  return returns_tensor

def moving_average(values, window=50):
    return np.convolve(values, np.ones(window)/window, mode='valid')


network_sizes = [[64], [128],[128,64],[128,64,32]]
#----------------------------------------------------------------------------------------------------------------#
# Main reinforce training loop
#----------------------------------------------------------------------------------------------------------------#
def main(args):
  #hyperparameters
  gamma = 0.99
  learning_rate = 0.001
  num_runs = 5
  max_steps = 10**6
  net_size = network_sizes[args.nsize]


  #initialization
  env = gym.make('CartPole-v1')
  state_dim = env.observation_space.shape[0]
  action_dim = env.action_space.n

  global_records = []


  for run in range(num_runs):
    global_steps = 0
    run_record = []

    reinf = Reinforce(state_dim, action_dim,net_size)
    optimizer = optim.Adam(reinf.parameters(), lr=learning_rate)

    while global_steps < max_steps:
      state,_ = env.reset()
      log_probs = []
      rewards = []

      done = False
      while not done:
        global_steps += 1

        action,log_prob = select_action(reinf,state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        log_probs.append(log_prob)

        rewards.append(reward)
        state = next_state

        if global_steps >= max_steps:
          done = True

      returns = compute_returns(rewards, gamma)
      returns = (returns - returns.mean()) / (returns.std() + 1e-9)

      loss = 0
      for log_prob, G in zip(log_probs, returns):
        loss += -log_prob * G

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      total_reward = sum(rewards)
      run_record.append((global_steps, total_reward))

      print(f"Run: {run + 1}, Global step {global_steps}, Episode Reward: {total_reward}")

      if global_steps >= max_steps:
        print("Max steps")
        break

    global_records.append(run_record)


  env.close()



  plt.figure(figsize=(12, 6))
  all_steps = []
  all_rewards = []
  for run_idx, run_record in enumerate(global_records):
      steps, rewards = zip(*run_record)
      all_steps.append(steps)
      all_rewards.append(rewards)
      plt.plot(steps, rewards, label=f'Run {run_idx+1}', alpha=0.6)

  min_episodes = min(len(r) for r in all_steps)
  avg_steps = [np.mean([all_steps[r][i] for r in range(num_runs)]) for i in range(min_episodes)]
  avg_rewards = [np.mean([all_rewards[r][i] for r in range(num_runs)]) for i in range(min_episodes)]
  plt.plot(avg_steps, avg_rewards, color='black', linewidth=3, label="Average Reward")
  plt.xlabel('Global Training Steps')
  plt.ylabel('Episode Reward')
  plt.title('Learning Curve (Episode Reward vs. Global Steps)')
  plt.legend()
  plt.show()


  plt.figure(figsize=(12, 6))
  all_smoothed_steps = []
  all_smoothed_rewards = []
  for run_idx, run_record in enumerate(global_records):
      steps, rewards = zip(*run_record)
      smoothed_rewards = moving_average(rewards, window=10)
      smoothed_steps = moving_average(steps, window=10)
      all_smoothed_steps.append(smoothed_steps)
      all_smoothed_rewards.append(smoothed_rewards)
      plt.plot(smoothed_steps, smoothed_rewards, label=f'Run {run_idx+1}', alpha=0.6)

  # Compute average smoothed curve using the minimum lengtsh among all runs (more successful runs will have less episodes)
  min_smoothed = min(len(s) for s in all_smoothed_steps)
  avg_smoothed_steps = [np.mean([all_smoothed_steps[r][i] for r in range(num_runs)]) for i in range(min_smoothed)]
  avg_smoothed_rewards = [np.mean([all_smoothed_rewards[r][i] for r in range(num_runs)]) for i in range(min_smoothed)]
  plt.plot(avg_smoothed_steps, avg_smoothed_rewards, color='black', linewidth=3, label="Average Smoothed Reward")
  plt.xlabel('Global Training Steps')
  plt.ylabel('Episode Reward (Smoothed)')
  plt.title('Smoothed Learning Curve (Episode Reward vs. Global Steps)')
  plt.legend()
  plt.show()


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Reinforce algorithm")
  parser.add_argument(
    "--nsize",
    type=int,
    default = 1,
    help="Network size, from 0 to 3 choosing one of the following network sizes: 0:[64], 1:[128], 2:[128,64], 3:[128,64,32]"
  )
  args = parser.parse_args()
  main(args)