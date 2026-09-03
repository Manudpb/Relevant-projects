import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse



# Uncomment next line if encountering OMP error
#os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


# --- Neural Network definitions --- #
# Actor network
class Reinforce_actor(nn.Module):
  def __init__(self, input, output):
    super(Reinforce_actor, self).__init__()
    self.fc1 = nn.Linear(input, 128)
    self.policy = nn.Linear(128, output)

    
  def forward(self, x):
    x = torch.relu(self.fc1(x))
    policy_prob = self.policy(x)
    policy = torch.softmax(policy_prob, dim=1)
    return policy

# Critic network
class Reinforce_critic(nn.Module):
  def __init__(self, input):
    super(Reinforce_critic, self).__init__()
    self.fc1 = nn.Linear(input, 128)
    self.reward = nn.Linear(128, 1)

    
  def forward(self, x):
    x = torch.relu(self.fc1(x))
    value = self.reward(x)
    return value

# Select an action based on the policy network
def select_action(pnet,state):
  state_tensor = torch.from_numpy(state).float().unsqueeze(0)
  action_probs = pnet(state_tensor)
  distribution = torch.distributions.Categorical(action_probs)
  action = distribution.sample()
  return action.item(), distribution.log_prob(action)

# Calculate the returns with n-step TD bootstrap
def compute_returns(gamma, n, trace, critic):
  Qn_val = []
  for t in range(len(trace)):
    R = 0
    for k in range(n):
      if t+k < len(trace):
        _, _, r_k, _ = trace[t+k]
        R += (gamma ** k) * r_k
      else:
        break

    if t+n < len(trace):
      s_tn = torch.tensor(trace[t+n][0], dtype = torch.float32).unsqueeze(0)
      value_stn = critic(s_tn)
      R += (gamma ** n) * value_stn.item()

    Qn_val.append(R)
  return Qn_val

def moving_average(values, window=50):
    return np.convolve(values, np.ones(window)/window, mode='valid')


#----------------------------------------------------------------------------------------------------------------#
# Main actor critic training loop
#----------------------------------------------------------------------------------------------------------------#

def main(args):
  #hyperparameters
  gamma = 0.99
  learning_rate = 0.001
  num_runs = 5
  n = args.n          
  max_steps = 10**6

  #initialization
  env = gym.make('CartPole-v1')
  state_dim = env.observation_space.shape[0]
  action_dim = env.action_space.n

  global_records = []

  for run in range(num_runs):
    reinf_actor = Reinforce_actor(state_dim, action_dim)
    reinf_critic = Reinforce_critic(state_dim)
    optimizer_actor = optim.Adam(reinf_actor.parameters(), lr=learning_rate)
    optimizer_critic = optim.Adam(reinf_critic.parameters(), lr=learning_rate)
  
    run_record = []
    global_steps = 0
  
    while global_steps < max_steps:
      state, _ = env.reset()
      trace = []
      done = False
      ep_reward = 0

      #start running episodes
      while not done:

        global_steps += 1

        action,log_prob = select_action(reinf_actor,state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
      
        trace.append((state, action, reward, log_prob))
        ep_reward += reward
        state = next_state

        if global_steps >= max_steps:
          done = True
          break

      returns = compute_returns(gamma, n, trace, reinf_critic)

      #normalize returns in theory helps with stability
      returns = torch.tensor(returns, dtype=torch.float32)
      returns = (returns - returns.mean()) / (returns.std() + 1e-9)

      #Update the critic parameters
      critic_loss = 0
      for t in range(len(trace)):
        state_t = torch.tensor(trace[t][0], dtype=torch.float32).unsqueeze(0)
        V_st = reinf_critic(state_t)
        target = torch.tensor(returns[t], dtype=torch.float32).unsqueeze(0)
        critic_loss += nn.functional.mse_loss(V_st, target)

      optimizer_critic.zero_grad()
      critic_loss.backward()
      optimizer_critic.step()

      #Update the actor parameters
      actor_loss = 0
      for t in range(len(trace)):
        state_t = torch.tensor(trace[t][0], dtype=torch.float32).unsqueeze(0)
        _, _, _, log_prob_t = trace[t]
        actor_loss += -returns[t] * log_prob_t

      optimizer_actor.zero_grad()
      actor_loss.backward()
      optimizer_actor.step()

      run_record.append((global_steps, ep_reward))
      print(f"Run: {run+1}, Global Step: {global_steps}, Episode Reward: {ep_reward}")

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

  #smoothed plots
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
    
  # Compute average smoothed curve using the minimum lengtsh among all runs 
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
  parser = argparse.ArgumentParser(description="Actor-Critic with TD bootstrapping")
  parser.add_argument(
    "--n",
    type=int,
    default = 15,
    help="Number of steps for n-step TD bootstrapping (default: 15), we have tried 5 and 20 as well"
  )
  args = parser.parse_args()
  main(args)

