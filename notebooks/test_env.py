import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src", "env"))

from pricing_env import DynamicPricingEnv


def run_single_episode(env, verbose=True):
    """Run one episode with a random policy and return total reward."""
    state, info = env.reset()
    total_reward = 0
    done = False
    steps = 0

    while not done:
        action = env.action_space.sample()
        state, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated
        steps += 1

        # Sanity checks (edge cases)
        assert state[0] >= 0, f"ERROR: Inventory went negative! State: {state}"
        assert state[1] >= 0, f"ERROR: Days went negative! State: {state}"

        if verbose:
            print(f"Step {steps}: Action={action}, Price={info['price']}, "
                  f"Bought={info['customer_bought']}, Reward={reward}, State={state}")

    return total_reward, steps


def run_multiple_episodes(env, num_episodes=5):
    """Run several episodes to check consistency and no crashes."""
    print(f"\n--- Running {num_episodes} episodes (sanity check) ---\n")
    rewards = []

    for ep in range(1, num_episodes + 1):
        total_reward, steps = run_single_episode(env, verbose=False)
        rewards.append(total_reward)
        print(f"Episode {ep}: Steps={steps}, Total Reward={total_reward}")

    print(f"\nAverage reward over {num_episodes} episodes: {sum(rewards)/len(rewards):.2f}")
    print(f"Min reward: {min(rewards)}, Max reward: {max(rewards)}")


if __name__ == "__main__":
    env = DynamicPricingEnv()

    print("=== Detailed single episode run ===\n")
    total_reward, steps = run_single_episode(env, verbose=True)
    print(f"\nSingle episode finished in {steps} steps with total reward: {total_reward}")

    run_multiple_episodes(env, num_episodes=5)

    print("\n✅ All sanity checks passed! Environment is working correctly.")