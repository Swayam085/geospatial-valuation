import gymnasium as gym
from gymnasium import spaces
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(__file__))


class DynamicPricingEnv(gym.Env):
    """
    Custom Gym environment for dynamic pricing of finite inventory
    (e.g., airline seats, hotel rooms) over a limited selling window.

    State  = [remaining_inventory, days_until_departure]
    Action = price_level (discrete)
    Reward = price_charged if customer buys, else 0
    """

    def __init__(self, max_inventory=100, max_days=30):
        super(DynamicPricingEnv, self).__init__()

        # ---- Config ----
        self.max_inventory = max_inventory
        self.max_days = max_days

        # ---- Action space: discrete price levels ----
        # 0 -> $100, 1 -> $150, 2 -> $200, 3 -> $250, 4 -> $300
        self.price_levels = [100, 150, 200, 250, 300]
        self.action_space = spaces.Discrete(len(self.price_levels))

        # ---- State space: [remaining_inventory, days_until_departure] ----
        self.observation_space = spaces.Box(
            low=np.array([0, 0]),
            high=np.array([self.max_inventory, self.max_days]),
            dtype=np.int32
        )

        # ---- Internal state ----
        self.remaining_inventory = None
        self.days_until_departure = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Reset state to starting values
        self.remaining_inventory = self.max_inventory
        self.days_until_departure = self.max_days

        state = np.array(
            [self.remaining_inventory, self.days_until_departure],
            dtype=np.int32
        )
        info = {}
        return state, info

    def step(self, action):
        from demand_function import simulate_purchase

        price = self.price_levels[action]

        # Simulate whether customer buys at this price
        customer_buys = simulate_purchase(
            price, self.days_until_departure, self.max_days
        )

        reward = 0
        if customer_buys and self.remaining_inventory > 0:
            reward = price
            self.remaining_inventory -= 1

        # Time always moves forward
        self.days_until_departure -= 1

        terminated = False
        truncated = False

        # Episode ends if sold out or deadline reached
        if self.remaining_inventory <= 0 or self.days_until_departure <= 0:
            terminated = True
            # Penalty for unsold inventory at episode end
            if self.remaining_inventory > 0:
                reward -= self.remaining_inventory * 10  # spoilage penalty

        state = np.array(
            [self.remaining_inventory, self.days_until_departure],
            dtype=np.int32
        )
        info = {"price": price, "customer_bought": customer_buys}
        return state, reward, terminated, truncated, info

    def render(self):
        print(
            f"Inventory left: {self.remaining_inventory} | "
            f"Days left: {self.days_until_departure}"
        )