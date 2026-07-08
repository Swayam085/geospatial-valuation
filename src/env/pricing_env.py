import gymnasium as gym
from gymnasium import spaces
import numpy as np


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
        # NOTE: demand function and reward logic will be implemented
        # on Day 4 and Day 5. This is just the skeleton for now.
        price = self.price_levels[action]

        # Placeholder logic (to be replaced with real demand function)
        reward = 0
        terminated = False
        truncated = False
        info = {"price": price}

        # Time always moves forward
        self.days_until_departure -= 1

        if self.remaining_inventory <= 0 or self.days_until_departure <= 0:
            terminated = True

        state = np.array(
            [self.remaining_inventory, self.days_until_departure],
            dtype=np.int32
        )
        return state, reward, terminated, truncated, info

    def render(self):
        print(
            f"Inventory left: {self.remaining_inventory} | "
            f"Days left: {self.days_until_departure}"
        )