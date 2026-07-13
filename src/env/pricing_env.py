import sys
import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces

sys.path.append(os.path.dirname(__file__))
from demand_function import simulate_purchase


class DynamicPricingEnv(gym.Env):
    """
    Custom Gymnasium environment for dynamic pricing of finite inventory
    (e.g., airline seats, hotel rooms) over a limited selling window.

    The agent sets a price at each time step. A customer may or may not
    purchase based on a stochastic demand function that depends on the
    price and how close the deadline is. The goal is to maximize total
    revenue before the inventory sells out or the deadline passes.

    State:
        [remaining_inventory (int), days_until_departure (int)]

    Action:
        Discrete price level (index into self.price_levels)

    Reward:
        price_charged if a customer buys, else 0.
        A penalty is applied at episode end for unsold inventory.
    """

    def __init__(self, max_inventory: int = 100, max_days: int = 30):
        """
        Args:
            max_inventory (int): Total units available to sell at the start.
            max_days (int): Length of the selling window in days.
        """
        super(DynamicPricingEnv, self).__init__()

        self.max_inventory = max_inventory
        self.max_days = max_days

        # Discrete price levels the agent can choose from
        self.price_levels = [100, 150, 200, 250, 300]
        self.action_space = spaces.Discrete(len(self.price_levels))

        # State: [remaining_inventory, days_until_departure]
        self.observation_space = spaces.Box(
            low=np.array([0, 0]),
            high=np.array([self.max_inventory, self.max_days]),
            dtype=np.int32
        )

        self.remaining_inventory = None
        self.days_until_departure = None

    def reset(self, seed: int = None, options: dict = None):
        """
        Reset the environment to the start of a new selling season.

        Returns:
            state (np.ndarray): Initial [remaining_inventory, days_until_departure]
            info (dict): Additional info (empty at reset)
        """
        super().reset(seed=seed)

        self.remaining_inventory = self.max_inventory
        self.days_until_departure = self.max_days

        state = np.array(
            [self.remaining_inventory, self.days_until_departure],
            dtype=np.int32
        )
        info = {}
        return state, info

    def step(self, action: int):
        """
        Execute one time step (one day) in the environment.

        Args:
            action (int): Index into self.price_levels representing the
                price to charge this day.

        Returns:
            state (np.ndarray): Updated [remaining_inventory, days_until_departure]
            reward (float): Revenue earned this step (with penalty at episode end)
            terminated (bool): Whether the episode has ended
            truncated (bool): Always False (no external time limit besides max_days)
            info (dict): Extra details -- price charged and whether a sale occurred
        """
        price = self.price_levels[action]

        customer_buys = simulate_purchase(
            price, self.days_until_departure, self.max_days
        )

        reward = 0
        if customer_buys and self.remaining_inventory > 0:
            reward = price
            self.remaining_inventory -= 1

        self.days_until_departure -= 1

        terminated = False
        truncated = False

        if self.remaining_inventory <= 0 or self.days_until_departure <= 0:
            terminated = True
            # Penalty for inventory that never sold (spoilage cost)
            if self.remaining_inventory > 0:
                reward -= self.remaining_inventory * 10

        state = np.array(
            [self.remaining_inventory, self.days_until_departure],
            dtype=np.int32
        )
        info = {"price": price, "customer_bought": customer_buys}
        return state, reward, terminated, truncated, info

    def render(self):
        """Print the current state of the environment to the console."""
        print(
            f"Inventory left: {self.remaining_inventory} | "
            f"Days left: {self.days_until_departure}"
        )