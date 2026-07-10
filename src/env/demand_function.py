import numpy as np


def calculate_purchase_probability(price, days_until_departure, max_days=30,
                                    base_demand=0.5, price_sensitivity=0.005):
    """
    Calculate the probability that a customer purchases at a given price,
    considering how much time is left before the deadline.

    Logic:
    - Higher price -> lower purchase probability (price sensitivity)
    - As days_until_departure decreases (deadline approaches), demand
      urgency increases slightly (last-minute buyers), which raises
      the probability a bit compared to early days.

    Args:
        price (float): The price currently set by the agent.
        days_until_departure (int): Days remaining before deadline.
        max_days (int): Total selling window length (for normalization).
        base_demand (float): Base market interest level (0 to 1).
        price_sensitivity (float): How strongly price affects demand.

    Returns:
        float: Purchase probability between 0 and 1.
    """
    # Price effect: probability decreases exponentially as price increases
    price_effect = np.exp(-price_sensitivity * price)

    # Time urgency effect: as days_until_departure -> 0, urgency_factor -> higher
    # normalized between 0 (start) and 1 (deadline day)
    time_progress = 1 - (days_until_departure / max_days)
    urgency_factor = 1 + (0.3 * time_progress)  # up to 30% boost near deadline

    probability = base_demand * price_effect * urgency_factor

    # Clip between 0 and 1 (probabilities can't exceed these bounds)
    probability = np.clip(probability, 0.0, 1.0)

    return probability


def simulate_purchase(price, days_until_departure, max_days=30):
    """
    Simulate whether a customer actually buys, based on purchase probability.

    Returns:
        bool: True if customer buys, False otherwise.
    """
    prob = calculate_purchase_probability(price, days_until_departure, max_days)
    return np.random.rand() < prob