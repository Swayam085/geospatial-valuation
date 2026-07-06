# MDP Design — Dynamic Pricing RL Agent

## Problem Statement
Selling finite inventory (airline seats / hotel rooms) over a limited time window.
Goal: learn an optimal pricing policy that maximizes total revenue before inventory
runs out or the deadline (flight departure / hotel date) passes.

## State Space
State = [remaining_inventory, days_until_departure]

- remaining_inventory: integer, number of unsold units left (e.g., 0 to 100)
- days_until_departure: integer, countdown of days left to sell (e.g., 0 to 30)

## Action Space
Action = price_level

Discrete price levels (starting point):
- Action 0: $100
- Action 1: $150
- Action 2: $200
- Action 3: $250
- Action 4: $300

## Demand Function (Stochastic)
Customer purchase probability depends on:
1. Price set (higher price -> lower probability of purchase)
2. Time left (as deadline approaches, demand behavior shifts)

Simplified formula (to be tuned in Week 1 Day 4):
purchase_probability = base_demand * exp(-k * price) * time_factor(days_until_departure)

Where:
- base_demand: constant representing overall market interest
- k: sensitivity of demand to price
- time_factor: adjusts probability based on urgency (e.g., last-minute demand spikes)

## Reward Function
- If customer buys at the set price: reward = price_charged
- If customer does not buy: reward = 0
- If episode ends with unsold inventory: apply penalty (spoilage cost)

## Episode Termination
Episode ends (done = True) when:
- remaining_inventory == 0 (sold out), OR
- days_until_departure == 0 (deadline reached)

## Next Steps
- Day 2: Build gym.Env skeleton (__init__, reset(), step(), render())
- Day 3: Implement state space in code
- Day 4: Implement demand function
- Day 5: Complete step() with reward logic