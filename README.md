## Week 1 — Dynamic Pricing RL Environment (Project 2)

### Overview
A custom Gymnasium environment simulating dynamic pricing of finite
inventory (e.g., airline seats, hotel rooms) over a limited selling window.

### Files
- `docs/mdp_design.md` — MDP formulation (state, action, reward design)
- `src/env/pricing_env.py` — Custom `gym.Env` implementation
- `src/env/demand_function.py` — Stochastic demand model
- `notebooks/test_env.py` — Sanity checks and multi-episode validation

### State Space
`[remaining_inventory, days_until_departure]`

### Action Space
Discrete price levels: `[100, 150, 200, 250, 300]`

### Reward
- Price charged if customer purchases, else 0
- Penalty applied at episode end for unsold inventory

### How to Test
```bash
python notebooks/test_env.py
```

### Status
✅ Week 1 complete — environment built, tested, and validated with a
random policy. Ready for Week 2 (baseline heuristics + Q-Learning).