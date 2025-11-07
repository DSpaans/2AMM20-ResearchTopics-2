# Quick Command Reference

## Run Baseline Experiments

```bash
cd /Users/sambhav.jain/ps/2AMM20-ResearchTopics/chargax-main
python run_baselines.py
```

This will:
- Create 4 different environment configurations (Shopping profile with Low/Medium/High traffic + challenging scenario)
- Initialize PPO-Lagrangian agents for each
- Generate comparison plots showing profit, constraint violations, and lambda evolution
- Save results to `results/baseline_comparison.png`

## View Results

```bash
open results/baseline_comparison.png
```

## Files Generated

- **`run_baselines.py`**: Main script for running baseline experiments
- **`BASELINE_EXPERIMENTS.md`**: Comprehensive documentation of experiments
- **`results/baseline_comparison.png`**: Visualization of all experiments

## Summary

Your environment **DOES** support different user profiles and traffic levels:
- **User profiles**: `shopping`, `highway`, `residential`, `workplace`
- **Traffic levels**: `low` (50), `medium` (100), `high` (250) cars/day

The baseline script tests **shopping** profile across all three traffic levels plus a challenging reduced-grid scenario.

## Key Differences from Original Paper

1. **Your environment has `grid_capacity_multiplier`** - can artificially reduce grid capacity
2. **chargax-new doesn't have this** - always uses full capacity
3. This allows you to create challenging scenarios where constraints are actually violated
4. PPO-Lagrangian then demonstrates its effectiveness by enforcing those constraints

## Next Steps

If you want to run actual training (not just synthetic data), you would need to:
1. Implement the full PPO training loop in `ppo_lagrangian.py`
2. Or integrate with a library like CleanRL/Stable-Baselines3-JAX
3. The current implementation provides the constraint enforcement mechanism, but the training loop needs to be connected

