import copy
import os

try:
    from mappo import main
    from mappo_fixed_sweep import PHYSICS_KEYS, SCENARIOS, SEEDS, base_config
except ImportError:
    from quad_physics_ppo.mappo import main
    from quad_physics_ppo.mappo_fixed_sweep import PHYSICS_KEYS, SCENARIOS, SEEDS, base_config


def slow_physics(cfg):
    cfg.update({"device": "cpu", "max_tilt": 0.32, "linear_drag": 0.55, "dt": 0.05, "physics_substeps": 8})
    for stage in cfg["curriculum"]:
        for key in PHYSICS_KEYS:
            stage.pop(key, None)


def tune(tag, seed, fn):
    cfg = copy.deepcopy(base_config)
    slow_physics(cfg)
    cfg.update({"run_name": f"{cfg['run_name']}_slow_sweep_{tag}_seed{seed}", "seed": seed, "viz_enabled": False})
    fn(cfg)
    slow_physics(cfg)
    cfg["total_timesteps"] = sum(stage["timesteps"] for stage in cfg["curriculum"])
    return cfg


if __name__ == "__main__":
    runs = [tune(tag, seed, fn) for tag, fn in SCENARIOS for seed in SEEDS]
    limit = int(os.environ.get("SWEEP_LIMIT", len(runs)))
    for idx, cfg in enumerate(runs[:limit], 1):
        print(f"\n### SLOW FIXED SWEEP {idx}/{min(limit, len(runs))}: {cfg['run_name']} ###\n", flush=True)
        main(cfg)
