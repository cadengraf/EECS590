import copy
import os

try:
    from mappo import main
    from mappo_config import config as base_config
except ImportError:
    from quad_physics_ppo.mappo import main
    from quad_physics_ppo.mappo_config import config as base_config


PHYSICS_KEYS = ("dt", "physics_substeps", "max_tilt", "linear_drag", "attitude_tau", "angular_damping", "target_altitude")
SEEDS = (7, 11, 23, 41)


def fixed_physics(cfg):
    cfg.update({"device": "auto", "max_tilt": 0.32, "linear_drag": 0.55, "dt": 0.05, "physics_substeps": 8, "viz_enabled": False})
    for stage in cfg["curriculum"]:
        for key in PHYSICS_KEYS:
            stage.pop(key, None)


def dynamic_stages(cfg):
    return [s for s in cfg["curriculum"] if s.get("randomize_package") or s.get("randomize_deliveries")]


def baseline(_):
    pass


def explore(cfg):
    cfg.update({"ent_coef": 0.012})
    for stage in dynamic_stages(cfg):
        stage.update({"actor_log_std": -0.20, "min_actor_log_std": -0.95, "ent_coef": 0.016})


def stable(cfg):
    cfg.update({"learning_rate": 1e-4, "clip_range": 0.10, "epochs": 4})


def delivery_heavy(cfg):
    cfg.update({"pickup_reward": 100.0, "delivery_reward": 750.0, "team_delivery_bonus": 1500.0})


def longer_dynamic(cfg):
    for stage in dynamic_stages(cfg):
        stage["timesteps"] = int(stage["timesteps"] * 1.4)
        stage["min_timesteps_before_promotion"] = int(stage.get("min_timesteps_before_promotion", 100_000) * 1.25)


SCENARIOS = (("base", baseline), ("explore", explore), ("stable", stable), ("delivery", delivery_heavy), ("long", longer_dynamic))


def tune(tag, seed, fn):
    cfg = copy.deepcopy(base_config)
    fixed_physics(cfg)
    cfg.update({"run_name": f"{cfg['run_name']}_dynamic_slow_sweep_{tag}_seed{seed}", "seed": seed})
    fn(cfg)
    fixed_physics(cfg)
    cfg["total_timesteps"] = sum(stage["timesteps"] for stage in cfg["curriculum"])
    return cfg


if __name__ == "__main__":
    runs = [tune(tag, seed, fn) for tag, fn in SCENARIOS for seed in SEEDS]
    limit = int(os.environ.get("SWEEP_LIMIT", len(runs)))
    for idx, cfg in enumerate(runs[:limit], 1):
        print(f"\n### DYNAMIC MAPPO SWEEP {idx}/{min(limit, len(runs))}: {cfg['run_name']} ###\n", flush=True)
        main(cfg)
