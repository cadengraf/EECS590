import copy
import os

try:
    from mappo import main
    from mappo_fixed_config import config as base_config
except ImportError:
    from quad_physics_ppo.mappo import main
    from quad_physics_ppo.mappo_fixed_config import config as base_config


PHYSICS_KEYS = ("dt", "physics_substeps", "max_tilt", "linear_drag", "attitude_tau", "angular_damping", "target_altitude")
SEEDS = (7, 11, 23, 41)


def fixed_physics(cfg):
    cfg.update({"device": "cpu", "max_tilt": 0.36, "linear_drag": 0.25, "dt": 0.05, "physics_substeps": 8})
    for stage in cfg["curriculum"]:
        for key in PHYSICS_KEYS:
            stage.pop(key, None)


def late_stages(cfg):
    return [stage for stage in cfg["curriculum"] if stage["name"].startswith(("s13", "s14", "s15", "s16", "s17", "s19", "s24", "s25"))]


def tune(cfg, tag, seed, fn):
    cfg = copy.deepcopy(cfg)
    fixed_physics(cfg)
    cfg.update({"run_name": f"{cfg['run_name']}_sweep_{tag}_seed{seed}", "seed": seed, "viz_enabled": False})
    fn(cfg)
    cfg["total_timesteps"] = sum(stage["timesteps"] for stage in cfg["curriculum"])
    return cfg


def baseline(_):
    pass


def explore(cfg):
    for stage in late_stages(cfg):
        stage.update({"actor_log_std": -0.15, "min_actor_log_std": -0.90, "ent_coef": 0.018})


def stable(cfg):
    for stage in late_stages(cfg):
        stage.update({"learning_rate": 1e-4, "clip_range": 0.10, "epochs": 4, "ent_coef": 0.010})


def longer_bridges(cfg):
    for stage in late_stages(cfg):
        stage["timesteps"] = int(stage["timesteps"] * 1.5)
        stage["min_timesteps_before_promotion"] = int(stage.get("min_timesteps_before_promotion", 100_000) * 1.25)


def delivery_heavy(cfg):
    cfg.update({"pickup_reward": 100.0, "delivery_reward": 750.0, "team_delivery_bonus": 1500.0})
    for stage in late_stages(cfg):
        stage.update({"promotion_success_rate": min(stage.get("promotion_success_rate", 0.6), 0.60), "continue_success_rate": 0.30})


SCENARIOS = (("base", baseline), ("explore", explore), ("stable", stable), ("long", longer_bridges), ("delivery", delivery_heavy))


if __name__ == "__main__":
    limit = int(os.environ.get("SWEEP_LIMIT", len(SEEDS) * len(SCENARIOS)))
    runs = [tune(base_config, tag, seed, fn) for tag, fn in SCENARIOS for seed in SEEDS]
    for idx, cfg in enumerate(runs[:limit], 1):
        print(f"\n### SWEEP {idx}/{min(limit, len(runs))}: {cfg['run_name']} ###\n", flush=True)
        main(cfg)
