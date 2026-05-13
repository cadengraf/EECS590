try:
    from mappo import main
    from mappo_fixed_config import config
except ImportError:
    from quad_physics_ppo.mappo import main
    from quad_physics_ppo.mappo_fixed_config import config


if __name__ == "__main__":
    main(config)
