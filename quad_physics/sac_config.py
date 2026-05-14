config = {
    "run_name": "quad_physics_sac_curriculum_v1",
    "seed": 7,

    # SAC settings
    "learning_rate": 3e-4,
    "gamma": 0.99,
    "buffer_size": 200_000,
    "learning_starts": 2_000,
    "batch_size": 256,
    "train_freq": 1,
    "gradient_steps": 1,
    "tau": 0.005,

    # SAC entropy/exploration
    # "auto" lets SAC tune exploration automatically
    "ent_coef": "auto",

    # Network
    "net_arch": [128, 128],

    # Rewards
    "step_penalty": -0.2,
    "invalid_move_penalty": 200.0,
    "revisit_penalty": 0.05,
    "revisit_penalty_cap": 1.5,
    "backtrack_penalty": 1.0,
    "pickup_reward": 300.0,
    "delivery_reward": 1000.0,
    "undelivered_package_penalty": 300.0,

    # Drone physics
    "dt": 0.05,
    "physics_substeps": 8,
    "max_tilt": 0.55,
    "linear_drag": 1.0,
    "attitude_tau": 0.18,
    "angular_damping": 0.82,
    "target_altitude": 1.0,

    # Printing / eval
    "env_print_freq": 0,
    "stage_eval_episodes": 20,
    "reset_replay_each_stage": True,
    "viz_enabled": True,
    "viz_delay": 0.12,
    "viz_save_path": "sac_quad_rollout.gif",
    "viz_save_images": True,
    "viz_image_dir": "sac_quad_rollout_images",

    "curriculum": [
        {
            "name": "s1_line_two_rewards_physics",
            "map_layout": "straight_line",
            "line_length": 4,
            "map_padding": 1,
            "max_steps": 80,
            "timesteps": 80_000,

            "fixed_start": [1, 1],
            "fixed_package": [1, 2],
            "fixed_delivery": [1, 3],

            "randomize_package": False,
            "randomize_delivery": False,

            "max_tilt": 0.55,
            "linear_drag": 1.0,
        },

        {
            "name": "s2_line_longer_physics",
            "map_layout": "straight_line",
            "line_length": 7,
            "map_padding": 1,
            "max_steps": 140,
            "timesteps": 120_000,

            "fixed_start": [1, 1],
            "fixed_package": [1, 3],
            "fixed_delivery": [1, 6],

            "randomize_package": False,
            "randomize_delivery": False,

            "max_tilt": 0.55,
            "linear_drag": 1.0,
        },

        {
            "name": "s3_t_junction_vertical_first",
            "map_layout": "t_junction",
            "stem_length": 5,
            "branch_left": 2,
            "branch_right": 2,
            "map_padding": 1,
            "max_steps": 240,
            "timesteps": 250_000,

            "fixed_start": [5, 3],
            "fixed_package": [3, 3],
            "fixed_delivery": [1, 3],

            "randomize_package": False,
            "randomize_delivery": False,

            "max_tilt": 0.50,
            "linear_drag": 1.0,
        },

        {
            "name": "s4_t_junction_turn_right",
            "map_layout": "t_junction",
            "stem_length": 5,
            "branch_left": 2,
            "branch_right": 2,
            "map_padding": 1,
            "max_steps": 300,
            "timesteps": 500_000,

            "fixed_start": [5, 3],
            "fixed_package": [3, 3],
            "fixed_delivery": [1, 5],

            "randomize_package": False,
            "randomize_delivery": False,

            "max_tilt": 0.50,
            "linear_drag": 1.0,
        },

        {
            "name": "s5_fixed_pipe_tiny",
            "grid_size": [2, 2],
            "map_seed": 42,
            "loop_prob": 0.25,
            "max_steps": 260,
            "timesteps": 300_000,

            "fixed_start": [2, 3],
            "fixed_package": [2, 5],
            "fixed_delivery": [2, 7],

            "randomize_package": False,
            "randomize_delivery": False,

            "max_tilt": 0.50,
            "linear_drag": 0.8,
        },
    ],
}

config["total_timesteps"] = sum(stage["timesteps"] for stage in config["curriculum"])
