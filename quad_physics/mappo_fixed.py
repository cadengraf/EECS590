import argparse
import copy
import json
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def get_trainer():
    try:
        from mappo import main as train_mappo
    except ModuleNotFoundError as exc:
        if exc.name != "mappo":
            raise
        from quad_physics.mappo import main as train_mappo
    return train_mappo


def get_base_config():
    try:
        from mappo_fixed_config import config as base_config
    except ModuleNotFoundError as exc:
        if exc.name != "mappo_fixed_config":
            raise
        from quad_physics.mappo_fixed_config import config as base_config
    return base_config


def parse_args():
    parser = argparse.ArgumentParser(description="Train fixed-position Quad MAPPO curriculum.")
    parser.add_argument(
        "--resume",
        nargs="?",
        const="auto",
        default=None,
        metavar="CHECKPOINT",
        help="Resume training. With no value, infer the latest safe resume point; otherwise load CHECKPOINT.",
    )
    parser.add_argument(
        "--resume-stage",
        default=None,
        help="Curriculum stage name to start from when resuming an explicit checkpoint.",
    )
    parser.add_argument(
        "--bayes-tune",
        action="store_true",
        help="Run Bayesian hyperparameter tuning with Optuna instead of a single training run.",
    )
    parser.add_argument("--tune-trials", type=int, default=12, help="Number of Bayesian tuning trials.")
    parser.add_argument("--tune-timeout", type=int, default=None, help="Optional tuning timeout in seconds.")
    parser.add_argument("--tune-study-name", default="quad_physics_mappo_fixed_bayes", help="Optuna study name.")
    parser.add_argument(
        "--tune-storage",
        default=None,
        help="Optional Optuna storage URL, for example sqlite:///mappo_fixed_bayes.db.",
    )
    parser.add_argument(
        "--tune-stages",
        type=int,
        default=3,
        help="Number of curriculum stages to use for each tuning trial.",
    )
    parser.add_argument(
        "--tune-timesteps-scale",
        type=float,
        default=0.25,
        help="Scale each selected stage's timestep budget during tuning.",
    )
    parser.add_argument(
        "--tune-min-stage-timesteps",
        type=int,
        default=60_000,
        help="Minimum timestep budget for each tuning stage.",
    )
    parser.add_argument(
        "--tune-eval-episodes",
        type=int,
        default=8,
        help="Evaluation episodes per stage/final eval during tuning.",
    )
    parser.add_argument("--tune-seed", type=int, default=1000, help="Base random seed for tuning trials.")
    parser.add_argument(
        "--tune-run-prefix",
        default="quad_physics_mappo_fixed_bayes",
        help="Checkpoint run-name prefix for tuning trials.",
    )
    parser.add_argument(
        "--tune-final-run",
        action="store_true",
        help="Train the best found parameters on the full curriculum after tuning.",
    )
    return parser.parse_args()


def suggest_hyperparameters(trial):
    rollout_steps = trial.suggest_categorical("rollout_steps", [512, 1024, 2048])
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
        "ent_coef": trial.suggest_float("ent_coef", 1e-4, 3e-2, log=True),
        "clip_range": trial.suggest_float("clip_range", 0.10, 0.30),
        "gae_lambda": trial.suggest_float("gae_lambda", 0.85, 0.98),
        "gamma": trial.suggest_float("gamma", 0.970, 0.995),
        "vf_coef": trial.suggest_float("vf_coef", 0.25, 1.00),
        "max_grad_norm": trial.suggest_float("max_grad_norm", 0.30, 1.00),
        "hidden_size": trial.suggest_categorical("hidden_size", [64, 128, 256]),
        "rollout_steps": rollout_steps,
        "minibatch_size": trial.suggest_categorical("minibatch_size", [128, 256, 512]),
        "epochs": trial.suggest_int("epochs", 3, 8),
        "actor_log_std": trial.suggest_float("actor_log_std", -0.70, -0.05),
        "min_actor_log_std": trial.suggest_float("min_actor_log_std", -1.20, -0.50),
    }


def tune_stage(stage, trial_config, params):
    stage["timesteps"] = max(
        trial_config["tune_min_stage_timesteps"],
        int(stage["timesteps"] * trial_config["tune_timesteps_scale"]),
    )
    stage["eval_episodes"] = trial_config["stage_eval_episodes"]
    stage["min_timesteps_before_promotion"] = min(
        stage.get("min_timesteps_before_promotion", trial_config["stage_min_timesteps_before_promotion"]),
        stage["timesteps"],
    )
    if "ent_coef" in stage:
        stage["ent_coef"] = params["ent_coef"]
    if "actor_log_std" in stage:
        stage["actor_log_std"] = params["actor_log_std"]
    if "min_actor_log_std" in stage:
        stage["min_actor_log_std"] = params["min_actor_log_std"]


def build_trial_config(base_config, args, trial, params):
    trial_config = copy.deepcopy(base_config)
    trial_config.update(
        {
            "run_name": f"{args.tune_run_prefix}_trial_{trial.number:03d}",
            "seed": args.tune_seed + trial.number,
            "viz_enabled": False,
            "viz_save_images": False,
            "resume_from_checkpoint": None,
            "resume_stage_name": None,
            "stage_eval_episodes": args.tune_eval_episodes,
            "tune_min_stage_timesteps": args.tune_min_stage_timesteps,
            "tune_timesteps_scale": args.tune_timesteps_scale,
            **params,
        }
    )
    selected_stages = copy.deepcopy(trial_config["curriculum"][: args.tune_stages])
    for stage in selected_stages:
        tune_stage(stage, trial_config, params)
    trial_config["curriculum"] = selected_stages
    return trial_config


def score_trial(summary):
    metrics = summary["final_metrics"]
    completed_fraction = summary["completed_stage_count"] / max(summary["total_stages"], 1)
    delivered_fraction = metrics["mean_delivered_agents"] / max(summary.get("final_num_drones", 2), 1)
    return completed_fraction + metrics["success_rate"] + 0.15 * delivered_fraction + 0.001 * metrics["mean_reward"]


def save_best_trial_config(study, best_config):
    output_dir = os.path.join(BASE_DIR, "checkpoints", "mappo", "tuning")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{study.study_name}_best_config.json")
    payload = {
        "study_name": study.study_name,
        "best_value": study.best_value,
        "best_params": study.best_params,
        "config": best_config,
    }
    with open(output_path, "w") as f:
        json.dump(payload, f, indent=4)
    print(f"Saved best tuning config -> {output_path}")


def apply_best_params_to_full_config(base_config, best_params):
    final_config = copy.deepcopy(base_config)
    final_config.update(best_params)
    final_config["run_name"] = f"{base_config['run_name']}_bayes_best"
    final_config["resume_from_checkpoint"] = None
    final_config["resume_stage_name"] = None
    for stage in final_config["curriculum"]:
        if "ent_coef" in stage:
            stage["ent_coef"] = best_params["ent_coef"]
        if "actor_log_std" in stage:
            stage["actor_log_std"] = best_params["actor_log_std"]
        if "min_actor_log_std" in stage:
            stage["min_actor_log_std"] = best_params["min_actor_log_std"]
    return final_config


def run_bayesian_tuning(args):
    if args.tune_trials < 1:
        raise SystemExit("--tune-trials must be at least 1.")
    if args.tune_stages < 1:
        raise SystemExit("--tune-stages must be at least 1.")
    if args.tune_timesteps_scale <= 0:
        raise SystemExit("--tune-timesteps-scale must be greater than 0.")

    try:
        import optuna
    except ImportError as exc:
        raise SystemExit(
            "Bayesian tuning requires Optuna. Install dependencies with `pip install -r requirements.txt` "
            "or install `optuna` directly."
        ) from exc

    base_config = copy.deepcopy(get_base_config())
    train_mappo = get_trainer()

    def objective(trial):
        params = suggest_hyperparameters(trial)
        trial_config = build_trial_config(base_config, args, trial, params)
        print(f"\n=== Bayesian trial {trial.number}: {trial_config['run_name']} ===")
        print(json.dumps(params, indent=2))
        summary = train_mappo(trial_config)
        value = score_trial(summary)
        trial.set_user_attr("summary", summary)
        print(f"Trial {trial.number} score={value:.4f}")
        return value

    sampler = optuna.samplers.TPESampler(seed=args.tune_seed, multivariate=True)
    study = optuna.create_study(
        direction="maximize",
        study_name=args.tune_study_name,
        storage=args.tune_storage,
        load_if_exists=args.tune_storage is not None,
        sampler=sampler,
    )
    study.optimize(objective, n_trials=args.tune_trials, timeout=args.tune_timeout)

    best_config = build_trial_config(base_config, args, study.best_trial, study.best_params)
    save_best_trial_config(study, best_config)
    print(f"Best trial: {study.best_trial.number} | score={study.best_value:.4f}")
    print(json.dumps(study.best_params, indent=2))

    if args.tune_final_run:
        train_mappo(apply_best_params_to_full_config(base_config, study.best_params))


if __name__ == "__main__":
    args = parse_args()
    if args.bayes_tune:
        run_bayesian_tuning(args)
    else:
        train_mappo = get_trainer()
        run_config = copy.deepcopy(get_base_config())
        run_config["resume_from_checkpoint"] = args.resume
        run_config["resume_stage_name"] = args.resume_stage
        train_mappo(run_config)
