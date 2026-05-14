import argparse
import copy

try:
    from mappo import main
    from mappo_fixed_config import config
except ImportError:
    from quad_physics.mappo import main
    from quad_physics.mappo_fixed_config import config


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
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_config = copy.deepcopy(config)
    run_config["resume_from_checkpoint"] = args.resume
    run_config["resume_stage_name"] = args.resume_stage
    main(run_config)
