"""Restricted live trial; the original preview launcher remains available."""
from piper_app.cli.graspnet_target_pick_demo import build_parser as preview_parser


def build_parser():
    parser = preview_parser()
    parser.description = "GraspNet upright-object pickup trial; defaults to dry-run."
    parser.set_defaults(speed_percent=None)  # use execution.speed_percent unless explicitly overridden
    return parser


def main():
    args = build_parser().parse_args()
    if args.replay_scene is not None:
        from piper_app.cli.graspnet_target_pick_demo import main as replay_main
        replay_main()
    else:
        from piper_app.graspnet.live_gui import run
        run(args)


if __name__ == "__main__":
    main()
