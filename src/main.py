'''Single CLI entry point for the whole pipeline: patch extraction, mask
repair, stain augmentation, training, Optuna HPO, and the final
architecture comparison. Run as `python main.py <command> [args...]`; for
`train`/`tune`, anything after the command is passed through to Hydra as a
config override (e.g. `python main.py train model.type=UKAN`).'''

import sys

COMMANDS = {
    "patch": "data.patcher",
    "repair-masks": "data.mask_repair",
    "augment": "data.generate_augs",
    "train": "train",
    "tune": "tune_optuna",
    "test": "test",
}


def _print_usage():
    print(__doc__)
    print(f"Available commands: {', '.join(COMMANDS)}")


def main():
    if len(sys.argv) < 2:
        _print_usage()
        sys.exit(1)

    command = sys.argv[1]

    if command in ("-h", "--help"):
        _print_usage()
        sys.exit(0)

    if command not in COMMANDS:
        print(f"Unknown command: '{command}'\n")
        _print_usage()
        sys.exit(1)

    sys.argv = [sys.argv[0]] + sys.argv[2:]

    module_name = COMMANDS[command]
    module = __import__(module_name, fromlist=["main"])
    module.main()


if __name__ == "__main__":
    main()
