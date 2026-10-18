"""
    cli.py
"""

import argparse
from pathlib import Path
from model import DIR_PATH, Ember, add_new_ember

def output_dir():
    dir_path = DIR_PATH.resolve()

    if not dir_path.exists():
        raise OSError(f"Folder {dir_path} doesn't exist or could not be found.... Current directory {Path('.').resolve()}")
    else:
        print("Current Stored Agents: ")
        for idx, sub_dir in enumerate(dir_path.iterdir(), start=1):
            print(f"{idx}. {sub_dir.stem.lower().capitalize()}")


def main():
    global_parser = argparse.ArgumentParser(description="Promptless - Prompt Chain Replacement Tool")
    subparser = global_parser.add_subparsers(
        title="Subcommands",
        help="Utils to handle mini agents"
    )

    # adding a new mini agent
    add_parser = subparser.add_parser("add", help="Add new agent")
    add_parser.add_argument('--name', type=str, required=True, help='Name of the prompt')
    add_parser.add_argument('--target', nargs='+', help='List of target phrases', required=True)
    add_parser.add_argument('--contrast', nargs='+', help='List of contrasting phrases', required=True)
    add_parser.set_defaults(func=add_new_ember)
    # viewing currently stored agents
    view_parser = subparser.add_parser("view", help="View all current agents")
    view_parser.set_defaults(func=output_dir)
    # loading / invoking stored agents
    load_parser = subparser.add_parser("load", help="load the mini agent")
    load_parser.add_argument('--name', help="Name of the mini agent", default='StopButton')
    load_parser.add_argument('--action', choices=['predict', 'predict_proba'], default='predict')
    load_parser.add_argument('--inputs', type=str, required=True)
    load_parser.set_defaults(func=Ember.load)

    args = global_parser.parse_args()

    if args.func == add_new_ember:
        args.func(
            name=args.name,
            pos=args.target,
            neg=args.contrast
        )
    elif args.func == Ember.load:
        pipe = args.func(
            name=args.name,
        )
        if args.action == 'predict':
            print(pipe.predict(args.inputs))
        else:
            print(pipe.predict_proba(args.inputs))
    else:
        args.func()
