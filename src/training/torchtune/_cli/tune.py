# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
import yaml
from pathlib import Path

from torchtune._cli.cp import Copy
from torchtune._cli.download import Download
from torchtune._cli.ls import List
from torchtune._cli.run import Run
from torchtune._cli.validate import Validate

from huggingface_hub import login

# ROOT_DIR = Path(os.getenv("PROJECT_ROOT", "/issai_qazllm"))
ROOT_DIR = Path("/issai_qazllm")
CREDENTIALS_PATH = ROOT_DIR / "conf" / "credentials.yaml"
print(CREDENTIALS_PATH)

def load_credentials(credentials_path: Path) -> str:
    """Load Hugging Face token from a credentials YAML file."""
    with open(credentials_path, "r") as file:
        return yaml.safe_load(file).get("hf_token", "")


class TuneCLIParser:
    """Holds all information related to running the CLI"""

    def __init__(self):
        # Initialize the top-level parser
        self._parser = argparse.ArgumentParser(
            prog="tune",
            description="Welcome to the torchtune CLI!",
            add_help=True,
        )
        # Default command is to print help
        self._parser.set_defaults(func=lambda args: self._parser.print_help())

        # Add subcommands
        subparsers = self._parser.add_subparsers(title="subcommands")
        Download.create(subparsers)
        List.create(subparsers)
        Copy.create(subparsers)
        Run.create(subparsers)
        Validate.create(subparsers)

    def parse_args(self) -> argparse.Namespace:
        """Parse CLI arguments"""
        return self._parser.parse_args()

    def run(self, args: argparse.Namespace) -> None:
        """Execute CLI"""
        args.func(args)


def main():
    hf_token = load_credentials(CREDENTIALS_PATH)
    login(hf_token)
    print("SUCCESFULLY LOGGED IN")
    parser = TuneCLIParser()
    args = parser.parse_args()
    parser.run(args)


if __name__ == "__main__":
    main()
