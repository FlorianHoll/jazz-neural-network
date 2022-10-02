"""Execute the training pipeline.

This is simply a wrapper script that only executes the pipeline with
the given .yml config file. As an input to the script, the file path
of the .yml file has to be passed.
"""
from argparse import ArgumentParser

import yaml

from rnn.training import TrainingPipeline

argparser = ArgumentParser()
argparser.add_argument(
    "--config_path",
    type=str,
    help="The file path of the config file.",
)
args = argparser.parse_args()

if __name__ == "__main__":

    with open(args.config_path) as file:
        configs = yaml.safe_load(file)

    pipeline = TrainingPipeline(configs)
    pipeline.train(**configs["training"])
