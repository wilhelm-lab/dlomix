import argparse
import os
from src.model_training import load_config, model_training, combine_into

# parse args
parser = argparse.ArgumentParser(prog='Baseline Model Training')
parser.add_argument('--config', type=str, required=True)
parser.add_argument('--cuda-device-nr', type=str, required=False)
parser.add_argument('--cpu-threads', type=str, required=False)

args = parser.parse_args()


# create config
config = load_config(args.config)
if "project" not in config:
    config["project"] = "baseline model training"

overwritten_params = {
    "processing": {}
}
if args.cuda_device_nr is not None:
    overwritten_params['processing']['cuda_device_nr'] = args.cuda_device_nr
if args.cpu_threads is not None:
    overwritten_params['processing']['num_proc'] = args.cpu_threads

# start run
combine_into(overwritten_params, config)
model_training(config)()