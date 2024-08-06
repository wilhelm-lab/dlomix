import argparse

# parse args
parser = argparse.ArgumentParser(prog='Run Automatic Pipeline')
parser.add_argument('--dataset-path', type=str, required=True)
parser.add_argument('--model-path', type=str, required=False, default=None)
parser.add_argument('--cpu-threads', type=int, required=False, default=16)
parser.add_argument('--cuda-device-nr', type=str, required=False, default=None)

args = parser.parse_args()


import os

if args.cuda_device_nr is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device_nr

os.environ['HF_HOME'] = '/cmnfs/proj/bmpc_dlomix/datasets'
os.environ['HF_DATASETS_CACHE'] = '/cmnfs/proj/bmpc_dlomix/datasets/hf_cache'

num_proc = args.cpu_threads
os.environ["OMP_NUM_THREADS"] = f"{num_proc}"
os.environ["TF_NUM_INTRAOP_THREADS"] = f"{num_proc}"
os.environ["TF_NUM_INTEROP_THREADS"] = f"{num_proc}"

import tensorflow as tf
tf.config.threading.set_inter_op_parallelism_threads(num_proc)
tf.config.threading.set_intra_op_parallelism_threads(num_proc)


from dlomix.data import load_processed_dataset
dataset = load_processed_dataset(args.dataset_path)


from dlomix.models import PrositIntensityPredictor
from dlomix.losses import masked_spectral_distance, masked_pearson_correlation_distance

model = None
if args.model_path is not None:
    model = tf.keras.models.load_model(args.model_path)


import datetime
now = datetime.datetime.now().isoformat()
results_dir = f'results/{now}/'


from dlomix.refinement_transfer_learning.automatic_rl_tl import AutomaticRlTlTraining, AutomaticRlTlTrainingConfig
config = AutomaticRlTlTrainingConfig(
    dataset=dataset,
    baseline_model=model,
    # use_wandb=True,
    improve_further=False,
    results_log=results_dir
)
trainer = AutomaticRlTlTraining(config)

new_model = trainer.train()