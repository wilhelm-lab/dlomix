import os

os.environ['HF_HOME'] = '/cmnfs/proj/bmpc_dlomix/datasets'
os.environ['HF_DATASETS_CACHE'] = '/cmnfs/proj/bmpc_dlomix/datasets/hf_cache'

num_proc = 16
os.environ["OMP_NUM_THREADS"] = f"{num_proc}"
os.environ["TF_NUM_INTRAOP_THREADS"] = f"{num_proc}"
os.environ["TF_NUM_INTEROP_THREADS"] = f"{num_proc}"


import tensorflow as tf
tf.config.threading.set_inter_op_parallelism_threads(num_proc)
tf.config.threading.set_intra_op_parallelism_threads(num_proc)

from dlomix.data import load_processed_dataset

# dataset = load_processed_dataset('/cmnfs/proj/bmpc_dlomix/datasets/processed/refinement_dataset_toy_orig_seq')
dataset = load_processed_dataset('/cmnfs/proj/bmpc_dlomix/datasets/processed/ptm_baseline_small_cleaned_bs1024')

from dlomix.models import PrositIntensityPredictor
from dlomix.losses import masked_spectral_distance, masked_pearson_correlation_distance

model = tf.keras.models.load_model('/cmnfs/proj/bmpc_dlomix/models/baseline_models/noptm_baseline_full_bs1024_unmod_extended/7ef3360f-2349-46c0-a905-01187d4899e2.keras')

from dlomix.refinement_transfer_learning.automatic_rl_tl import AutomaticRlTlTraining, AutomaticRlTlTrainingConfig

config = AutomaticRlTlTrainingConfig(
    dataset=dataset,
    baseline_model=model,
    min_warmup_sequences_new_weights = 100, 
    min_warmup_sequences_whole_model = 100, 
    improve_further = True,
    use_wandb=False
)


trainer = AutomaticRlTlTraining(config) 

new_model = trainer.train()


