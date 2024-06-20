import tensorflow as tf
import dlomix.losses
import wandb
import dlomix

MODEL_PATH = '/cmnfs/proj/prosit_astral/bmpc_dlomix_group/models/baseline_models/noptm_baseline_full_bs1024/85c6c918-4a2a-42e5-aab1-e666121c69a6.keras'

model = tf.keras.models.load_model(MODEL_PATH, custom_objects={'masked_spectral_distance': dlomix.losses.masked_spectral_distance})
print(model.summary())