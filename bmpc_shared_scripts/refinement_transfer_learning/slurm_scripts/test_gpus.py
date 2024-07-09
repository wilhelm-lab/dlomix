import tensorflow as tf

print(tf.__version__)
print(f'Number of GPUs available: {len(tf.config.experimental.list_physical_devices("GPU"))}')
print(tf.config.experimental.list_physical_devices('GPU'))
