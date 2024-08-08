import tensorflow as tf
from tensorflow.keras.callbacks import Callback
from typing import Optional
import math

class CustomCSVLogger(tf.keras.callbacks.Callback):
    def __init__(self, filename, separator=',', append=True):
        super().__init__()
        self.filename = filename
        self.separator = separator
        self.append = append
        self.file_writer = None
        self.keys = ['phase', 'epoch', 'batch', 'learning_rate', 'loss', 'masked_pearson_correlation_distance', 'val_loss', 'val_masked_pearson_correlation_distance']
        self.epoch = 0
        self.batch_counter = 0
        self.val_loss = None
        self.val_masked_pearson_correlation_distance = None
        self.phase = 0

    def on_train_begin(self, logs=None):
        mode = 'a' if self.append else 'w'
        self.file_writer = open(self.filename, mode)
        # Set up headers if file is empty
        if not self.append or self.file_writer.tell() == 0:
            header = self.separator.join(self.keys)
            self.file_writer.write(header + '\n')

    def on_batch_end(self, batch, logs=None):
        logs = logs or {}
        self.batch_counter += 1
        logs['phase'] = self.phase
        logs['epoch'] = self.epoch
        logs['batch'] = self.batch_counter
        logs['learning_rate'] = float(tf.keras.backend.get_value(self.model.optimizer.lr))

        # Ensure all keys are present, even if some values are missing
        data_to_log = {key: logs.get(key, '') for key in self.keys}
        data_to_log['val_loss'] = self.val_loss
        data_to_log['val_masked_pearson_correlation_distance'] = self.val_masked_pearson_correlation_distance

        # Write the log data for the current batch
        row = [str(data_to_log.get(key, '')) for key in self.keys]
        row_line = self.separator.join(row)
        self.file_writer.write(row_line + '\n')
        self.file_writer.flush()

    def on_epoch_end(self, epoch, logs=None):
        self.epoch += 1

    def on_train_end(self, logs=None):
        if self.file_writer:
            self.file_writer.close()

    def reset_phase(self):
        """Resets the epoch counter and increments the phase."""
        self.phase += 1

    def set_validation_metrics(self, val_loss, val_masked_pearson_correlation_distance):
        self.val_loss = val_loss
        self.val_masked_pearson_correlation_distance = val_masked_pearson_correlation_distance



class BatchEvaluationCallback(tf.keras.callbacks.Callback):
    def __init__(self, evaluate_model_func, batch_interval):
        super().__init__()
        self.evaluate_model_func = evaluate_model_func
        self.batch_interval = batch_interval
        self.batch_counter = 0

    def on_batch_end(self, batch, logs=None):
        self.batch_counter += 1
        if self.batch_counter % self.batch_interval == 0:
            self.evaluate_model_func()


class OverfittingEarlyStopping(tf.keras.callbacks.Callback):
    max_validation_train_difference : float

    def __init__(self, max_validation_train_difference):
        super().__init__()
        self.max_validation_train_difference = max_validation_train_difference

    def on_epoch_end(self, epoch, logs=None):
        train_loss = logs['loss']
        val_loss = logs['val_loss']

        if math.isfinite(train_loss) and math.isfinite(val_loss):
            if val_loss - train_loss < self.max_validation_train_difference:
                return
        
        self.model.stop_training = True



class InflectionPointDetector:
    min_improvement : float
    patience : int
    ignore_first_n : int
    smoothing_window : int
    wandb_log : bool
    wandb_log_name : str = 'InflectionPointDetector'

    change_sum : float = 0
    num_steps : int = 0
    previous_loss : float = 0
    global_min : float = float('inf')
    initial_loss : float = None
    patience_counter : int = 0
    current_changes : list[float] = []

    def __init__(self, min_improvement : float, patience : int, ignore_first_n : int = 0, wandb_log : bool = False):
        self.min_improvement = min_improvement
        self.patience = patience
        self.ignore_first_n = ignore_first_n
        self.smoothing_window = 3 * patience
        self.wandb_log = wandb_log
        
        if self.wandb_log:
            global wandb
            import wandb
    
    def reset_detector(self):
        # self.change_sum = 0
        # self.num_steps = 0 
        self.patience_counter = 0
        # self.current_changes = []

    
    def inflection_reached(self, loss : float):
        if loss < self.global_min:
            self.global_min = loss

        if self.initial_loss is None:
            self.initial_loss = loss

        loss = self.initial_loss - self.global_min

        change = loss - self.previous_loss
        self.change_sum += change
        self.num_steps += 1

        self.current_changes.append(change)
        if len(self.current_changes) > self.smoothing_window:
            self.current_changes.pop(0)
        change = sum(self.current_changes) / len(self.current_changes)

        avg_change = self.change_sum / self.num_steps 

        if self.wandb_log:
            wandb.log({
                f'{self.wandb_log_name}_avg_change': avg_change,
                f'{self.wandb_log_name}_current_change': change,
                f'{self.wandb_log_name}_current_patience': self.patience_counter / self.patience
            })
        
        self.previous_loss = loss
        
        if self.num_steps < self.ignore_first_n:
            return False

        if self.num_steps > self.patience:
            # enough datapoints to do estimation
            if change < self.min_improvement and change < avg_change:
                # we are likely after the inflection point and have a low avg change
                self.patience_counter += 1

                if self.patience_counter >= self.patience:
                    return True
            else:
                self.patience_counter = 0

        return False


class InflectionPointEarlyStopping(tf.keras.callbacks.Callback, InflectionPointDetector):
    stopped_early : bool = False

    def __init__(self, *args, **kwargs):
        InflectionPointDetector.__init__(self, *args, **kwargs)
        tf.keras.callbacks.Callback.__init__(self)
        self.wandb_log_name = 'InflectionPointEarlyStopping'

    def on_train_batch_end(self, batch, logs):
        loss = logs['loss']
        
        if self.inflection_reached(loss):
            self.stopped_early = True
            self.model.stop_training = True

class InflectionPointLRReducer(tf.keras.callbacks.Callback, InflectionPointDetector):
    factor : float

    def __init__(self, factor : float, *args, **kwargs):
        InflectionPointDetector.__init__(self, *args, **kwargs)
        tf.keras.callbacks.Callback.__init__(self)
        self.wandb_log_name = 'InflectionPointLRReducer'
        self.factor = factor

    def on_train_batch_end(self, batch, logs):
        loss = logs['loss']
        
        if self.inflection_reached(loss):
            lr = self.model.optimizer.lr.read_value()
            lr *= self.factor
            self.model.optimizer.lr.assign(lr)
            self.reset_detector()


class LearningRateWarmupPerStep(tf.keras.callbacks.Callback):
    num_steps : int
    start_lr : float
    end_lr : float

    steps_counter : int = 0

    def __init__(self, num_steps : int, start_lr : float, end_lr : float, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.num_steps = num_steps
        self.start_lr = start_lr
        self.end_lr = end_lr

    def on_train_batch_begin(self, batch, logs):
        lr = self.model.optimizer.lr.read_value()
        if self.steps_counter < self.num_steps:
            factor = self.steps_counter / self.num_steps
            # lr = factor * self.end_lr + (1-factor) * self.start_lr
            lr = self.end_lr ** factor * self.start_lr ** (1-factor)
            self.model.optimizer.lr.assign(lr)
        
        self.steps_counter += 1