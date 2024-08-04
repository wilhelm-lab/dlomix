import tensorflow as tf


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
        
        if self.num_steps < self.ignore_first_n:
            return

        if self.num_steps > self.patience:
            # enough datapoints to do estimation
            if change < self.min_improvement and change < avg_change:
                # we are likely after the inflection point and have a low avg change
                self.patience_counter += 1

                if self.patience_counter >= self.patience:
                    return True
            else:
                self.patience_counter = 0

        self.previous_loss = loss
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