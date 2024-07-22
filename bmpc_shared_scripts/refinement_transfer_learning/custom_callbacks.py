import tensorflow as tf

class InflectionPointEarlyStopping(tf.keras.callbacks.Callback):
    min_improvement : float
    patience : int
    ignore_first_n : int

    change_sum : float = 0
    num_steps : int = 0
    previous_loss : float = None
    patience_counter : int = 0
    stopped_early : bool = False

    def __init__(self, min_improvement : float, patience : int, ignore_first_n : int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.min_improvement = min_improvement
        self.patience = patience
        self.ignore_first_n = ignore_first_n

    def on_train_batch_end(self, batch, logs):
        loss = logs['loss'].numpy()
        if self.previous_loss is None:
            self.previous_loss = loss
        else:
            change = loss - self.previous_loss
            self.change_sum += change
            self.num_steps += 1

            if self.num_steps < self.ignore_first_n:
                return

            avg_change = self.change_sum / self.num_steps 

            if avg_change < change or avg_change > -1e-5:
                # current improvement (reduction of loss) was less strong than the average improvement => we are passed the inflection point

                if change > -self.min_improvement:
                    self.patience_counter += 1

                    if self.patience_counter >= self.patience:
                        self.stopped_early = True
                        self.model.stop_training = True
                else:
                    self.patience_counter = 0


class InflectionPointLRReducer(tf.keras.callbacks.Callback):
    min_improvement : float
    patience : int
    factor : float

    change_sum : float = 0
    num_steps : int = 0
    previous_loss : float = None
    patience_counter : int = 0

    def __init__(self, min_improvement : float, patience : int, factor : float, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.min_improvement = min_improvement
        self.patience = patience
        self.factor = factor

    def on_train_batch_end(self, batch, logs):
        loss = logs['loss'].numpy()
        if self.previous_loss is None:
            self.previous_loss = loss
        else:
            change = loss - self.previous_loss
            self.change_sum += change
            self.num_steps += 1

            if self.num_steps < self.ignore_first_n:
                return

            avg_change = self.change_sum / self.num_steps 

            if avg_change < change or avg_change > -1e-5:
                # current improvement (reduction of loss) was less strong than the average improvement => we are passed the inflection point

                if change > -self.min_improvement:
                    self.patience_counter += 1

                    if self.patience_counter >= self.patience:
                        self.model.optimizer.learning_rate *= self.factor
                        self.change_sum = 0
                        self.num_steps = 0 
                        self.patience_counter = 0
                else:
                    self.patience_counter = 0


class LearningRateWarmupPerStep(tf.keras.callbacks.Callback):
    num_steps : int
    start_lr : float
    end_lr : float

    steps_counter : int = 0

    def __main__(self, num_steps : int, start_lr : float, end_lr : float, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.num_steps = num_steps
        self.start_lr = start_lr
        self.end_lr = end_lr

    def on_train_batch_begin(self):
        lr = self.model.optimizer.learning_rate
        if self.steps_counter < self.num_steps:
            factor = self.steps_counter / self.num_steps
            lr = factor * self.end_lr + (1-factor) * self.start_lr
            self.model.optimizer.learning_rate = lr