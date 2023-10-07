import pickle
import os
import numpy as np
import tensorflow as tf
from scipy.stats import ks_2samp


from dlomix.constants import QR_pipeline_parameters
from dlomix.data.RetentionTimeDataset import RetentionTimeDataset
from dlomix.models import PrositRetentionTimePredictor
from dlomix.losses import QuantileLoss
from dlomix.reports import UncertaintyReport

from dlomix.eval.interval_conformal import IntervalSize, AbsoluteIntervalSize, RelativeCentralDistance, \
                                           IntervalConformalScore, IntervalConformalQuantile


class QRpipeline():

    def __init__(self, frozen, alpha=0.1):
        self.model = None
        self.train_val_dataset = None
        self.test_dataset = None
        self.init_predictions = None
        self.qr_predictions = None

        self.frozen = frozen
        self.alpha = alpha
        self.label = "-" + str(frozen) + " layer(s)" if frozen else "complete"

        self._load_model()

    def _load_model(self):
        self.model = PrositRetentionTimePredictor(**QR_pipeline_parameters["model_params"])
        self.model.load_weights(QR_pipeline_parameters["base_model_path"]).expect_partial()

    def load_data(self, 
                  train_val_source=QR_pipeline_parameters["train_val_path"],
                  test_source=QR_pipeline_parameters["test_path"],
                  val_ratio=0.2, batchsize=32):
        
        if not (isinstance(train_val_source, str) or isinstance(train_val_source, np.ndarray)):
            raise ValueError(
                "train_val dataset should be provided either as a numpy array or a string pointing to a file."
            )
        if not (isinstance(test_source, str) or isinstance(test_source, np.ndarray)):
            raise ValueError(
                "train_val dataset should be provided either as a numpy array or a string pointing to a file."
            )
        
        self.train_val_dataset = RetentionTimeDataset(
            data_source=train_val_source,
            **QR_pipeline_parameters["data_params"], 
            batch_size=batchsize,
            val_ratio=val_ratio,
            test=False)
        
        self.test_dataset = RetentionTimeDataset(
            data_source=test_source,
            **QR_pipeline_parameters["data_params"], 
            batch_size=batchsize, 
            test=True)

    def predict(self, qr_prediction):

        predictions = self.model.predict(self.test_dataset.test_data)
        #predictions = self.test_dataset.denormalize_targets(predictions)

        if qr_prediction:
            self.qr_predictions = predictions
        else:
            predictions = predictions.ravel()
            self.init_predictions = predictions

    def adapt_model_for_qr(self):

        if not self.frozen:
            #change weights to random since whole model is unfrozen
            self.model = PrositRetentionTimePredictor(**QR_pipeline_parameters["model_params"])
        else:
            #"freeze" all layers up until the specified val
            for layer in self.model.layers[:-self.frozen]:
                layer.trainable = False
        self.model.output_layer = tf.keras.Sequential([tf.keras.layers.Dense(2)])      

    def train_model_on_ql(self, epochs=10, batch_size=32, checkpoint_dir='checkpoints'):
        
        train_steps = batch_size * len(self.train_val_dataset.train_data) * epochs
        lr_fn = tf.optimizers.schedules.PolynomialDecay(1e-3, train_steps, 1e-6, 2)
        opt = tf.optimizers.Adam(lr_fn)
        
        self.model.compile(optimizer=opt, 
                           loss=QuantileLoss(tf.constant([[self.alpha, 1-self.alpha]])),
                           metrics=['mean_absolute_error', 
                                    RelativeCentralDistance(),
                                    IntervalConformalQuantile(alpha=self.alpha),
                                    AbsoluteIntervalSize()])
        #self.model.build(input_shape=(None, 30))

        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(checkpoint_dir, self.label, 'best.hdf5'),
                                                                       save_weights_only=True,
                                                                       monitor='val_loss',
                                                                       mode='min',
                                                                       save_best_only=True)
        
        history = self.model.fit(self.train_val_dataset.train_data,
                            validation_data=self.train_val_dataset.val_data,
                            callbacks=[model_checkpoint_callback],
                            epochs=epochs)
        #return history


    def report(self):

        test_targets = self.test_dataset.get_split_targets(split="test")
        
        conf_scores = IntervalConformalScore(reduction='none')(np.expand_dims(test_targets,-1), self.qr_predictions).numpy()
        conf_quantile = IntervalConformalQuantile(alpha=0.1, reduction='none')(np.expand_dims(test_targets,-1), self.qr_predictions).numpy()
        
        intervals = self.qr_predictions.copy()
        intervals[:,0] -= conf_quantile
        intervals[:,1] += conf_quantile
        interval_sizes = intervals[:,1] - intervals[:,0]
        within = (test_targets >= intervals[:,0]) & (test_targets <= intervals[:,1])

        UncertaintyReport.plot_conformal_scores(conf_scores, quantile=conf_quantile)
        UncertaintyReport.plot_predictions_with_intervals(test_targets, self.init_predictions, intervals)
        UncertaintyReport.plot_conformalized_interval_size(interval_sizes)

        pvalue = ks_2samp(interval_sizes[within], interval_sizes[~within]).pvalue # prob. for Null: distr are identical
        print('####', self.label, '####\nconformal quantile:', conf_quantile, '\n')

        UncertaintyReport.plot_conformalized_interval_size_PDFs(interval_sizes, within, pvalue)


    def save_data(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.qr_predictions, f)

    def run_full_pipeline(self, epochs=10, save=True, save_path='QR_preds.pkl'):

        self.load_data()
        self.predict(qr_prediction=False)
        self.adapt_model_for_qr()
        self.train_model_on_ql(epochs=epochs)
        self.predict(qr_prediction=True)
        self.report()
        if save:
            self.save_data(path=save_path)