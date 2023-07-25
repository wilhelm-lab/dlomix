import pickle
import numpy as np
import tensorflow as tf
from scipy.stats import ks_2samp

from dlomix.constants import MCDO_pipeline_parameters
from dlomix.data.RetentionTimeDataset import RetentionTimeDataset
from dlomix.models import PrositRetentionTimePredictor
from dlomix.eval.scalar_conformal import ScalarConformalScore, ScalarConformalQuantile
from dlomix.losses import QuantileLoss
from dlomix.reports import MonteCarloReport



class MCDOPipeline:
    def __init__(self, alpha=0.1):
        self.base_model = None
        self.test_dataset = None
        self.alpha = alpha
        self.res = None
        self.label = None
        
        self._build_base_model()

    def _build_base_model(self):

        self.base_model = PrositRetentionTimePredictor(**MCDO_pipeline_parameters["model_params"])
        self.base_model.load_weights(MCDO_pipeline_parameters["base_model_path"]).expect_partial()

    def _predict_with_dropout(self, n):
        
        predictions = []
        for i in range(n):
            res = np.concatenate([self.base_model(batch[0], training=True).numpy() for batch in list(self.test_dataset.test_data)])
            predictions.append(res)
        return np.column_stack(predictions)
    
    def load_data(self, data=MCDO_pipeline_parameters["test_set_path"], batchsize=32):
        
        if not (isinstance(data, str) or isinstance(data, np.ndarray)):
            raise ValueError(
                "Dataset should be provided either as a numpy array or a string pointing to a file."
            )
        
        self.test_dataset = RetentionTimeDataset(
            data_source=data,
            **MCDO_pipeline_parameters["data_params"], 
            batch_size=batchsize, 
            test=True)

    def predict(self, n=3):
        
        res = {}
        self.label = f"PROSIT_MCDO_n={n}"
        res[self.label] = {}
        print("model :", self.label, ", n =", n)
        pred = self._predict_with_dropout(n=n)
        res[self.label]['data'] = np.array((pred.mean(axis=1), pred.std(axis=1)))
        self.res = res
        
        return res

    def report(self):
        
        test_targets = self.test_dataset.get_split_targets(split="test")
        avgs, stds = self.res[self.label]['data'][0], self.res[self.label]['data'][1]
        print(f'#### {self.label} ####')

        conf_scores = ScalarConformalScore(reduction='none')(test_targets, self.res[self.label]['data'].T).numpy()
        conf_quantile = ScalarConformalQuantile()(test_targets, self.res[self.label]['data'].T).numpy()

        print(f"alpha = {self.alpha}, conformal quantile: {conf_quantile:.2f}")

        intervals = np.array([avgs - stds * conf_quantile, avgs + stds * conf_quantile]).T
        interval_sizes = intervals[:,1] - intervals[:,0]
        within = (test_targets >= intervals[:,0]) & (test_targets <= intervals[:,1])

        MonteCarloReport.plot_conformal_scores(conf_scores, quantile=conf_quantile)
        MonteCarloReport.plot_predictions_with_intervals(test_targets, avgs, intervals)
        MonteCarloReport.plot_conformalized_interval_size(interval_sizes)

        pvalue = ks_2samp(interval_sizes[within], interval_sizes[~within]).pvalue # prob. for Null: distr are identical
        print(f"p = {pvalue:.5f} : {'Reject' if pvalue < 0.01 else 'Accept'} Null Hypothesis (Distr. identical)")

        MonteCarloReport.plot_conformalized_interval_size_PDFs(interval_sizes, within, pvalue)
    
    def save_results(self, path="MCDO_results.pkl"):

        with open(path, 'wb') as f:
            pickle.dump(self.res, f)


            

