import numpy as np

from constants import retention_time_pipeline_parameters
from data import RTDataset
from models import *


class RetentionTimePipeline:
    def __init__(self):
        super(RetentionTimePipeline, self).__init__()
        self.model = None
        self.dataset = None

        self._build_model()

    def _build_model(self):
        self.model = RTPredictor(**retention_time_pipeline_parameters['model_params'])
        self.model.load_weights(retention_time_pipeline_parameters['trained_model_path'])

    '''
    
    Predict retention times given data either as numpy array of sequences or a filepath to a csv file
    
    '''

    def predict(self, data=None):
        if not (isinstance(data, str) or isinstance(data, np.ndarray)):
            raise ValueError("Dataset should be provided either as a numpy array or a string pointing to a file.")

        self.dataset = RTDataset(data, **retention_time_pipeline_parameters['data_params'], val_ratio=0, test=True)
        RTDataset.TARGETS_MEAN, RTDataset.TARGETS_STD = retention_time_pipeline_parameters['trained_model_stats']

        predictions = self.model.predict(self.dataset.get_tf_dataset('test'))

        predictions = self.dataset.denormalize_targets(predictions)

        # here more reporting of results using the RTReport class

        return predictions
