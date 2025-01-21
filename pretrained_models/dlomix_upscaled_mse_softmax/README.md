Trained model created by Noah Nussbaumer, building on work of Florian Boessl.
Copied from `/cmnfs/data/proteomics/origin/nussbaumer_datasets/models/dlomix/improvement/dlomix_upscaled_mse_softmax/trained_model.keras`

* close to original (no hyperparam tuning)
* uses softmax
* uses a scaled MSE (scaled to value range 0 - 100)
* turned out to be the best of the ones tried (but no big differences)

loading (custom objects are provided in this repo):
´´´
model = keras.saving.load_model(
    filepath,
    custom_objects={
        'upscaled_mean_squared_error': upscaled_mean_squared_error,
        'euclidean_similarity': euclidean_similarity, 
        'masked_pearson_correlation_distance': masked_pearson_correlation_distance, 
        'masked_spectral_distance': masked_spectral_distance, 
        'ChargeStateDistributionPredictor': ChargeStateDistributionPredictor
    }
)
´´´