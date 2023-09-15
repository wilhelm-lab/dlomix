from dlomix.models import PrecursorChargeStatePredictor
from dlomix.data import PrecursorChargeStateDataset

'''
Give path to folder. Dataset-Class parses files if file_types [*.parquet, *.tsv, *.csv].
'''
DATAPATH = 'C:/Users/micro/OneDrive/Dokumente/GitHub/Masterpraktikum/data/'

'''
classification_type available: "multi_class", "multi_label"
model_type available: 
- multi_class: "embedding", "conv2d", "prosit"
- multi_label: "multilabel", "multihead"
'''


pretrained_version = True # TODO Set whether to use pretrained model or not
if pretrained_version:

    model_path = '../pretrained_models/precursor_charge_state/example_multiclass_embedding/embed_150epoch_09152023.h5'
    prospect_pretrained = PrecursorChargeStatePredictor(pretrained_model=model_path, sequence="EM[UNIMOD:35]LTRAIKTQLVLLT")

else:
    prospect_dataset = PrecursorChargeStateDataset(classification_type="multi_class", model_type="embedding",
                                                   charge_states=[1, 2, 3, 4, 5, 6], dir_path=DATAPATH,
                                                   columns_to_keep=['modified_sequence', 'precursor_charge',
                                                                    'precursor_intensity'])

    model_class = PrecursorChargeStatePredictor(prospect_dataset)
    model_class.summary()
    model_class.compile()
    # model_class.wandb_init(api_key="YOUR_API_KEY", project_name="YOUR_PROJECT_NAME")
    model_class.fit(epochs=30, no_wandb=True)  # TODO set no_wandb to False if model_class.wand_init() is called
    model_class.evaluate()
    model_class.predict()  # Predict offers verification if given labels. Default is Prediction on Testdata.
