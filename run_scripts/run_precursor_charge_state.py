from dlomix.models import PrecursorChargeStatePredictor
from dlomix.data import PrecursorChargeStateDataset

'''
Give path to folder. Dataset-Class parses files if file_types [*.parquet, *.tsv, *.csv].
'''
DATAPATH = 'C:/Users/micro/OneDrive/Dokumente/GitHub/Masterpraktikum/data/'

'''
classification_type available: "multi_class", "multi_label"
model_type available: 
- multi_class (embedding)
- multi_label (embedding)
'''


pretrained_version = False # TODO Set whether to use pretrained model or not
if pretrained_version:

    # try example_multilabel_embedding/embed_30epoch_multilabel.h5 for pretrained multi_label model
    model_path = '../pretrained_models/precursor_charge_state/example_multiclass_embedding/embed_30epoch_multiclass.h5'
    prospect_pretrained = PrecursorChargeStatePredictor(pretrained_model=model_path, sequence="EM[UNIMOD:35]LTRAIKTQLVLLT")

else:
    prospect_dataset = PrecursorChargeStateDataset(classification_type="multi_class",
                                                   charge_states=[1,2,3,4,5,6], dir_path=DATAPATH,
                                                   columns_to_keep=['modified_sequence', 'precursor_charge',
                                                                    'precursor_intensity'])

    model_class = PrecursorChargeStatePredictor(
        classification_type=prospect_dataset.classification_type,
        charge_states=prospect_dataset.charge_states,
        voc_len=prospect_dataset.voc_len,
        padding_length=prospect_dataset.padding_length,
    )
    model_class.summary()
    model_class.compile()
    # model_class.wandb_init(api_key="YOUR_API_KEY", project_name="YOUR_PROJECT_NAME")
    model_class.fit(epochs=30, no_wandb=True, # TODO set no_wandb to False if model_class.wand_init() is called
                    training_data=prospect_dataset.train_data,
                    training_label=prospect_dataset.train_label,
                    validation_data=prospect_dataset.val_data,
                    validation_label=prospect_dataset.val_label,
                    batch_size=4096)
    model_class.evaluate(
        test_data=prospect_dataset.test_data,
        test_label=prospect_dataset.test_label
    )
    model_class.predict(
        test_data=prospect_dataset.test_data,
        test_label=prospect_dataset.test_label,
        classification_type=prospect_dataset.classification_type,
        charge_states=prospect_dataset.charge_states
    )
