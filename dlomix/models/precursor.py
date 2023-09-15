import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Dense, Embedding, Flatten, Dropout, Bidirectional, GRU, Conv2D, Lambda
from tensorflow.keras.models import Model
from dlomix.layers.attention import AttentionLayer, DecoderAttentionLayer
import subprocess
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, \
    f1_score, ConfusionMatrixDisplay
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import wandb
from wandb.keras import WandbCallback
import re


class PrecursorChargeStatePredictor:

    def __init__(self, dataset=None, pretrained_model=None, sequence=None):
        self.model = None
        self.sequence = None
        self.shape = None
        self.classification_type = None
        self.model_type = None
        self.max_len_seq = None
        self.voc_len = None
        self.num_classes = None
        self.dataset = None
        self.history = None
        self.loss = None
        self.metrics = None
        self.evaluation = None
        self.evaluated = None
        self.prediction = None
        self.predicted = None
        self.skip_training = None
        self.pretrained_model = None
        self.wandb = False
        self.compiled = False
        self.fitted = False
        self.pretrained = False
        if dataset is not None:
            self.initiate_with_dataset(dataset)
        elif pretrained_model is not None and sequence is not None:
            self.initiate_with_pretrained_model(pretrained_model, sequence)
        else:
            raise ValueError(
                "PrecursorChargeStatePredictor must be initiated with either a dataset or a pretrained model.")

    def initiate_with_dataset(self, dataset):
        self.pretrained_model = pretrained_model
        self.skip_training = skip_training
        self.dataset = dataset
        self.num_classes = dataset.num_classes
        self.voc_len = dataset.voc_len
        self.max_len_seq = dataset.padding_length
        self.model_type = dataset.model_type
        self.classification_type = dataset.classification_type
        self.shape = dataset.train_data[0].shape

        if self.model_type == "embedding":
            self.model = self.embedding_model()
        elif self.model_type == "conv2d":
            self.model = self.conv2d_model()
        elif self.model_type == "prosit":
            self.model = self.prosit_model()
        elif self.model_type == "multihead":
            self.model = self.multihead_model()
        elif self.model_type == "multilabel":
            self.model = self.multilabel_model()
        else:
            raise ValueError(
                "model_type must be one of the following: 'embedding', 'conv2d', 'prosit', 'multihead', 'multilabel'")

    def prosit_model(self):
        input_prosit = Input(shape=self.shape)
        x = Model(inputs=input_prosit, outputs=input_prosit)
        # Embedding, no vocabulary
        y = Embedding(input_dim=self.voc_len, output_dim=self.max_len_seq, input_length=self.max_len_seq)(input_prosit)
        # Encoder
        y = Bidirectional(GRU(256, return_sequences=True))(y)
        y = Dropout(0.5)(y)
        y = GRU(512, return_sequences=True)(y)
        y = Dropout(0.5)(y)
        # Attention
        y = AttentionLayer(y)
        # Regressor
        y = Dense(512, activation="relu")(y)
        y = Dropout(0.1)(y)
        # Output
        out = Dense(self.num_classes, activation="softmax")(y)
        model_prosit = Model(inputs=[x.input], outputs=out)
        return model_prosit

    def conv2d_model(self):
        input_convolution = Input(shape=self.shape)
        x = Model(inputs=input_convolution, outputs=input_convolution)
        y = Rescaling(scale=1. / 100)(input_convolution)
        y = Conv2D(filters=128, kernel_size=(1, 3), strides=1, activation="relu", padding='same')(y)
        y = Flatten()(y)
        y = Dense(210, activation="relu")(y)
        z = Dense(self.num_classes, activation="softmax")(y)
        model_convolution = Model(inputs=[x.input], outputs=z)
        return model_convolution

    def embedding_model(self):
        input_embedding = Input(shape=self.shape)
        # the first branch operates on the first input
        x = Model(inputs=input_embedding, outputs=input_embedding)
        y = Embedding(input_dim=self.voc_len, output_dim=self.max_len_seq, input_length=self.max_len_seq)(
            input_embedding)
        y = Flatten()(y)
        y = Dense(self.max_len_seq, activation="relu")(y)
        z = Dense(self.num_classes, activation="softmax")(y)
        model_embed = Model(inputs=[x.input], outputs=z)
        return model_embed

    def multihead_model(self):
        input_multihead = Input(shape=self.shape)
        x = Model(inputs=input_multihead, outputs=input_multihead)
        y = Embedding(input_dim=self.voc_len, output_dim=self.max_len_seq, input_length=self.max_len_seq)(
            input_multihead)
        y = Flatten()(y)
        branch_outputs = []
        for i in range(6):
            out = Lambda(lambda a: a[:, i:i + 1])(y)
            out = Dense(2, activation="sigmoid")(out)
            branch_outputs.append(out)
        model_multihead = Model(inputs=[x.input], outputs=branch_outputs)
        return model_multihead

    def multilabel_model(self):
        input_multilabel = Input(shape=self.shape)
        x = Model(inputs=input_multilabel, outputs=input_multilabel)
        y = Embedding(input_dim=self.voc_len, output_dim=self.max_len_seq, input_length=self.max_len_seq)(
            input_multilabel)
        y = Flatten()(y)
        y = Dense(self.max_len_seq, activation="relu")(y)
        z = Dense(self.num_classes, activation="sigmoid")(y)
        model_multilabel = Model(inputs=[x.input], outputs=z)
        return model_multilabel

    def summary(self):
        self.model.summary()

    def wandb_init(self, api_key=None,
                   project_name="precursor-charge-state-prediction"):
        if api_key is None:
            raise ValueError("No API key provided. Use model_class.wandb_init(api_key= '...', project_name = '...')")
        subprocess.call(['wandb', 'login', api_key])
        wandb.init(project=project_name)
        config = wandb.config
        config.model_type = self.model_type
        config.classification_type = self.classification_type
        config.num_classes = self.num_classes
        config.voc_len = self.voc_len
        config.max_len_seq = self.max_len_seq
        self.wandb = True

    def compile(self, lr=0.0001):
        if self.classification_type == "multi_class":
            if self.model == "prosit":
                self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
                self.loss = 'categorical_crossentropy'
                self.metrics = 'categorical_accuracy'
            else:
                self.model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr), loss='categorical_crossentropy',
                                   metrics=['categorical_accuracy'])
                self.loss = 'categorical_crossentropy'
                self.metrics = 'categorical_accuracy'

        elif self.classification_type == "multi_label":
            self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['binary_accuracy'])
            self.loss = 'binary_crossentropy'
            self.metrics = 'binary_accuracy'
        else:
            raise ValueError("classification_type must be one of the following: 'multi_class', 'multi_label'")
        self.compiled = True

    def fit(self, batch_size=4096, callbacks=None, epochs=30, no_wandb=False):
        if not self.compiled:
            raise ValueError("Model must be compiled before fitting. Use model_class.compile().")
        elif not self.wandb and not no_wandb:
            raise ValueError(
                "You did not initialize weights&biases. Set model_class.init(no_wandb=True) or use "
                "model_class.wandb_init(api_key= '...', project_name = '...')")
        else:
            if callbacks is None:
                if no_wandb:
                    callbacks = []
                else:
                    callbacks = [WandbCallback()]
            # print(self.shape) print(callbacks, len(self.dataset.train_data), len(self.dataset.train_label),
            # len(self.dataset.val_data), len(self.dataset.val_label))
            self.history = self.model.fit(self.dataset.train_data, self.dataset.train_label, epochs=epochs,
                                          batch_size=batch_size,
                                          validation_data=(self.dataset.val_data, self.dataset.val_label),
                                          callbacks=callbacks, verbose=1)

            self.fitted = True

    def save(self, output_path=None):
        if output_path is None:
            output_path = f"{self.model_type}_model.h5"
        else:
            if not output_path.endswith(".h5"):
                output_path = f"{output_path}.h5"
        self.model.save(output_path)

    def plot_training(self):
        if self.fitted:
            # Access the loss, validation loss, and accuracy from the history object
            loss = self.history['loss']
            val_loss = self.history['val_loss']
            accuracy = self.history[self.metrics]
            val_accuracy = history.history[self.loss]

            # Plot the loss, validation loss, and accuracy curves
            epochs = range(1, len(loss) + 1)

            # Create subplots
            fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

            # Plot loss and validation loss
            ax1.plot(epochs, loss, 'b', label='Training Loss')
            ax1.plot(epochs, val_loss, 'r', label='Validation Loss')
            ax1.set_title('Training and Validation Loss')
            ax1.set_xlabel('Epochs')
            ax1.set_ylabel('Loss')
            ax1.legend()

            # Plot accuracy and validation accuracy
            ax2.plot(epochs, accuracy, 'b', label='Training Accuracy')
            ax2.plot(epochs, val_accuracy, 'r', label='Validation Accuracy')
            ax2.set_title('Training and Validation Accuracy')
            ax2.set_xlabel('Epochs')
            ax2.set_ylabel('Accuracy')
            ax2.legend()

            # Adjust spacing between subplots
            plt.tight_layout()

            # Show the plots
            plt.show()
        else:
            raise ValueError("Model was not trained. No data to plot. Use model_class.fit()")

    def load_weights(self, path):
        self.model = tf.keras.saving.load_model(path)
        self.pretrained = True

    def evaluate(self, test_data=None, test_label=None, test_mode=False):
        if not self.fitted:
            if not test_mode:
                if test_data is None or test_label is None:
                    raise ValueError(
                        "You did not provide test_data and test_label. Use model_class.evaluate(test_data, "
                        "test_label) or set apply test_ratio>0 to model_class")
                else:
                    self.evaluation = self.model.evaluate(test_data, test_label)
                    self.evaluated = True
            else:
                self.evaluation = self.model.evaluate(self.dataset.test_data, self.dataset.test_label)
                self.evaluated = True
        else:
            if self.pretrained:
                if test_data is None or test_label is None:
                    raise ValueError(
                        "You did not provide test_data and test_label. Use model_class.evaluate(test_data, "
                        "test_label) or set apply test_ratio>0 to model_class")
                else:
                    self.evaluation = self.model.evaluate(test_data, test_label)
                    self.evaluated = True
            else:
                self.evaluation = self.model.evaluate(self.dataset.test_data, self.dataset.test_label)
                self.evaluated = True

        print(f"test loss, test acc: {self.evaluation}")

    def predict(self, test_data=None, test_label=None, test_mode=False, no_verification=False):
        if not self.fitted:
            if not test_mode:
                if test_data is None:
                    raise ValueError(
                        "You did not provide test_data and test_label. Use model_class.evaluate(test_data, "
                        "test_label) or set apply test_ratio>0 to model_class")
                else:
                    self.prediction = self.model.predict(test_data)
                    self.predicted = True
            else:
                self.prediction = self.model.predict(test_data)
                self.predicted = True
        else:
            if self.pretrained:
                if test_data is None:
                    raise ValueError(
                        "You did not provide test_data and test_label. Use model_class.predict(test_data, test_label) "
                        "or set apply test_ratio>0 to model_class")
                else:
                    self.prediction = self.model.predict(test_data)
                    self.predicted = True
            else:
                test_data = self.dataset.test_data
                test_label = self.dataset.test_label
                self.prediction = self.model.predict(test_data)
                self.predicted = True

        if not no_verification:
            if self.classification_type == "multi_class":
                if test_label is None:
                    raise ValueError("You did not provide test_label for prediction-verification.")
                else:
                    predicted_labels = np.argmax(self.prediction, axis=1)
                    true_labels = np.argmax(test_label, axis=1)

                    cm = confusion_matrix(true_labels, predicted_labels)
                    print(cm)
                    print("Accuracy: ", accuracy_score(true_labels, predicted_labels))
                    print("Precision_weighted: ", precision_score(true_labels, predicted_labels, average='weighted'))
                    print("Recall_weighted: ", recall_score(true_labels, predicted_labels, average='weighted'))
                    print("F1_weighted: ", f1_score(true_labels, predicted_labels, average='weighted'))
                    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=self.dataset.charge_states)
                    disp.plot(cmap=plt.cm.Blues, xticks_rotation=45)
                    # add legend title and axis labels
                    plt.xlabel('Predicted Label')
                    plt.ylabel('True Label')
                    plt.title('Confusion Matrix')
                    # plt.colorbar(label="Number of Samples")
                    plt.show()

                    new_df = pd.DataFrame()
                    new_df['charge'] = [1, 2, 3, 4, 5, 6]
                    new_df['precision'] = precision_score(true_labels, predicted_labels, average=None)
                    new_df['recall'] = recall_score(true_labels, predicted_labels, average=None)
                    new_df['f1'] = f1_score(true_labels, predicted_labels, average=None)
                    print(new_df)

            else:
                raise ValueError("Not implemented for multi-class.")

    def initiate_with_pretrained_model(self, pretrained_model=None, sequence=None):

        # raise error if sequence longer than 63
        if len(sequence) > 63:
            raise ValueError("Sequence must be shorter than 63 amino acids in our pretrained model.")

        self.pretrained_model = pretrained_model
        self.sequence = sequence
        self.model = keras.models.load_model(self.pretrained_model)
        self.prediction = None

        def pretrained_seq_translator(sequence, print_result=False, no_padding=False):
            """
            Translates a sequence into a vector of integers
            :param print_result:
            :param max_len:
            :param sequence: string
            :param dictionary: dictionary
            :return: list
            """
            pattern = r'[A-Z]\[[^\]]*\]|.'  # regex pattern to match amino acids and modifications
            # pattern = r'(\w\[UNIMOD:\d+\])' # regex pattern to match amino acids and modifications

            dictionary = ['X', 'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V',
                          'W', 'Y', 'C[UNIMOD:4]', 'M[UNIMOD:35]']
            max_len = 63

            result = [match for match in re.findall(pattern, sequence)]

            # Fill the list with "X" characters until it reaches a length of 40
            if not no_padding:
                result += ['X'] * (max_len - len(result))
            if print_result:
                print(result)

            aa_dictionary = dict()
            for index, aa in enumerate(dictionary):
                aa_dictionary[aa] = index

            return [aa_dictionary[aa] for aa in result]

        def generate_charge_prediction_text(charge_predictions, input_sequence):
            max_charge_index = np.argmax(charge_predictions)
            max_charge_value = round(charge_predictions[max_charge_index], 2)

            charge_text = f"The predicted charge state for the input sequence '{input_sequence}' is {max_charge_index + 1} [{round(max_charge_value * 100, 2)}%]."
            percentage_text = "Prediction percentages for all states:\n"

            for index, prediction in enumerate(charge_predictions):
                if index != max_charge_index:
                    percentage = round(prediction * 100, 2)
                    percentage_text += f"Charge state {index + 1}: {percentage}%\n"
                else:
                    percentage = round(prediction * 100, 2)
                    percentage_text += f"--Charge state {index + 1}: {percentage}%\n"

            full_text = charge_text + "\n" + percentage_text
            return full_text

        def predictor(self, sequence):
            print("Sequence: ", sequence)
            encoded_sequence = pretrained_seq_translator(sequence)
            encoded_sequence = np.expand_dims(tf.convert_to_tensor(np.array(encoded_sequence)), axis=0)
            sequence_prediction = self.model.predict(encoded_sequence, verbose=False)
            print("Weights_per_Charge_State: ", [round(x, 2) for x in sequence_prediction[0]], "Sum: ",
                  sum([round(x, 2) for x in sequence_prediction[0]]))
            print("-----------------------------")
            print(generate_charge_prediction_text(sequence_prediction[0], sequence))
            return sequence_prediction

        if isinstance(self.sequence, str):
            prediction = predictor(self, self.sequence)
            self.prediction = prediction

        elif isinstance(self.sequence, list):
            prediction = []
            for seq in self.sequence:
                prediction.append(predictor(self, seq))
            self.prediction = prediction

        return self.prediction
