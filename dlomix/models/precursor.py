import re
import subprocess
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import wandb
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    accuracy_score, precision_score, recall_score, f1_score

)
from tensorflow import keras
from tensorflow.keras.layers import (
    Input, Dense, Embedding, Flatten
)
from tensorflow.keras.models import Model
from wandb.keras import WandbCallback


class PrecursorChargeStatePredictor(tf.keras.Model):
    def __init__(self,
                 # if retraining
                 classification_type="multi_class",
                 padding_length=63,
                 voc_len=23,
                 charge_states=None,

                 # if pretrained
                 pretrained_model=None,
                 sequence=None):
        super(PrecursorChargeStatePredictor, self).__init__()
        if charge_states is None:
            charge_states = [1, 2, 3, 4, 5, 6]
        self.model = None
        self.sequence = None
        self.shape = np.array([padding_length, ])
        self.classification_type = classification_type
        self.max_len_seq = padding_length
        self.voc_len = voc_len
        self.charge_states = charge_states
        self.num_classes = len(charge_states)
        self.history = None
        self.loss = None
        self.in_metrics = None
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

        if pretrained_model is not None and sequence is not None:
            self.initiate_with_pretrained_model(pretrained_model, sequence)

        if self.classification_type == "multi_class":
            self.model = self.multiclass_model()
        elif self.classification_type == "multi_label":
            self.model = self.multilabel_model()
            if len(charge_states) <= 1:
                raise ValueError(f"For multiclass classification, more than one charge state is required. "
                                 f"You've entered: {charge_states} ")

    def multiclass_model(self):
        """
        Multiclass model with embedding layer
        :return: model
        """
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

    def multilabel_model(self):
        """
        Multilabel model with embedding layer
        """
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
        """
        Prints a summary of the model
        Overwrites Keras' summary method - Otherwise "not built, call 'build()' error
        """
        self.model.summary()

    def wandb_init(self, api_key=None,
                   project_name="precursor-charge-state-prediction"):
        """
        Initializes weights&biases
        @param api_key: str
        @param project_name: str
        """
        if api_key is None:
            raise ValueError(f"No API key provided. Use model_class.wandb_init(api_key= '...', project_name = '...')."
                             f" You entered: {api_key}, {project_name}")
        subprocess.call(['wandb', 'login', api_key])
        wandb.init(project=project_name)
        config = wandb.config
        config.classification_type = self.classification_type
        config.num_classes = self.num_classes
        config.voc_len = self.voc_len
        config.max_len_seq = self.max_len_seq
        self.wandb = True

    def compile(self, lr=0.0001):
        """
        Compiles the model
        Overwrites Keras' compile method, setting defaults according to the classification type
        @param lr: float
        """
        if self.classification_type == "multi_class":
            self.model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr), loss='categorical_crossentropy',
                               metrics=['categorical_accuracy'])
            self.loss = 'categorical_crossentropy'
            self.in_metrics = 'categorical_accuracy'

        elif self.classification_type == "multi_label":
            self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['binary_accuracy'])
            self.loss = 'binary_crossentropy'
            self.in_metrics = 'binary_accuracy'
        else:
            raise ValueError("classification_type must be one of the following: 'multi_class', 'multi_label'")
        self.compiled = True

    def fit(self, batch_size=4096, callbacks=None, epochs=30, training_label=None, training_data=None,
            validation_label=None, validation_data=None, no_wandb=False):
        """
        Fits the model
        Overwrites Keras' fit method, defining input parameters according to expected dataset input
        @param batch_size: int
        @param callbacks: list
        @param epochs: int
        @param training_label: list
        @param training_data: list
        @param validation_label: list
        @param validation_data: list
        @param no_wandb: bool
        """
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
            self.history = self.model.fit(training_data, training_label, epochs=epochs,
                                          batch_size=batch_size,
                                          validation_data=(validation_data, validation_label),
                                          callbacks=callbacks, verbose=1)

            self.fitted = True

    def save(self, output_path=None):
        """
        Saves the model
        Overwrites Keras' save method
        """
        if output_path is None:
            output_path = f"{self.classification_type}_model.h5"
        else:
            if not output_path.endswith(".h5"):
                output_path = f"{output_path}.h5"
        self.model.save(output_path)

    def plot_training(self):
        """
        Plots the training curves (Training and Validation Loss, Training and Validation Accuracy)
        Basic Matplotlib plot
        """
        if self.fitted:
            # Access the loss, validation loss, and accuracy from the history object
            loss = self.history['loss']
            val_loss = self.history['val_loss']
            accuracy = self.history[self.in_metrics]
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
        """
        Loads weights from a given path
        Overwrites Keras' load_weights method
        """
        self.model = tf.keras.saving.load_model(path)
        self.pretrained = True

    def evaluate(self, test_data=None, test_label=None):
        """
        Evaluates the model
        Overwrites Keras' evaluate method, defining input parameters according to expected dataset input
        @param test_data: list
        @param test_label: list
        """
        if test_data is None or test_label is None:
            raise ValueError(
                "You did not provide test_data and test_label. Use model_class.evaluate(test_data, "
                "test_label) or set apply test_ratio>0 to model_class")
        else:
            self.evaluation = self.model.evaluate(test_data, test_label)
            self.evaluated = True

        print(f"test loss, test acc: {self.evaluation}")

    def predict(self, test_data=None, test_label=None, verification=True, classification_type=None,
                charge_states=None):
        """
        Predicts the labels for a given test dataset, optionally verifying the prediction on the given labels
        Overwrites Keras' predict method, defining input parameters according to expected dataset input
        Outputs a confusion matrix and a dataframe with precision, recall and f1-score for each class label
        @param test_data: list
        @param test_label: list
        @param verification: bool
        @param classification_type: str
        @param charge_states: list
        """
        if test_data is None:
            raise ValueError(
                "You did not provide test_data and test_label. Use model_class.evaluate(test_data, "
                "test_label) or set apply test_ratio>0 to model_class")
        else:
            self.prediction = self.model.predict(test_data)
            self.predicted = True

        if test_label is None and verification:
            raise ValueError("You did not provide test_label for prediction-verification.")

        else:
            if classification_type == "multi_class":
                predicted_labels = np.argmax(self.prediction, axis=1)
                true_labels = np.argmax(test_label, axis=1)

                cm = confusion_matrix(true_labels, predicted_labels)
                print(cm)
                print("Accuracy: ", accuracy_score(true_labels, predicted_labels))
                print("Precision_weighted: ", precision_score(true_labels, predicted_labels, average='weighted'))
                print("Recall_weighted: ", recall_score(true_labels, predicted_labels, average='weighted'))
                print("F1_weighted: ", f1_score(true_labels, predicted_labels, average='weighted'))
                disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=charge_states)
                disp.plot(cmap=plt.cm.Blues, xticks_rotation=45)
                # add legend title and axis labels
                plt.xlabel('Predicted Label')
                plt.ylabel('True Label')
                plt.title('Confusion Matrix')
                # plt.colorbar(label="Number of Samples")
                plt.show()

                new_df = pd.DataFrame()
                new_df['charge'] = self.charge_states
                new_df['precision'] = precision_score(true_labels, predicted_labels, average=None)
                new_df['recall'] = recall_score(true_labels, predicted_labels, average=None)
                new_df['f1'] = f1_score(true_labels, predicted_labels, average=None)
                print(new_df)

            else:
                charge_dict_true = dict()
                charge_dict_pred = dict()
                for index, row in enumerate(test_label):
                    for index2 in range(len(row)):
                        if index2 + charge_states[0] not in charge_dict_true:
                            charge_dict_true[index2 + charge_states[0]] = []
                            charge_dict_pred[index2 + charge_states[0]] = []
                            charge_dict_true[index2 + charge_states[0]].append(row[index2])
                            charge_dict_pred[index2 + charge_states[0]].append(row[index2])
                        else:
                            charge_dict_true[index2 + charge_states[0]].append(row[index2])
                            charge_dict_pred[index2 + charge_states[0]].append(row[index2])

                for key, value in charge_dict_true.items():
                    true_labels = value
                    predicted_labels = charge_dict_pred[key]

                    cm = confusion_matrix(true_labels, predicted_labels)
                    print(cm)

                    classes_here = [0, key]
                    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes_here)

                    disp.plot(cmap=plt.cm.Blues, xticks_rotation=45)
                    # add legend title and axis labels
                    plt.xlabel('Predicted Label')
                    plt.ylabel('True Label')
                    plt.title(f'Confusion Matrix, Charge State {key}')
                    # plt.colorbar(label="Number of Samples")
                    # write text below plot
                    plt.text(-0.5, 2.1,
                             f"Accuracy: {round(accuracy_score(true_labels, predicted_labels), 3)}\nPrecision (weighted): {round(precision_score(true_labels, predicted_labels, average='weighted'), 3)}\nRecall (weighted): {round(recall_score(true_labels, predicted_labels, average='weighted'), 3)}\nF1-score (weighted): {round(f1_score(true_labels, predicted_labels, average='weighted'), 3)}",
                             fontsize=10)

                    # write text for topk below plot
                    # plt.text(3.5, 7.5, f"TopK-Accuracy: {round(topK_accuracy, 3)}\n\n\n", fontsize=10)

                    plt.show()

                new_df3 = pd.DataFrame(columns=['charge'])

                charge_dict_true = dict()
                charge_dict_pred = dict()
                for index, row in enumerate(test_label):
                    for index2 in range(len(row)):
                        if index2 + charge_states[0] not in charge_dict_true:
                            charge_dict_true[index2 + charge_states[0]] = []
                            charge_dict_pred[index2 + charge_states[0]] = []
                            charge_dict_true[index2 + charge_states[0]].append(row[index2])
                            charge_dict_pred[index2 + charge_states[0]].append(row[index2])
                        else:
                            charge_dict_true[index2 + charge_states[0]].append(row[index2])
                            charge_dict_pred[index2 + charge_states[0]].append(row[index2])

                new_df3['charge'] = [0, 1]
                for key, value in charge_dict_true.items():
                    matrix = confusion_matrix(value, charge_dict_pred[key])
                    new_df3[f'{key}_accuracy'] = matrix.diagonal() / matrix.sum(axis=1)
                    new_df3[f'{key}_precision'] = precision_score(value, charge_dict_pred[key], average=None)
                    new_df3[f'{key}_recall'] = recall_score(value, charge_dict_pred[key], average=None)
                    new_df3[f'{key}_f1'] = f1_score(value, charge_dict_pred[key], average=None)

                new_df2 = pd.DataFrame(columns=['charge', 'accuracy', 'precision', 'recall', 'f1'])

                charges_list = []
                to_take_list = []
                for num in range(charge_states[0], charge_states[-1] + 1):
                    for value in ["not-", ""]:
                        col_name = value + str(num)
                        charges_list.append(col_name)
                        to_take_list.append(f'{num}')
                new_df2['charge'] = charges_list

                counter = 0
                for index, row in new_df2.iterrows():

                    if counter > 1:
                        counter = 0
                    new_df2.at[index, 'accuracy'] = new_df3.at[counter, f'{to_take_list[index]}_accuracy']
                    new_df2.at[index, 'precision'] = new_df3.at[counter, f'{to_take_list[index]}_precision']
                    new_df2.at[index, 'recall'] = new_df3.at[counter, f'{to_take_list[index]}_recall']
                    new_df2.at[index, 'f1'] = new_df3.at[counter, f'{to_take_list[index]}_f1']
                    counter += 1

                print(new_df2)

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

            charge_text = (f"The predicted charge state for the input sequence '{input_sequence}' is "
                           f"{max_charge_index + 1} [{round(max_charge_value * 100, 2)}%].")
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
