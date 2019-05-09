from __future__ import print_function
import argparse
import json

import numpy as np
import os
from keras import Input, Model

from app.services.paths import *
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout
from keras.layers import LSTM, Bidirectional
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint

from app.services.data_preparation import get_processed_data_sets_for_model

"""To run this code, you'll need to first download and extract the text dataset
    from here: http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz. Change the
    data_path variable below to your local exraction path"""

# data_path = os.path.join("simple-examples", "data")
DATA_PATH = CHAT_DATA_TYPE_FILES


def get_sentence_vectors(data_type):
    filename = os.path.join(DATA_PATH, data_type + "-sentence-vectors.txt")
    vectors_data = []
    with open(filename) as vectors_file:
        for vector in vectors_file:
            vector = [float(v) for v in vector.strip().split(",")]
            vectors_data.append(vector)
    return vectors_data


def create_time_steps(data, number_of_steps):
    """
    Create time steps for LSTM model
    :param number_of_steps: int number of steps to be included in each data step
    :type data: list of data items
    """
    if len(data) < 1:
        return []

    time_stepped_data = []
    start_index, end_index = 0, number_of_steps
    padding = number_of_steps - 1
    data_unit_length = len(data[0])
    zeros_padding = np.zeros((padding, data_unit_length))
    data = np.concatenate((data, zeros_padding), axis=0)

    while end_index <= len(data):
        time_stepped_data.append(data[start_index:end_index])
        start_index += 1
        end_index += 1
    return np.array(time_stepped_data)


def load_sentence_vectors():
    # get the data paths
    train_data = get_sentence_vectors("train")
    valid_data = get_sentence_vectors("validation")
    test_data = get_sentence_vectors("test")
    return np.array(train_data), np.array(valid_data), np.array(test_data)


def get_labels_and_features_data(include_word_sequences=False):
    processed_data, data_type_log_ids, train_validation_and_test_dates = (
        get_processed_data_sets_for_model(include_word_sequences=include_word_sequences)
    )

    processed_data["train_features_X"] = processed_data["train_features_X"].values,
    processed_data["validation_features_X"] = processed_data["validation_features_X"].values,
    processed_data["test_features_X"] = processed_data["test_features_X"].values

    return processed_data,  data_type_log_ids, train_validation_and_test_dates


class KerasBatchGenerator(object):

    def __init__(self, data, num_steps, batch_size, vocabulary, skip_step=5):
        self.data = data
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.vocabulary = vocabulary
        # this will track the progress of the batches sequentially through the
        # data set - once the data reaches the end of the data set it will reset
        # back to zero
        self.current_idx = 0
        # skip_step is the number of words which will be skipped before the next
        # batch is skimmed from the data set
        self.skip_step = skip_step

    def generate(self):
        x = np.zeros((self.batch_size, self.num_steps))
        y = np.zeros((self.batch_size, self.num_steps, self.vocabulary))
        while True:
            for i in range(self.batch_size):
                if self.current_idx + self.num_steps >= len(self.data):
                    # reset the index back to the start of the data set
                    self.current_idx = 0
                x[i, :] = self.data[self.current_idx:self.current_idx + self.num_steps]
                temp_y = self.data[self.current_idx + 1:self.current_idx + self.num_steps + 1]
                # convert all of temp_y into a one hot representation
                y[i, :, :] = to_categorical(temp_y, num_classes=self.vocabulary)
                self.current_idx += self.skip_step
            yield x, y


def main(data_path):
    parser = argparse.ArgumentParser()
    parser.add_argument('run_opt', type=int, default=1, help='An integer: 1 to train, 2 to test')
    parser.add_argument('--data_path', type=str, default=data_path, help='The full path of the training data')
    args = parser.parse_args()

    if args.data_path:
        data_path = args.data_path

    # Model Parameters
    # ----------------
    batch_size = 16
    num_epochs = 100
    hidden_nodes_size = 32
    use_dropout = True
    time_steps = 100
    input_dimension = 150
    # input_dimension = train_features_X.shape[1] + 150
    bidirectional = False
    dropout_rate = 0.5

    # Data Preparation
    # ----------------
    train_sentence_vectors, validation_sentence_vectors, test_sentence_vectors = load_sentence_vectors()
    processed_data, data_type_log_ids, train_validation_and_test_dates = get_labels_and_features_data()
    (
        train_y, validation_y, test_y,
        train_features_X, validation_features_X, test_features_X,
        train_sequences_X, validation_sequences_X, test_sequences_X,
        train_chat_logs, validation_chat_logs, test_chat_logs
    ) = (
        processed_data["train_y"], processed_data["validation_y"], processed_data["test_y"],
        processed_data["train_features_X"], processed_data["validation_features_X"],
        processed_data["test_features_X"],
        processed_data["train_sequences_X"], processed_data["validation_sequences_X"],
        processed_data["test_sequences_X"],
        processed_data["train_chat_logs"], processed_data["validation_chat_logs"],
        processed_data["test_chat_logs"]
    )

    train_sentence_vectors_features_concat = np.concatenate((train_sentence_vectors, train_features_X), axis=1)
    validation_sentence_vectors_features_concat = np.concatenate(
        (validation_sentence_vectors, validation_features_X),
        axis=1
    )
    test_sentence_vectors_features_concat = np.concatenate((test_sentence_vectors, test_features_X), axis=1)

    train_sentence_vectors = create_time_steps(train_sentence_vectors, time_steps)
    validation_sentence_vectors = create_time_steps(validation_sentence_vectors, time_steps)
    test_sentence_vectors = create_time_steps(test_sentence_vectors, time_steps)

    print("Train: ", train_sentence_vectors.shape)
    print("Validation: ", validation_sentence_vectors.shape)
    print("test: ", test_sentence_vectors.shape)

    train_features_X, validation_features_X, test_features_X = (
        create_time_steps(train_features_X, time_steps),
        create_time_steps(validation_features_X, time_steps),
        create_time_steps(test_features_X, time_steps)
    )

    (
        train_sentence_vectors_features_concat,
        validation_sentence_vectors_features_concat,
        test_sentence_vectors_features_concat
    ) = (
        create_time_steps(train_sentence_vectors_features_concat, time_steps),
        create_time_steps(validation_sentence_vectors_features_concat, time_steps),
        create_time_steps(test_sentence_vectors_features_concat, time_steps)
    )

    print("Concat Shape: ", train_sentence_vectors_features_concat.shape)

    if args.run_opt == 1:
        model = Sequential()
        # model.add(Embedding(vocabulary, hidden_nodes_size, input_length=num_steps))
        # model.add(Dropout(0.5))
        model.add(
            # Bidirectional(
                LSTM(
                    hidden_nodes_size,
                    return_sequences=True,
                    input_shape=(time_steps, input_dimension)
                    )
            # )
        )
        if use_dropout:
            model.add(Dropout(0.5))
        # model.add(Bidirectional(LSTM(hidden_nodes_size, return_sequences=True)))
        model.add(LSTM(hidden_nodes_size, return_sequences=True))
        if use_dropout:
            model.add(Dropout(0.5))
        # model.add(Bidirectional(LSTM(hidden_nodes_size)))
        model.add(LSTM(hidden_nodes_size))
        if use_dropout:
            model.add(Dropout(0.5))

        model.add(Dense(2, activation='softmax'))
        # model.add(Activation('softmax'))

        model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
        model_dir = join(data_path, 'models', '{}_ts_{}_dro_{}_hino_lstm2'.format(
            time_steps, "05", hidden_nodes_size))
        if not os.path.exists(model_dir):
            os.mkdir(model_dir)

        print(model.summary())

        model_summary_filename = join(model_dir, "model_summary.txt")
        with open(model_summary_filename, "w") as model_summary_file:
            print("\nSaving model summary to ", model_summary_filename, "\n")
            model.summary(print_fn=lambda x: model_summary_file.write(x + "\n"))

        checkpointer = ModelCheckpoint(
            filepath=join(model_dir, "model-{epoch:02d}.hdf5"),
            verbose=1,
            period=1
        )

        model_history = model.fit(
            train_sentence_vectors, train_y,
            batch_size=batch_size, epochs=num_epochs,
            validation_data=(validation_sentence_vectors, validation_y),
            callbacks=[checkpointer]
        )

        model.save(os.path.join(model_dir, "final_model.hdf5"))
        with open(os.path.join(model_dir, "model_details.json"), "w") as history_file:
            model_history.validation_data = []
            model_history.model = str(model_history.model.__dict__)
            history_file.write(json.dumps(model_history.__dict__, indent=2))

    elif args.run_opt == 2:
        # model = load_model(data_path + "/model-40.hdf5")
        model = load_model(data_path + "/model-250.hdf5")
    elif args.run_opt == 3:
        sentence_features_input = Input(
            shape=(time_steps, input_dimension), dtype='float32', name='sentence_features_input'
        )
        if bidirectional:
            lstm_out = Bidirectional(
                    LSTM(hidden_nodes_size, return_sequences=True)
                )(sentence_features_input)
        else:
            lstm_out = LSTM(hidden_nodes_size, return_sequences=True, name="lstm_out_1")(sentence_features_input)

        dropout_1 = Dropout(dropout_rate, name="dropout_1")(lstm_out)

        if bidirectional:
            lstm_out_2 = Bidirectional(
                LSTM(hidden_nodes_size, return_sequences=True)
            )(dropout_1)
        else:
            lstm_out_2 = LSTM(hidden_nodes_size, return_sequences=True, name="lstm_out_2")(dropout_1)
        dropout_2 = Dropout(dropout_rate, name="dropout_2")(lstm_out_2)

        if bidirectional:
            lstm_out_3 = Bidirectional(
                LSTM(hidden_nodes_size)
            )(dropout_2)
        else:
            lstm_out_3 = LSTM(hidden_nodes_size, name="lstm_out_3")(dropout_2)
        dropout_3 = Dropout(dropout_rate, name="dropout_3")(lstm_out_3)

        sentence_features_output = Dense(2, activation='softmax', name="sentence_features_output")(dropout_3)

        # Define a model with two inputs and two outputs
        merged_model = Model(
            inputs=[sentence_features_input],
            outputs=[sentence_features_output]
            # inputs=[sentence_features_input],
            # outputs=[sentence_features_output]
        )

        model_folder_name = join(
            data_path, "models", "{}_ts_{}_dro_lstm".format(
                time_steps, dropout_rate)
                )
        if not os.path.exists(model_folder_name):
            os.mkdir(model_folder_name)

        checkpointer = ModelCheckpoint(
            filepath=join(model_folder_name, 'model-{epoch:02d}.hdf5'),
            verbose=1,
            period=1
        )

        merged_model.compile(
            optimizer="rmsprop",
            # optimizer="adam",
            loss='categorical_crossentropy',
            metrics=['accuracy']
            # loss_weights=[.9, 0.7],
            # sample_weight_mode="temporal"
        )

        print(merged_model.summary())
        model_summary_filename = join(model_folder_name, "model_summary.txt")
        with open(model_summary_filename, "w") as model_summary_file:
            print("\nSaving model summary to ", model_summary_filename, "\n")
            merged_model.summary(print_fn=lambda x: model_summary_file.write(x + "\n"))

        merged_model_history = merged_model.fit(
            [train_sentence_vectors],
            [train_y],
            # validation_split=0.2,
            validation_data=[
                [validation_sentence_vectors],
                [validation_y]
            ],
            epochs=num_epochs,
            batch_size=batch_size,
            callbacks=[checkpointer]
            # class_weight=[class_weights, class_weights]
        )

        merged_model.save(join(model_folder_name, "final_model.hdf5"))
        with open(join(model_folder_name, "model_details.json"),
                  "w") as history_file:
            merged_model_history.validation_data = []
            merged_model_history.model = str(merged_model_history.model.__dict__)
            history_file.write(json.dumps(merged_model_history.__dict__, indent=2))


if __name__ == "__main__":
    pass
    # main(DATA_PATH)

