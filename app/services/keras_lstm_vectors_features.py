from __future__ import print_function
import argparse
import collections
import json

import keras
import numpy as np
import os
import tensorflow as tf
from keras import Input, Model

from app.services.paths import *
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Embedding, Dropout, TimeDistributed, Flatten
from keras.layers import LSTM, Bidirectional
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint

from app.services.data_preparation import get_processed_data_sets_for_model

"""To run this code, you'll need to first download and extract the text dataset
    from here: http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz. Change the
    data_path variable below to your local exraction path"""

# data_path = os.path.join("simple-examples", "data")
DATA_PATH = CHAT_DATA_TYPE_FILES


def read_words(filename):
    with tf.gfile.GFile(filename, "r") as f:
        return f.read().replace("\n", "<eos>").split()


def build_vocab(filename):
    data = read_words(filename)

    counter = collections.Counter(data)
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

    words, _ = list(zip(*count_pairs))
    word_to_id = dict(zip(words, range(len(words))))

    return word_to_id


def file_to_word_ids(filename, word_to_id):
    data = read_words(filename)
    return [word_to_id[word] for word in data if word in word_to_id]


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


def load_data():
    # get the data paths
    train_path = os.path.join(DATA_PATH, "train.txt")
    valid_path = os.path.join(DATA_PATH, "validation.txt")
    test_path = os.path.join(DATA_PATH, "test.txt")

    # build the complete vocabulary, then convert text data to list of integers
    word_to_id = build_vocab(train_path)
    # train_data = file_to_word_ids(train_path, word_to_id)
    # valid_data = file_to_word_ids(valid_path, word_to_id)
    # test_data = file_to_word_ids(test_path, word_to_id)
    train_data = get_sentence_vectors("train")
    valid_data = get_sentence_vectors("validation")
    test_data = get_sentence_vectors("test")
    # print("Train: ", train_data[:1])

    vocabulary = len(word_to_id)
    reversed_dictionary = dict(zip(word_to_id.values(), word_to_id.keys()))

    # print(train_data[:5])
    # print(word_to_id)
    # print(vocabulary)
    # print(" ".join([reversed_dictionary[x] for x in train_data[:10]]))
    return np.array(train_data), np.array(valid_data), np.array(test_data), vocabulary, reversed_dictionary


def get_labels_and_features_data():
    processed_data, data_type_log_ids, train_validation_and_test_dates = (
        get_processed_data_sets_for_model(include_word_sequences=True)
    )
    return processed_data


train_sequences, validation_sequences, test_sequences, vocabulary, reversed_dictionary = load_data()
labels_and_features_data = get_labels_and_features_data()
train_y, validation_y, test_y = (
    labels_and_features_data["train_y"],
    labels_and_features_data["validation_y"],
    labels_and_features_data["test_y"]
)
train_features_X, validation_features_X, test_features_X = (
    labels_and_features_data["train_features_X"],
    labels_and_features_data["validation_features_X"],
    labels_and_features_data["test_features_X"]
)

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


num_steps = 30
batch_size = 16
num_epochs = 100
train_data_generator = KerasBatchGenerator(train_sequences, num_steps, batch_size, vocabulary,
                                           skip_step=num_steps)
valid_data_generator = KerasBatchGenerator(validation_sequences, num_steps, batch_size, vocabulary,
                                           skip_step=num_steps)

hidden_nodes_size = 16
use_dropout = True
time_steps = 50
# input_dimension = 150
input_dimension = train_features_X.shape[1]
bidirectional = True

print("Train: ", train_sequences.shape)
print("Validation: ", validation_sequences.shape)
print("test: ", test_sequences.shape)

# t = create_time_steps(np.array([[1,2,3],[4,5,6]]), 4)
# print(t)

train_sequences = create_time_steps(train_sequences, time_steps)
validation_sequences = create_time_steps(validation_sequences, time_steps)
test_sequences = create_time_steps(test_sequences, time_steps)
print("Train: ", train_sequences.shape)
print("Validation: ", validation_sequences.shape)
print("test: ", test_sequences.shape)
# train_data = train_data.reshape((train_data.shape[0], time_steps, train_data.shape[1]))
# valid_data = valid_data.reshape((valid_data.shape[0], 1, valid_data.shape[1]))
# test_data = test_data.reshape((test_data.shape[0], 1, test_data.shape[1]))

train_features_X, validation_features_X, test_features_X = (
    create_time_steps(train_features_X.values, time_steps),
    create_time_steps(validation_features_X.values, time_steps),
    create_time_steps(test_features_X.values, time_steps)
)


def main(data_path):
    parser = argparse.ArgumentParser()
    parser.add_argument('run_opt', type=int, default=1, help='An integer: 1 to train, 2 to test')
    parser.add_argument('--data_path', type=str, default=data_path, help='The full path of the training data')
    args = parser.parse_args()

    if args.data_path:
        data_path = args.data_path

    if args.run_opt == 3:
        model = Sequential()
        # model.add(Embedding(vocabulary, hidden_nodes_size, input_length=num_steps))
        # model.add(Dropout(0.5))
        model.add(
            Bidirectional(
                LSTM(hidden_nodes_size, return_sequences=True),
                input_shape=(time_steps, input_dimension)
            )
        )
        if use_dropout:
            model.add(Dropout(0.5))
        model.add(Bidirectional(LSTM(hidden_nodes_size, return_sequences=True)))
        if use_dropout:
            model.add(Dropout(0.5))
        model.add(Bidirectional(LSTM(hidden_nodes_size)))
        if use_dropout:
            model.add(Dropout(0.5))


        # model.add(TimeDistributed(Dense(vocabulary)))
        model.add(Dense(2, activation='softmax'))
        # model.add(Activation('softmax'))

        # optimizer = Adam()

        model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

        print(model.summary())
        checkpointer = ModelCheckpoint(
            filepath=data_path + '/models/model-{epoch:02d}' + '-{}ts_bidi.hdf5'.format(time_steps),
            verbose=1,
            period=5
        )

        model_history = model.fit(
            train_sequences, train_y,
            batch_size=batch_size, epochs=num_epochs,
            validation_data=(validation_sequences, validation_y),
            callbacks=[checkpointer]
        )
        # model_history = model.fit_generator(
        #     train_data_generator.generate(), len(train_data)//(batch_size*num_steps), num_epochs,
        #     validation_data=valid_data_generator.generate(),
        #     validation_steps=len(valid_data)//(batch_size*num_steps), callbacks=[checkpointer]
        # )
        # model.fit_generator(train_data_generator.generate(), 2000, num_epochs,
        #                     validation_data=valid_data_generator.generate(),
        #                     validation_steps=10)
        model.save(os.path.join(data_path, "/models/final_model_{}ts_bidi.hdf5".format(time_steps)))
        with open(os.path.join(data_path, "/models/model_history{}ts_bidi.json".format(time_steps)), "w") as history_file:
            model_history.validation_data = []
            model_history.model = str(model_history.model.__dict__)
            history_file.write(json.dumps(model_history.__dict__, indent=2))

    elif args.run_opt == 2:
        # model = load_model(data_path + "/model-40.hdf5")
        model = load_model(data_path + "/model-250.hdf5")
        dummy_iters = 40
        example_training_generator = KerasBatchGenerator(train_sequences, num_steps, 1, vocabulary,
                                                         skip_step=1)
        # print("Training data:")
        # for i in range(dummy_iters):
        #     dummy = next(example_training_generator.generate())
        # num_predict = 10
        # true_print_out = "Actual words: "
        # pred_print_out = "Predicted words: "
        # for i in range(num_predict):
        #     data = next(example_training_generator.generate())
        #     prediction = model.predict(data[0])
        #     predict_word = np.argmax(prediction[:, num_steps-1, :])
        #     true_print_out += reversed_dictionary[train_data[num_steps + dummy_iters + i]] + " "
        #     pred_print_out += reversed_dictionary[predict_word] + " "
        # print(true_print_out)
        # print(pred_print_out)
        # # test data set
        # dummy_iters = 40
        # example_test_generator = KerasBatchGenerator(
        #     test_data, num_steps, 1, vocabulary, skip_step=1)
        # print("Test data:")
        # for i in range(dummy_iters):
        #     dummy = next(example_test_generator.generate())
        # num_predict = 10
        # true_print_out = "Actual words: "
        # pred_print_out = "Predicted words: "
        # for i in range(num_predict):
        #     data = next(example_test_generator.generate())
        #     prediction = model.predict(data[0])
        #     predict_word = np.argmax(prediction[:, num_steps - 1, :])
        #     true_print_out += reversed_dictionary[test_data[num_steps + dummy_iters + i]] + " "
        #     pred_print_out += reversed_dictionary[predict_word] + " "
        # print(true_print_out)
        # print(pred_print_out)
    elif args.run_opt == 1:
        sentence_features_input = Input(
            shape=(time_steps, input_dimension), dtype='float32', name='sentence_features_input'
        )
        if bidirectional:
            lstm_out = Bidirectional(
                    LSTM(hidden_nodes_size, return_sequences=True)
                )(sentence_features_input)
        else:
            lstm_out = LSTM(hidden_nodes_size, return_sequences=True, name="lstm_out_1")(sentence_features_input)

        dropout_1 = Dropout(0.2, name="dropout_1")(lstm_out)

        if bidirectional:
            lstm_out_2 = Bidirectional(
                LSTM(hidden_nodes_size, return_sequences=True)
            )(dropout_1)
        else:
            lstm_out_2 = LSTM(hidden_nodes_size, return_sequences=True, name="lstm_out_2")(dropout_1)
        dropout_2 = Dropout(0.2, name="dropout_2")(lstm_out_2)

        if bidirectional:
            lstm_out_3 = Bidirectional(
                LSTM(hidden_nodes_size)
            )(dropout_2)
        else:
            lstm_out_3 = LSTM(hidden_nodes_size, name="lstm_out_3")(dropout_2)
        dropout_3 = Dropout(0.2, name="dropout_3")(lstm_out_3)

        sentence_features_output = Dense(2, activation='softmax', name="sentence_features_output")(dropout_3)

        # Define a model with two inputs and two outputs
        merged_model = Model(
            inputs=[sentence_features_input],
            outputs=[sentence_features_output]
            # inputs=[sentence_features_input],
            # outputs=[sentence_features_output]
        )

        model_folder_name = join(data_path, "models", "{}_ts_bidi_lstm_features".format(time_steps))
        if not os.path.exists(model_folder_name):
            os.mkdir(model_folder_name)

        checkpointer = ModelCheckpoint(
            filepath=join(model_folder_name, 'model-{epoch:02d}.hdf5'),
            verbose=1,
            period=5
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
            [train_features_X],
            [train_y],
            # validation_split=0.2,
            validation_data=[
                [validation_features_X],
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
    # pass
    main(DATA_PATH)
