import json
import linecache
import pandas as pd
import random

from keras.utils import to_categorical
from app.services.logs import get_log_messages_by_date
from os.path import join, exists
from sklearn.preprocessing import StandardScaler

# DATA_FILES_DIR = join("..", "feature_extraction", "data_files")
# DATA_FILES_V2_DIR = join("..", "feature_extraction", "data_files_v2")
# FEATURES_DIR = join("..", "feature_extraction", "feature_outputs")

DATA_FILES_DIR = join("feature_extraction", "data_files")
DATA_FILES_V2_DIR = join("feature_extraction", "data_files_v2")
FEATURES_DIR = join("feature_extraction", "feature_outputs")

CHAT_LOGS_FILENAME = join(DATA_FILES_DIR, "gnue_irc_chat_logs_preprocessed.txt")
CHAT_DATA_TYPE_FILES = join("app", "services", "chat_data_files")

ALL_CHAT_FEATURES_FILENAME = join(FEATURES_DIR, "all_chat_features.csv")
CONVERSATION_LOG_IDS_FILENAME = join(DATA_FILES_V2_DIR, "conversation_log_ids.json")
SUMMARIZED_CHAT_DATE_PARTITIONS_FILENAME = join(
    DATA_FILES_DIR, "summarized_chat_date_partitions_cumulative_count.csv"
)
SUMMARIZED_CHAT_FEATURES_FILENAME = join(FEATURES_DIR, "summarized_chats_features.csv")
SUMMARIZED_CHAT_LOG_IDS_FILENAME = join(DATA_FILES_DIR, "summarized_chat_log_ids.csv")
TIME_OFFSET = 1554206530

COLUMNS_TO_DROP = ["log_id", "mean_tf_idf", "mean_tf_isf"]
FEATURES_COLUMN_NAMES = [  # In the correct order
    'log_id',
    'absolute_sentence_position',
    'sentence_length',
    'number_of_special_terms',
    'sentiment_score',
    'mean_tf_idf',
    'normalized_mean_tf_idf',
    'mean_tf_isf',
    'normalized_mean_tf_isf',
    'is_summary'
]
FEATURES_DATA_KEYS = {"train_features_X", "validation_features_X", "test_features_X"}
LABELS_DATA_KEYS = {"train_y", "validation_y", "test_y"}
TOTAL_NUMBER_OF_CHATS = 20715


def get_train_validation_and_test_dates(conversation_dates, validation_ratio=0.2, test_ratio=0.2, random_seed=4):
    """
    Get dates to use for training, validation and testing
    :param conversation_dates: A list of dates to randomize and partition for training, validation and test data
    :param random_seed: A seed to use for the randomizer in order to get the same shuffle results every time
    :param validation_ratio: Ratio of number of conversations to total conversation to be used for the validation set
    :param test_ratio: Ratio of number of conversation to total conversation to be used for the test set
    :return: A dictionary containing a random list of dates for the training, validation and test sets
    """
    # Ensure there's some training set
    assert validation_ratio + test_ratio < 1

    number_of_dates = len(conversation_dates)
    shuffled_indexes = [index for index in range(number_of_dates)]
    random.Random(random_seed).shuffle(shuffled_indexes)

    validation_split = int(validation_ratio * number_of_dates)
    test_split = int(test_ratio * number_of_dates)

    index_for_train_validation_split = (test_split + validation_split) * -1
    index_for_validation_test_split = test_split * -1

    train_dates = [conversation_dates[index] for index in shuffled_indexes[:index_for_train_validation_split]]
    validation_dates = [conversation_dates[index] for index in shuffled_indexes[
                                                               index_for_train_validation_split:index_for_validation_test_split
                                                               ]]
    test_dates = [conversation_dates[index] for index in shuffled_indexes[index_for_validation_test_split:]]

    return {
        "train": train_dates,
        "validation": validation_dates,
        "test": test_dates
    }


def validate_column_order(columns):
    current_column_index = expected_column_index = 0
    while expected_column_index < len(FEATURES_COLUMN_NAMES) and current_column_index < len(columns):
        if FEATURES_COLUMN_NAMES[expected_column_index] == columns[current_column_index]:
            current_column_index += 1
        expected_column_index += 1
    if current_column_index != len(columns):
        raise ValueError("Wrong column order: ", columns)


def validate_column_names(column_names):
    if column_names != FEATURES_COLUMN_NAMES:
        raise ValueError("Column names do not match expected column names")


def validate_total_number_of_records(records_data_frames_dict, with_labels=False):
    total_records = 0
    for records in records_data_frames_dict.values():

        total_records += len(records)
    if with_labels:
        total_records //= 2

    if total_records != TOTAL_NUMBER_OF_CHATS:
        raise ValueError("Number of records ({}) does not match the expected total number ({})".format(
            total_records, TOTAL_NUMBER_OF_CHATS
        ))


def convert_list_to_data_frame(data_records, column_names):
    return pd.DataFrame(data_records, columns=column_names)


def concatenate_data_type_log_ids(train_validation_and_test_dates, conversation_log_ids_dict):
    """
    :param train_validation_and_test_dates: Dates for the training, validation and test sets. Have the shape:
        {
            "train": [date_4, date_1, date_6, ...],
            "validation": [date_8, date_3, date_9, ...],
            "test": [date_5, date_2, date_7, ...],
        }
    :param conversation_log_ids_dict: A dictionary containing dates as keys and values as
     a list of log_ids in the shape:
        {
            "date_of_log_1": [log_id_1, log_id_2, ...],
            "date_of_log_2": [log_id_56, log_id_57, ...]
        }
    :return: Concatenated_log_ids_for each data type (ie. train, validation etc.) in the shape:
        {
            "train": [log_id_1, log_id_2, ...],
            "validation": [log_id_56, log_id_57, ...]
            "test": [log_id_71, log_id_72, ...]
        }
    """
    data_type_log_ids = {
        "train": [],
        "validation": [],
        "test": []
    }
    for data_type, dates in train_validation_and_test_dates.items():
        for date in dates:
            log_ids_for_date = conversation_log_ids_dict[date]
            data_type_log_ids[data_type].extend(log_ids_for_date)

    validate_total_number_of_records(data_type_log_ids)
    return data_type_log_ids


def get_train_validation_and_test_data(
        features_filename,
        data_type_log_ids
):
    """
    Get features data for train, validation and test sets
    :param data_type_log_ids: dict containing all log ids for each data type. Has the form:
        {
            "train": [log_id_1, log_id_2, ...],
            "validation": [log_id_56, log_id_57, ...]
            "test": [log_id_71, log_id_72, ...]
        }
    :param features_filename: The path to the file that contains all the features
    :return: Pandas DataFrame with the shape:
        {
            "train_features_X": Pandas DataFrame,
            "validation_features_X": Pandas DataFrame,
            "test_features_X": Pandas DataFrame,
            "train_y": Pandas DataFrame,
            "validation_y": Pandas DataFrame,
            "test_y": Pandas DataFrame,
        }
    """
    #
    data_records = {
        key: [] for key in FEATURES_DATA_KEYS ^ LABELS_DATA_KEYS
    }
    column_names = linecache.getline(features_filename, 1).strip().split(",")

    validate_column_names(column_names)

    for data_type, log_ids in data_type_log_ids.items():
        x_data_key, y_data_key = data_type + "_features_X", data_type + "_y"
        for log_id in log_ids:
            feature_data = linecache.getline(features_filename, log_id + 1).strip().split(",")
            feature_data = [float(value) for value in feature_data]
            assert int(feature_data[0]) == log_id
            data_records[x_data_key].append(feature_data[:-1])
            data_records[y_data_key].append(feature_data[-1])

    x_column_names, y_column_name = column_names[:-1], column_names[-1:]

    # Create Pandas DataFrames
    for data_type, data_record_group in data_records.items():
        if data_type in FEATURES_DATA_KEYS:
            column_names = x_column_names
        elif data_type in LABELS_DATA_KEYS:
            column_names = y_column_name
        else:
            raise ValueError("Invalid Data Key: ", data_type)
        data_records[data_type] = convert_list_to_data_frame(data_record_group, column_names)
    validate_total_number_of_records(data_records, with_labels=True)

    return data_records


def get_conversation_log_ids():
    """
    Load conversation log ids. Is in the form:
        {
            "date_of_log_1": [log_id_1, log_id_2, ...],
            "date_of_log_2": [log_id_56, log_id_57, ...]
        }
    :return:
    """
    with open(CONVERSATION_LOG_IDS_FILENAME) as f:
        return json.load(f)


# conversation_log_ids = get_conversation_log_ids()
# train_validation_and_test_dates = get_train_validation_and_test_dates(list(conversation_log_ids.keys()))
# import pprint
# pprint.pprint(concatenate_data_type_log_ids(train_validation_and_test_dates, conversation_log_ids))


def normalize_data_set(
        data_set_df,
        columns_to_normalize=(
                "number_of_special_terms",
                "sentence_length",
                "sentiment_score")
):
    """
    Normalize data in the data set
    :param columns_to_normalize: The columns to be normalized
    :param data_set_df: A pandas data frame containing the data to be normalized
    :return: A Pandas DataFrame containing the normalized data
    """
    data_set_df = data_set_df.abs()
    data_max_values = data_set_df.max()
    data_min_values = data_set_df.min()
    new_data_set_df = data_set_df.copy()

    for column in columns_to_normalize:
        new_data_set_df[column] = (new_data_set_df[column] - data_min_values[column]) / (
                data_max_values[column] - data_min_values[column])
    return new_data_set_df


def standardize_data_set(data_set_df):
    scaler = StandardScaler()
    scaler.fit(data_set_df)
    scaled_array = scaler.transform(data_set_df)
    columns = data_set_df.columns.values
    validate_column_order(columns)

    return pd.DataFrame(scaled_array,  columns=data_set_df.columns.values)


def process_features_data(features_data_frame, columns_to_drop, method="standardize"):
    """
    Process features X data. Drop columns and then standardize or normalize the data
    :param features_data_frame: The data to be processed
    :param columns_to_drop: list of columns to drop
    :param method: Can be either standardize or normalize
    :return:
    """
    features_data_frame = features_data_frame.drop(
            columns=columns_to_drop
        )
    if method == "standardize":
        features_data_frame = standardize_data_set(features_data_frame)
    elif method == "normalize":
        features_data_frame = normalize_data_set(features_data_frame)
    else:
        raise ValueError("Invalid method: ", method)

    return features_data_frame


def process_labels_data(labels_data_frame):
    """
    Convert labels data into categorical one-hot vector
    :param labels_data_frame: Labels data to be processed
    :return: array representing the labels' one-hot vectors
    """
    return to_categorical(labels_data_frame)


def process_data_sets_for_model(
        train_validation_and_test_data,
        columns_to_drop=COLUMNS_TO_DROP
):
    """
    Drop unnecessary columns and normalize data in preparation for the model
    :param columns_to_drop: The columns to drop from the features DataFrames
    :param train_validation_and_test_data: dict containing features and labels. Has the shape:
        {
            "train_features_X": Pandas DataFrame,
            "validation_features_X": Pandas DataFrame,
            "test_features_X": Pandas DataFrame,
            "train_y": array one-hot vector,
            "validation_y": array one-hot vector,
            "test_y": array one-hot vector,
        }
    :return: A dict of Pandas DataFrames with processed data
    """
    for data_key, data_records in train_validation_and_test_data.items():
        if data_key in FEATURES_DATA_KEYS:
            train_validation_and_test_data[data_key] = process_features_data(
                train_validation_and_test_data[data_key], columns_to_drop,
                method="normalize"
            )
        elif data_key in LABELS_DATA_KEYS:
            train_validation_and_test_data[data_key] = process_labels_data(
                train_validation_and_test_data[data_key]
            )
        else:
            raise ValueError("Invalid Data Key: ", data_key)

    validate_total_number_of_records(train_validation_and_test_data, with_labels=True)

    return train_validation_and_test_data


def create_test_validation_and_train_files(train_validation_and_test_dates):
    for data_type, dates in train_validation_and_test_dates.items():
        filename = join(CHAT_DATA_TYPE_FILES, data_type) + ".txt"
        if exists(filename):
            append_write = "a"  # append if already exists
        else:
            append_write = "w"

        print(append_write)
        with open(filename, append_write) as chat_data_file:
            for date in dates:
                messages = get_log_messages_by_date(date)
                for message in messages:
                    chat_data_file.write(message["line_message"] + "\n")


def get_processed_data_sets_for_model():
    conversation_log_ids = get_conversation_log_ids()
    train_validation_and_test_dates = get_train_validation_and_test_dates(list(conversation_log_ids.keys()))
    data_type_log_ids = concatenate_data_type_log_ids(train_validation_and_test_dates, conversation_log_ids)
    data = get_train_validation_and_test_data(
        ALL_CHAT_FEATURES_FILENAME,
        data_type_log_ids
    )
    create_test_validation_and_train_files(train_validation_and_test_dates)

    return process_data_sets_for_model(data), data_type_log_ids, train_validation_and_test_dates


if __name__ == "__main__":
    print(get_processed_data_sets_for_model()[0]["test_features_X"].tail())


"""
      absolute_sentence_position  ...  normalized_mean_tf_isf
3400                    1.666524  ...               -0.859767
3401                    1.682270  ...                0.163343
3402                    1.698016  ...               -0.397122
3403                    1.713762  ...               -0.957587
3404                    1.729508  ...                1.090890
"""
