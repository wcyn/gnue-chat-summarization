import json
import linecache
import pandas as pd
import random

from keras.utils import to_categorical
from os.path import join
from sklearn.preprocessing import StandardScaler

DATA_FILES_DIR = join("..", "..", "feature_extraction", "data_files")
DATA_FILES_V2_DIR = join("..", "..", "feature_extraction", "data_files_v2")
FEATURES_DIR = join("..", "..", "feature_extraction", "feature_outputs")

ALL_CHAT_FEATURES_FILENAME = join(FEATURES_DIR, "all_chat_features.csv")
CHAT_LOGS_FILENAME = join(DATA_FILES_DIR, "gnue_irc_chat_logs_preprocessed.txt")
CONVERSATION_LOG_IDS_FILENAME = join(DATA_FILES_V2_DIR, "conversation_log_ids.json")
SUMMARIZED_CHAT_DATE_PARTITIONS_FILENAME = join(
    DATA_FILES_DIR, "summarized_chat_date_partitions_cumulative_count.csv"
)
SUMMARIZED_CHAT_FEATURES_FILENAME = join(FEATURES_DIR, "summarized_chats_features.csv")
SUMMARIZED_CHAT_LOG_IDS_FILENAME = join(DATA_FILES_DIR, "summarized_chat_log_ids.csv")

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
    return current_column_index == len(columns)


def validate_column_names(column_names):
    return column_names == FEATURES_COLUMN_NAMES


def validate_total_number_of_records(records_data_frames_dict):
    train_rows = records_data_frames_dict["train_features_X"].shape[0]
    validation_rows = records_data_frames_dict["validation_features_X"].shape[0]
    test_rows = records_data_frames_dict["test_features_X"].shape[0]
    return train_rows + validation_rows + test_rows == TOTAL_NUMBER_OF_CHATS


def convert_list_to_data_frame(data_records, column_names):
    return pd.DataFrame(data_records, columns=column_names)


def get_train_validation_and_test_data(
        features_filename,
        train_validation_and_test_dates,
        conversation_log_ids_dict
):
    """
    Get features data for train, validation and test sets
    :param features_filename: The path to the file that contains all the features
    :param train_validation_and_test_dates: Dates for the training, validation and test sets
    :param conversation_log_ids_dict: A dictionary containing dates as keys and values as a list of log_ids in the shape:
        {
            "date_of_log_1": [log_id_1, log_id_2, ...],
            "date_of_log_2": [log_id_56, log_id_57, ...]
        }
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
    linecache.getline(features_filename, 33)
    column_names = linecache.getline(features_filename, 1).strip().split(",")

    if not validate_column_names(column_names):
        raise ValueError("Column names do not match expected column names")

    for data_type, dates in train_validation_and_test_dates.items():
        x_data_key, y_data_key = data_type + "_features_X", data_type + "_y"
        for date in dates:
            log_ids = conversation_log_ids_dict[date]
            for log_id in log_ids:
                feature_data = linecache.getline(features_filename, log_id + 1).strip().split(",")
                feature_data = [float(value) for value in feature_data]
                # print(feature_data)
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
    if not validate_total_number_of_records(data_records):
        raise ValueError("Number of records does not match the expected total number")
    return data_records


def get_conversation_log_ids():
    with open(CONVERSATION_LOG_IDS_FILENAME) as f:
        return json.load(f)


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
    data_max_values = data_set_df.max()
    data_min_values = data_set_df.min()
    new_data_set_df = data_set_df.copy()

    for column in columns_to_normalize:
        new_data_set_df[column] = (
                                           new_data_set_df[column] - data_min_values[column]) / (
                                           data_max_values[column] - data_min_values[column]
                                   )
    return new_data_set_df


def standardize_data_set(data_set_df):
    scaler = StandardScaler()
    scaler.fit(data_set_df)
    scaled_array = scaler.transform(data_set_df)
    columns = data_set_df.columns.values
    if not validate_column_order(columns):
        raise ValueError("Wrong column order: ", columns)
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
    :param train_validation_and_test_data: dict containing Pandas DataFrames. Has the shape:
        {
            "train_features_X": Pandas DataFrame,
            "validation_features_X": Pandas DataFrame,
            "test_features_X": Pandas DataFrame,
            "train_y": Pandas DataFrame,
            "validation_y": Pandas DataFrame,
            "test_y": Pandas DataFrame,
        }
    :return: A dict of Pandas DataFrames with processed data
    """
    # import pdb; pdb.set_trace()
    for data_key, data_records in train_validation_and_test_data.items():
        if data_key in FEATURES_DATA_KEYS:
            train_validation_and_test_data[data_key] = process_features_data(
                train_validation_and_test_data[data_key], columns_to_drop
            )
        elif data_key in LABELS_DATA_KEYS:
            train_validation_and_test_data[data_key] = process_labels_data(
                train_validation_and_test_data[data_key]
            )
        else:
            raise ValueError("Invalid Data Key: ", data_key)

    if not validate_total_number_of_records(train_validation_and_test_data):
        raise ValueError("Number of records does not match the expected total number")

    return train_validation_and_test_data


def main():
    conversation_log_ids = get_conversation_log_ids()
    train_validation_and_test_dates = get_train_validation_and_test_dates(list(conversation_log_ids.keys()))
    data = get_train_validation_and_test_data(
        ALL_CHAT_FEATURES_FILENAME,
        train_validation_and_test_dates,
        conversation_log_ids
    )

    return process_data_sets_for_model(data)


if __name__ == "__main__":
    print(main()["test_features_X"].tail())
