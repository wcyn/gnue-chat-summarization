import MySQLdb
import numpy as np

from app.database import database
from app.services.data_preparation import (
    get_processed_data_sets_for_model
)
from app.services import keras_lstm
from app.services.logs import get_logs_by_date
from app.services.paths import *
from keras.models import load_model
from os.path import join

connection = database.DatabaseConnection()
db, cursor = connection.db, connection.cursor

merged_hybrid_filename = join(DATA_FILES_DIR, "models", "merged_hybrid_model.h5")
merged_hybrid_2_filename = join(DATA_FILES_DIR, "models", "1157354df_500e_64bs_adam_342min_merged_hybrid_model.h5")
# time_stepped_lstm_filename = join(CHAT_DATA_TYPE_FILES, "models", "model-20-30ts.hdf5")
time_stepped_lstm_filename = join(CHAT_DATA_TYPE_FILES, "models", "model-45-30ts_bidi.hdf5")

MODEL_CONFIGS = {
    "hybrid_lstm_feed_forward": {
        "filename": merged_hybrid_2_filename,
        "has_sequences": True,
        "sequence_size": 100
    },
    "time_stepped_lstm": {
        "filename": time_stepped_lstm_filename,
        "has_sequences": False,
        "sequence_size": None
    },
    "feed_forward":{
        "filename": join(
            DATA_FILES_DIR, "models", "04-02_963_1000_64_adam_5min_ff_neural_net.h5"),
        "has_sequences": False,
        "sequence_size": None
    }
}


def get_neural_network_model(model_filename):
    return load_model(model_filename)


def get_processed_data_set(processed_data, type_of_data):
    """
    Gets the processed_data_set by type, to be used for prediction in a model
    :param processed_data: dict containing features and labels data. Has the shape:
        {
            "train_features_X": Pandas DataFrame,
            "validation_features_X": Pandas DataFrame,
            "test_features_X": Pandas DataFrame,
            "train_y": array one-hot vector,
            "validation_y": array one-hot vector,
            "test_y": array one-hot vector,
            "train_sequences_X": (optional data)list of integers,
            "validation_sequences_X": (optional data)list of integers,
            "test_sequences_X": (optional data)list of integers,
            "train_chat_logs": (optional data)list of strings representing chat logs,
            "validation_chat_logs": (optional data)list of strings representing chat logs,
            "test_chat_logs": (optional data)list of strings representing chat logs,
        }
    :param type_of_data: string indicating which data to get. Can be "train", "validation" or "test"
    :return: dict containing data sets stored in Pandas DataFrames and has the shape:
        {
            "input_features": Pandas DataFrame,
            "input_labels": "numpy array one-hot vector"
        }
    """

    data_set = {
        "input_features": [],
        "input_labels": []
    }

    if type_of_data == "test":
        data_set["input_features"] = processed_data["test_features_X"]
        data_set["input_labels"] = processed_data["test_y"]
        data_set["input_sequences"] = processed_data.get("test_sequences_X")
        data_set["chat_logs"] = processed_data.get("test_chat_logs")
    elif type_of_data == "validation":
        data_set["input_features"] = processed_data["validation_features_X"]
        data_set["input_labels"] = processed_data["validation_y"]
        data_set["input_sequences"] = processed_data.get("validation_sequences_X")
        data_set["chat_logs"] = processed_data.get("validation_chat_logs")
    elif type_of_data == "train":
        data_set["input_features"] = processed_data["train_features_X"]
        data_set["input_labels"] = processed_data["train_y"]
        data_set["input_sequences"] = processed_data.get("train_sequences_X")
        data_set["chat_logs"] = processed_data.get("train_chat_logs")
    else:
        raise ValueError("Invalid type of data: ", type_of_data)

    data_set["input_features"] = [np.array(data_set["input_features"])]
    if data_set["input_sequences"] is not None:
        data_set["input_features"].insert(0, data_set["input_sequences"])
    return data_set


def generate_predictions(
        input_features,
        model_name
):
    """
    Generate predictions
    :param input_features: Numpy array containing the features data records
    :param model_name:
    :return:
    """

    model = get_neural_network_model(MODEL_CONFIGS[model_name]["filename"])
    predictions = model.predict(input_features)

    return predictions


def get_prediction_labels(predictions, log_ids, predictions_index=-1):
    """
    Get prediction labels for each log_id.
    :param predictions: A list of zeros and ones representing model predictions
    :param log_ids: All log ids for the particular data slice
    :param predictions_index: The index to use for prediction values.
    Multi output models will have multiple predictions
    :return: a list of log ids with true predictions
    """
    def get_label(categorical_values):
        if categorical_values[0] > categorical_values[1]:
            return 0
        return 1

    predictions = predictions[predictions_index]
    prediction_labels = []
    for prediction, log_id in zip(predictions, log_ids):
        prediction_labels.append((str(log_id), get_label(prediction), prediction[0], prediction[1]))

    return prediction_labels


def update_chat_log_predictions(predictions_labels, data_type):
    logs_prediction_format = ", ".join([str(log) for log in predictions_labels])
    # print(logs_prediction_format)
    try:
        cursor.execute(
            u"INSERT INTO GNUeIRCLogs (log_id, prediction, categorical_value_1, categorical_value_2) "
            u"VALUES {} "
            u"ON DUPLICATE KEY UPDATE log_id=VALUES(log_id), "
            u"prediction=VALUES(prediction), "
            u"categorical_value_1=VALUES(categorical_value_1), "
            u"categorical_value_2=VALUES(categorical_value_2)".format(logs_prediction_format),
        )
        print("\n{} TOTAL CHAT LOG PREDICTIONS UPDATED FOR {} DATA\n".format(len(predictions_labels), data_type.upper()))
        db.commit()
    except MySQLdb.Error as error:
        print("ERROR: {}".format(error))
        db.rollback()
    except Exception as error:
        print(error)


def update_db_train_validation_and_test_dates(train_validation_and_test_dates):
    dates_tuples = []
    print(train_validation_and_test_dates)
    for data_type, dates in train_validation_and_test_dates.items():
        for date in dates:
            dates_tuples.append((date, data_type))

    dates_tuples = tuple(dates_tuples)
    dates_tuples_format = ", ".join([str(data_type) for data_type in dates_tuples])

    try:
        cursor.execute(
            u"INSERT INTO conversation_statistics_2 (conversation_date, data_type) "
            u"VALUES {} "
            u"ON DUPLICATE KEY UPDATE conversation_date=VALUES(conversation_date), "
            u"data_type=VALUES(data_type)".format(dates_tuples_format),
        )
        print("{} TOTAL DATA TYPES UPDATED\n".format(len(dates_tuples)))
        db.commit()
        return cursor.fetchone()
    except MySQLdb.Error as error:
        print("ERROR: {}".format(error))
        db.rollback()
    except Exception as error:
        print(error)


def update_conversation_statistics_by_date(date_of_logs):
    logs = get_logs_by_date(date_of_logs)

    number_of_summaries = number_of_true_predictions = number_of_sentences = 0
    for log in logs:
        number_of_true_predictions += log["prediction"]
        number_of_summaries += log["is_summary"]
        number_of_sentences += 1

    values = {
        "conversation_date": date_of_logs,
        "number_of_true_predictions": number_of_true_predictions,
        "number_of_summaries": number_of_summaries,
        "number_of_sentences": number_of_sentences,
    }

    print("\n\nVALUES: ", values)
    try:
        cursor.execute(
            u"INSERT INTO conversation_statistics_2 ("
            u"conversation_date, number_of_true_predictions, "
            u"number_of_summaries, number_of_sentences) "
            u"VALUES ('{conversation_date}', {number_of_true_predictions}, "
            u"{number_of_summaries}, {number_of_sentences}) "
            u"ON DUPLICATE KEY UPDATE "
            u"conversation_date=VALUES(conversation_date), "
            u"number_of_true_predictions=VALUES(number_of_true_predictions), "
            u"number_of_summaries=VALUES(number_of_summaries), "
            u"number_of_sentences=VALUES(number_of_sentences)".format(
                **values
            )
        )
        print("UPDATED CONVERSATION STATISTICS FOR DATE: ", date_of_logs)
        db.commit()
        return cursor.fetchone()
    except MySQLdb.Error as error:
        print("ERROR: {}".format(error))
        print(cursor.description)
        db.rollback()
    except Exception as error:
        print(error)


if __name__ == "__main__":
    # MODEL_NAME = "feed_forward"
    # MODEL_NAME = "hybrid_lstm_feed_forward"
    MODEL_NAME = "time_stepped_lstm"
    model_config = MODEL_CONFIGS[MODEL_NAME]
    INCLUDE_WORD_SEQUENCES = model_config["has_sequences"]
    processed_data, data_type_log_ids, train_validation_and_test_dates = get_processed_data_sets_for_model(
        include_word_sequences=INCLUDE_WORD_SEQUENCES,
        sequence_size=model_config["sequence_size"]
    )
    data_types_to_predict = [
        "train",
        "validation",
        "test"
    ]

    for data_type in data_types_to_predict:
        if MODEL_NAME != "time_stepped_lstm":
            processed_data_set = get_processed_data_set(processed_data, data_type)
            preds = generate_predictions(
                processed_data_set["input_features"],
                model_name=MODEL_NAME
            )
        else:
            sentence_vectors = keras_lstm.get_sentence_vectors(data_type)
            sentence_vectors = keras_lstm.create_time_steps(np.array(sentence_vectors), keras_lstm.time_steps)
            preds = generate_predictions(
                sentence_vectors,
                model_name=MODEL_NAME
            )

        if not INCLUDE_WORD_SEQUENCES:
            preds = [preds]

        print("\nNumber of Pred Outputs: ", len(preds))
        prediction_labels = get_prediction_labels(
            preds,
            data_type_log_ids[data_type],
            predictions_index=-1
        )
        test_actual_labels = np.argmax(processed_data[data_type + "_y"], axis=1)
        print(len([i for i in test_actual_labels if i == 1]))
        update_chat_log_predictions(prediction_labels, data_type)
        for date in train_validation_and_test_dates[data_type]:
            update_conversation_statistics_by_date(date)

    update_db_train_validation_and_test_dates(train_validation_and_test_dates)

    print("\nTrain Dates\n----------\n", train_validation_and_test_dates["train"])
