import MySQLdb
import numpy as np

from app.database import database
from app.services.data_preparation import (
    DATA_FILES_DIR,
    get_processed_data_sets_for_model
)
from app.services.sentence_tokenizer import get_chat_log_sequences
from keras.models import load_model
from os.path import join

connection = database.DatabaseConnection()
db, cursor = connection.db, connection.cursor

MODEL_FILENAMES = {
    "hybrid_lstm_feed_forward": join(
        DATA_FILES_DIR, "models", "merged_hybrid_model.h5"
    )
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
    elif type_of_data == "validation":
        data_set["input_features"] = processed_data["validation_features_X"]
        data_set["input_labels"] = processed_data["validation_y"]
    elif type_of_data == "train":
        data_set["input_features"] = processed_data["train_features_X"]
        data_set["input_labels"] = processed_data["train_y"]
    else:
        raise ValueError("Invalid type of data: ", type_of_data)

    data_set["input_features"] = np.array(data_set["input_features"])
    return data_set


def generate_predictions(
        input_features,
        model_name="hybrid_lstm_feed_forward"
):
    """
    Generate predictions
    :param input_features: Numpy array containing the features data records
    :param model_name:
    :return:
    """

    model = get_neural_network_model(MODEL_FILENAMES[model_name])
    return model.predict(input_features)


def get_log_ids_for_true_predictions(predictions, log_ids, predictions_index=-1):
    """
    Get logs ids for sentences categorized as summaries.
    :param predictions: A list of zeros and ones representing model predictions
    :param log_ids: All log ids for the particular data slice
    :param predictions_index: The index to use for prediction values.
    Multi output models will have multiple predictions
    :return: a list of log ids with true predictions
    """
    predictions_argmax = np.argmax(predictions[predictions_index], axis=1)
    print(predictions_argmax)
    print(len(predictions_argmax))
    true_predictions_log_ids = {
        log_ids[index] for index, label in enumerate(predictions_argmax) if label == 1
    }
    return true_predictions_log_ids


def update_chat_log_predictions(true_predictions_log_ids, all_log_ids):
    logs_predictions_tuples = (
        (str(log), 1) if log in true_predictions_log_ids
        else (str(log), 0) for log in all_log_ids)
    logs_prediction_format = ", ".join([str(log) for log in logs_predictions_tuples])
    print(logs_prediction_format)

    if true_predictions_log_ids:
        try:
            cursor.execute(
                u"INSERT INTO GNUeIRCLogs (log_id, prediction) "
                u"VALUES {} "
                u"ON DUPLICATE KEY UPDATE log_id=VALUES(log_id), "
                u"prediction=VALUES(prediction)".format(logs_prediction_format),
            )
            print("UPDATED")
            db.commit()
            return cursor.fetchone()
        except MySQLdb.Error as error:
            print("ERROR: {}".format(error))
            db.rollback()
        except Exception as error:
            print(error)
    else:
        print("No logs to update predictions for")


if __name__ == "__main__":
    processed_data, data_type_log_ids, train_validation_and_test_dates = get_processed_data_sets_for_model()
    processed_data_set = get_processed_data_set(processed_data, "test")
    preds = generate_predictions(
        [get_chat_log_sequences(data_type_log_ids["test"]), processed_data_set["input_features"]]
    )
    log_ids_for_true_predictions = get_log_ids_for_true_predictions(preds, data_type_log_ids["test"])

    # print(log_ids_for_true_predictions)
    print(len(log_ids_for_true_predictions))

    actual_labels = np.argmax(processed_data["test_y"], axis=1)
    print(actual_labels, len(actual_labels))
    print(len([i for i in actual_labels if i == 1]))
    print("Dates:\n-----\n", train_validation_and_test_dates)

    update_chat_log_predictions(log_ids_for_true_predictions, data_type_log_ids["test"])
