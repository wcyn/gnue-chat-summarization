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
    ),
    "feed_forward": join(
        DATA_FILES_DIR, "models", "04-02_963_1000_64_adam_5min_ff_neural_net.h5"
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
    predictions_argmax = np.argmax(predictions, axis=1)
    print(predictions_argmax)
    true_predictions_log_ids = {
        log_ids[index] for index, label in enumerate(predictions_argmax) if label == 1
    }
    print(true_predictions_log_ids)

    prediction_labels = []
    for prediction, log_id in zip(predictions, log_ids):
        prediction_labels.append((str(log_id), get_label(prediction), prediction[0], prediction[1]))

    return prediction_labels


def update_chat_log_predictions(predictions_labels, data_type_log_ids, predicted_type="test"):
    # logs_predictions_tuples = []
    other_log_ids = []
    for data_type, log_ids in data_type_log_ids.items():
        if data_type != predicted_type:
            other_log_ids.extend(log_ids)

    logs_predictions_tuples = (
        (str(log), 0, 0, 0) for log in other_log_ids)
    predictions_labels += logs_predictions_tuples
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
        print("{} TOTAL CHAT LOG PREDICTIONS UPDATED".format(len(predictions_labels)))
        db.commit()
        return cursor.fetchone()
    except MySQLdb.Error as error:
        print("ERROR: {}".format(error))
        db.rollback()
    except Exception as error:
        print(error)


if __name__ == "__main__":
    processed_data, data_type_log_ids, train_validation_and_test_dates = get_processed_data_sets_for_model()
    processed_data_set = get_processed_data_set(processed_data, "test")
    # preds = generate_predictions(
    #     [get_chat_log_sequences(data_type_log_ids["test"]), processed_data_set["input_features"]]
    # )
    preds = generate_predictions(
        [processed_data_set["input_features"]], model_name="feed_forward"
    )
    # print(preds)
    prediction_labels = get_prediction_labels(
        [preds],
        data_type_log_ids["test"],
        predictions_index=-1
    )
    # all_log_ids = data_type_log_ids["test"] + data_type_log_ids["train"] + data_type_log_ids["validation"]

    # print(log_ids_for_true_predictions)
    # print(len(log_ids_for_true_predictions))

    actual_labels = np.argmax(processed_data["test_y"], axis=1)
    print(len([i for i in actual_labels if i == 1]))
    print("\nTrain Dates\n----------\n", train_validation_and_test_dates["train"])

    update_chat_log_predictions(prediction_labels, data_type_log_ids)
