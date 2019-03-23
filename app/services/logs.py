import random
import MySQLdb
import numpy as np

from app.database import database
from keras.models import load_model
from os.path import join

connection = database.DatabaseConnection()
db, cursor = connection.db, connection.cursor
DATA_FILES_DIR = join("..", "feature_extraction", "data_files")
MODEL_FILENAMES = {
    "hybrid_lstm_feed_forward": join(
        DATA_FILES_DIR, "models", "merged_hybrid_model.h5"
    )
}
LOG_IDS_FILENAME = join(
    DATA_FILES_DIR, "summarized_chat_log_ids.csv"
)


def get_logs_by_date(date_of_log):
    try:
        cursor.execute(
            u"SELECT log_id, line_message, is_summary, send_user, "
            u"SUM(is_summary) OVER () AS summary_sum "
            u"FROM GNUeIRCLogs "
            u"WHERE date_of_log=%s ", (date_of_log,)
        )
        return cursor.fetchall()
    except MySQLdb.Error as error:
        print("ERROR: {}".format(error))
        db.rollback()


def get_log_ids_and_messages_by_date(date_of_log):
    try:
        cursor.execute(
            u"SELECT log_id, line_message "
            u"FROM GNUeIRCLogs "
            u"WHERE date_of_log=%s ", (date_of_log,)
        )
        return cursor.fetchall()
    except MySQLdb.Error as error:
        print("ERROR: {}".format(error))
        db.rollback()


def get_log_by_id(log_id):
    try:
        cursor.execute(
            u"SELECT log_id, line_message, is_summary, date_of_log, send_user FROM GNUeIRCLogs "
            u"WHERE log_id=%s", (log_id,)
        )
        return cursor.fetchone()
    except MySQLdb.Error as error:
        print("ERROR: {}".format(error))
        db.rollback()
    except Exception as error:
        print(error)


def update_log_by_id(log_id, is_summary):
    try:
        cursor.execute(
            u"UPDATE GNUeIRCLogs "
            u"SET is_summary=%s WHERE log_id=%s", (is_summary, log_id)
        )
        print("UPDATED")
        db.commit()
        return cursor.fetchone()
    except MySQLdb.Error as error:
        print("ERROR: {}".format(error))
        db.rollback()
    except Exception as error:
        print(error)


def get_username_colors(usernames):
    colors = {username: generate_random_color() for username in usernames}
    return colors


def generate_random_color():
    colors = [
        # French
        "#fad390", "#f8c291", "#6a89cc", "#82ccdd", "#b8e994",
        "#f6b93b", "#e55039", "#4a69bd", "#60a3bc", "#78e08f",
        "#fa983a", "#eb2f06", "#1e3799", "#3c6382", "#38ada9",
        "#e58e26", "#b71540", "#0c2461", "#0a3d62", "#079992",
        # Dutch
        '#FFC312', '#C4E538', '#12CBC4', '#FDA7DF', '#ED4C67',
        '#F79F1F', '#A3CB38', '#1289A7', '#D980FA', '#B53471',
        '#EE5A24', '#009432', '#0652DD', '#9980FA', '#833471',
        '#EA2027', '#006266', '#1B1464', '#5758BB', '#6F1E51',
        # Flat UI
        '#1abc9c', '#2ecc71', '#3498db', '#9b59b6', '#34495e',
        '#16a085', '#27ae60', '#2980b9', '#8e44ad', '#2c3e50',
        '#f1c40f', '#e67e22', '#e74c3c', '#f39c12', '#d35400',
        '#c0392b',
        # Spanish
        '#40407a', '#2c2c54', '#ff5252', '#b33939', '#706fd3',
        '#474787', '#ff793f', '#cd6133', '#34ace0', '#227093',
        '#ffb142', '#33d9b2', '#218c74', '#ffda79',
        # Swedish
        '#ef5777', '#575fcf', '#4bcffa', '#34e7e4', '#0be881',
        '#f53b57', '#3c40c6', '#0fbcf9', '#00d8d6', '#05c46b',
        '#ffc048', '#ffdd59', '#ff5e57', '#ffa801', '#ffd32a',
        '#ff3f34'
    ]
    return random.choice(colors)


def update_log_message_summaries(date_of_log, logs, summary_log_ids):
    logs_summary_tuples = ((str(log['log_id']), 1) if str(log['log_id']) in summary_log_ids
                           else (str(log['log_id']), 0) for log in logs)
    logs_summary_format = ", ".join([str(log) for log in logs_summary_tuples])

    if logs:
        try:
            cursor.execute(
                u"INSERT INTO GNUeIRCLogs (log_id, is_summary) "
                u"VALUES {} "
                u"ON DUPLICATE KEY UPDATE log_id=VALUES(log_id), "
                u"is_summary=VALUES(is_summary)"
                .format(logs_summary_format),
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
        print("No log messages available for: ", date_of_log)


def get_summary_and_quotes_by_date(date):
    try:
        cursor.execute(
            u"SELECT para, "
            u"GROUP_CONCAT(GNUeSummaryParaQuotes.quote ORDER BY paraquoteid ASC SEPARATOR ' ^&##m*_^-> ') AS quotes "
            u"FROM GNUeSummaryPara INNER JOIN GNUeSummaryParaQuotes "
            u"ON GNUeSummaryPara.paraid = GNUeSummaryParaQuotes.paraid "
            u"WHERE GNUeSummaryParaQuotes.quote_date=%s "
            u"GROUP BY para, GNUeSummaryPara.paraid "
            u"ORDER BY GNUeSummaryPara.paraid ASC", (date,)
        )
        return cursor.fetchall()
    except MySQLdb.Error as error:
        print("ERROR: {}".format(error))
        db.rollback()


def get_neural_network_model(model_filename):
    return load_model(model_filename)


def predict_summaries(model, input_list):
    """
    Give predictions using the model based on the  input list
    :param model: The model to use for the prediction
    :param input_list: A list of the input values. Should contain as many items as there are types of inputs ie.
    multiple items for multi-input models
    :return: y predictions predicted by the model.
    """
    return model.predict(input_list)


def get_log_ids_for_true_predictions(predictions, line_number_offset, log_ids_filename, predictions_index=-1):
    """
    Get logs ids for sentences categorized as summaries.
    :param predictions: A list of zeros and ones representing model predictions
    :param line_number_offset: index from where to get the chat logs
    :param log_ids: Filename to get all log ids from
    :param predictions_index: The index to use for prediction values.
    Multi output models will have multiple predictions
    :return: a list of log id with true predictions
    """
    predictions_argmax = np.argmax(predictions[predictions_index], axis=1)
    true_predictions_line_numbers = [
        line_number_offset + index + 1 for index, value in enumerate(predictions_argmax) if value == 1
    ]

    log_ids = []
    with open(log_ids_filename) as log_ids_file:
        for line_number in true_predictions_line_numbers:
            log_ids_file.seek(line_number)
            log_ids.append(log_ids_file.readline().strip())
    return log_ids


def update_chat_log_predictions_by_date(logs, date_of_log, true_predictions_log_ids):
    logs_predictions_tuples = (
        (str(log['log_id']), 1) if str(log['log_id']) in true_predictions_log_ids
        else (str(log['log_id']), 0) for log in logs)
    logs_prediction_format = ", ".join([str(log) for log in logs_predictions_tuples])

    if logs:
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
        print("No log messages available for: ", date_of_log)


def generate_and_update_predictions(
        logs_dates, line_number_offset, log_ids_filename, model_name="hybrid_lstm_feed_forward"):
    for logs_date in logs_dates:
        # log_ids = get_log_ids_and_messages_by_date(logs_date)
        model = get_neural_network_model(MODEL_FILENAMES[model_name])
        input_list = ()
        predictions = predict_summaries(model, input_list)
        true_predictions_log_ids = get_log_ids_for_true_predictions(
            predictions, line_number_offset, log_ids_filename, predictions_index=0
        )
