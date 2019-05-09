import csv
import os
import time
import multiprocessing
import numpy as np

start_imports = time.time()

from app.services import keras_lstm_features, data_preparation
from app.services.paths import CHAT_DATA_TYPE_FILES
from keras.engine.saving import load_model
from os.path import join
from sklearn.metrics import classification_report, confusion_matrix
from neural_net_models.rouge_metrics import get_rouge_results


BASE_MODELS_DIR = join(CHAT_DATA_TYPE_FILES, "models")

processed_data, data_type_log_ids, train_validation_and_test_dates = data_preparation.get_processed_data_sets_for_model(
    include_word_sequences=True,
    sequence_size=73
)
(
    train_sentence_vectors,
    validation_sentence_vectors,
    test_sentence_vectors
) = keras_lstm_features.load_sentence_vectors()

MODEL_FIELDS = [
    "Model Name", "Class 0 Precision", "Class 0 Recall", "Class 0 F1 Score",
    "Class 1 Precision", "Class 1 Recall", "Class 1 F1 Score", "Confusion Matrix",
    "Rouge 1 - Precision", "Rouge 1 - Recall",  "Rouge 1 - F1 Score",
    "Rouge 2 - Precision", "Rouge 2 - Recall",  "Rouge 2 - F1 Score",
    "Rouge L - Precision", "Rouge L - Recall",  "Rouge L - F1 Score"
]
RESULTS_FILENAME = join(CHAT_DATA_TYPE_FILES, "results", "results.csv")
RESULTS_FILENAME_2 = join(CHAT_DATA_TYPE_FILES, "results", "results_2.csv")

test_sequences_X = processed_data["test_sequences_X"]
test_features_X = processed_data["test_features_X"].values
test_chat_logs = processed_data["test_chat_logs"]


test_y = processed_data["test_y"]
test_y_argmax = np.argmax(test_y, axis=1)

test_sentence_vectors_features_concat = np.concatenate((test_sentence_vectors, test_features_X), axis=1)

MODELS_FOLDERS = {
    # "30_ts_0.5_dro_lstm": {
    #     "inputs": keras_lstm_features.create_time_steps(test_sentence_vectors, 30)
    # },
    # "30_ts_bidi_lstm_hybrid": {
    #     "inputs": [keras_lstm_features.create_time_steps(test_sentence_vectors, 30), test_features_X]
    # },
    # "30_ts_lstm_hybrid": {
    #     "inputs": [keras_lstm_features.create_time_steps(test_sentence_vectors, 30), test_features_X]
    # },
    # "30_ts_lstm_sequences_features": {
    #     "inputs": keras_lstm_features.create_time_steps(test_sentence_vectors_features_concat, 30)
    # },
    # "30ts_0.5dro_lstm2": {
    #     "inputs": keras_lstm_features.create_time_steps(test_sentence_vectors, 30)
    # },
    # "50_ts_bidi_lstm_features": {
    #     "inputs": keras_lstm_features.create_time_steps(test_features_X, 50)
    # },
    # "50_ts_lstm_features": {
    #     "inputs": keras_lstm_features.create_time_steps(test_features_X, 50)
    # },
    # "100_ts_05_dro_32_hino_lstm2": {
    #     "inputs": keras_lstm_features.create_time_steps(test_sentence_vectors, 100)
    # },
    "100_ts_05_dro_lstm2": {
        "inputs": keras_lstm_features.create_time_steps(test_sentence_vectors, 100)
    }
}

end_data = time.time()

print("Pre Processes: ", end_data - start_imports, " seconds")


def get_model_predictions(model, input_values):
    if isinstance(input_values, list):
        predictions = np.array(model.predict(input_values))
    else:
        predictions = np.array(model.predict([input_values]))
        model = None
    if len(predictions.shape) == 2:
        predictions = [predictions]
    return predictions


def get_predictions_argmax(predictions):
    return np.argmax(predictions[-1], axis=1)


def format_model_results(model_directory, model_name, classification_report_results, confusion_matrix_results, rouge_scores):
    zero_results = classification_report_results["0"]
    one_results = classification_report_results["1"]
    rouge_1 = rouge_scores["rouge-1"]
    rouge_2 = rouge_scores["rouge-2"]
    rouge_l = rouge_scores["rouge-l"]
    model_identifier = join(model_directory, model_name)
    values = [
        model_identifier,
        zero_results["precision"], zero_results["recall"], zero_results["f1-score"],
        one_results["precision"], one_results["recall"], one_results["f1-score"],
        confusion_matrix_results,
        rouge_1["p"], rouge_1["r"], rouge_1["f"],
        rouge_2["p"], rouge_2["r"], rouge_2["f"],
        rouge_l["p"], rouge_l["r"], rouge_l["f"],
    ]
    assert len(MODEL_FIELDS) == len(values)
    return values


def write_model_fields_to_file():
    if not os.path.isfile(RESULTS_FILENAME):
        with open(RESULTS_FILENAME, "w") as results_file:
            csv_file = csv.writer(results_file)
            csv_file.writerow(MODEL_FIELDS)


def append_model_results_to_file(model_results):
    with open(RESULTS_FILENAME, "a") as results_file:
        csv_file = csv.writer(results_file)
        csv_file.writerow(model_results)


def get_rouge_scores(y_actual_argmax, y_predictions_argmax):
    predicted_chat_logs = [test_chat_logs[index] for index, value in enumerate(y_predictions_argmax) if value == 1]
    summaries_chat_logs = [test_chat_logs[index] for index, value in enumerate(y_actual_argmax) if value == 1]
    predicted_chat_logs = "".join(log for log in predicted_chat_logs)
    summaries_chat_logs = "".join(log for log in summaries_chat_logs)
    hypotheses = [predicted_chat_logs]
    references = [summaries_chat_logs]
    return get_rouge_results(hypotheses, references)


def load_model_files(models_folders_dict):
    write_model_fields_to_file()
    for model_folder, model_details in models_folders_dict.items():
        get_model_and_process_results(model_folder, MODELS_FOLDERS)


def get_model_epoch(model_filename):
    epoch = model_filename.split(".")[0].split("-")
    if len(epoch) > 1 and epoch[1].isdigit():
        return int(epoch[1])
    return -1


def get_model_and_process_results(model_folder, model_dict):

    model_dir = join(BASE_MODELS_DIR, model_folder)
    print("\nCurrent directory: ", model_dir)

    model_files = sorted(os.listdir(model_dir))
    LAST_MODEL_INDEX =  model_files.index("model-95.hdf5")
    model_files = model_files[LAST_MODEL_INDEX+1:]

    for model_filename in model_files:
        if model_filename.endswith(".hdf5"):
            epoch = get_model_epoch(model_filename)
            if epoch > 100:
                continue

            start = time.time()
            model = load_model(join(model_dir, model_filename))
            end = time.time()
            print(end - start, " seconds")
            y_predictions = get_model_predictions(model, model_dict[model_folder]["inputs"])
            y_predictions_argmax = get_predictions_argmax(y_predictions)
            _, rouge_scores = get_rouge_scores(test_y_argmax, y_predictions_argmax)
            classification_report_results = classification_report(
                test_y_argmax, y_predictions_argmax, output_dict=True)
            confusion_matrix_results = confusion_matrix(test_y_argmax, y_predictions_argmax)
            formatted_model_results = format_model_results(
                model_folder, model_filename, classification_report_results,
                confusion_matrix_results, rouge_scores)
            append_model_results_to_file(formatted_model_results)
            print("\tModel: ", model_filename, "\n")


def transfer_results_with_epoochs():
    with open(RESULTS_FILENAME, "r") as results_file, open(RESULTS_FILENAME_2, "w") as results_file_2:
        csv_file = csv.reader(results_file)
        results_2_csv = csv.writer(results_file_2)
        # next(csv_file)  # skip headers
        headers = next(csv_file)
        headers.insert(1, "Epochs")
        headers.insert(1, "Model Folder")
        results_2_csv.writerow(headers)
        for row in csv_file:
            model_name = row[0]
            model_folder, model_filename = model_name.split("/")
            epoch = get_model_epoch(model_filename)
            row.insert(1, epoch)
            row.insert(1, model_folder)
            results_2_csv.writerow(row)


if __name__ == "__main__":
    # load_model_files(MODELS_FOLDERS)
    # get_model_and_process_results("100_ts_05_dro_lstm2", MODELS_FOLDERS)
    pass
    transfer_results_with_epoochs()
