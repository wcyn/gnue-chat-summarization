# from keras.preprocessing.text import Tokenizer

from os.path import join

FEATURES_DIR = join("..", "feature_extraction", "feature_outputs")
DATA_FILES_DIR = join("..", "feature_extraction", "data_files")
CHAT_LOGS_FILENAME = join(DATA_FILES_DIR, "gnue_irc_chat_logs_preprocessed.txt")
SUMMARIZED_CHAT_LOG_IDS_FILENAME = join(DATA_FILES_DIR, "summarized_chat_log_ids.csv")

# tokenizer_object = Tokenizer(
#         num_words=None,
#         filters="\n",
#         lower=False,
#         split=' '
#     )


def get_summarized_chat_log_ids(log_ids_filename):
    summarized_chat_log_ids = []
    with open(log_ids_filename) as log_ids_file:
        for log_id in log_ids_file:
            log_id = log_id.strip().split(",")[0]
            if log_id:
                summarized_chat_log_ids.append(int(log_id))
            else:
                print("Yay", log_id)
    return summarized_chat_log_ids


def get_summarized_chat_logs(chat_logs_filename, summarized_chat_log_ids):
    summarized_chat_log_ids = set(summarized_chat_log_ids)
    chats = {}
    with open(chat_logs_filename) as chat_logs:
        line_number = 1
        chat_log_count = 1
        for chat_log in chat_logs:
            if line_number in summarized_chat_log_ids:
                chats[chat_log_count] = chat_log
                chat_log_count += 1
            line_number += 1
    return chats


SUMMARIZED_CHAT_LOG_IDS = get_summarized_chat_log_ids(SUMMARIZED_CHAT_LOG_IDS_FILENAME)
CHAT_LOGS = get_summarized_chat_logs(CHAT_LOGS_FILENAME, SUMMARIZED_CHAT_LOG_IDS)

# print(SUMMARIZED_CHAT_LOG_IDS)
# print(CHAT_LOGS)


def get_tokenized_sequences(chat_logs, tokenizer):
    # Create a Tokenizer Object
    tokenizer.fit_on_texts(chat_logs)


def get_sentences_of_line_numbers(chat_logs, line_numbers):
    return "".join(chat_logs[line_number] for line_number in line_numbers)

# print(len(SUMMARIZED_CHAT_LOG_IDS))
# print(max(CHAT_LOGS.keys()))
# print(get_sentences_of_line_numbers(CHAT_LOGS, [
# 20086, 20106, 20107, 20110, 20137, 20139, 20167, 20203, 20212, 20249]))
