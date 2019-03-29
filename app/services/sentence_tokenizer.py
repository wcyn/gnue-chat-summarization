import linecache


from app.services.data_preparation import DATA_FILES_DIR
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from os.path import join

CHAT_LOGS_FILENAME = join(DATA_FILES_DIR, "gnue_irc_chat_logs_preprocessed.txt")


def _get_summarized_chat_logs(chat_logs_filename, chat_log_ids):
    """
    Gets actual chat logs from the main file. Filters out by log ids
    :param chat_logs_filename: Filename to obtain chat logs from
    :param chat_log_ids: IDs to filter the chat logs by
    :return: list of all the chats for the given log ids
    """
    chats = []
    for log_id in chat_log_ids:
        chats.append(linecache.getline(chat_logs_filename, log_id))
    return chats


# chat_logs = _get_summarized_chat_logs(CHAT_LOGS_FILENAME, [1, 2, 3, 4])
# print(chat_logs)


def _generate_chat_log_sequences(chat_logs):
    """
    Create sequences for the chat logs using the Keras Tokenizer object
    :param chat_logs: A list of chat logs to convert into sequences of numbers
    :return: A list of sequences representing the chat logs
    """
    tokenizer = Tokenizer(
        num_words=None,
        filters="\n",
        lower=False,
        split=' '
    )
    tokenizer.fit_on_texts(chat_logs)
    return tokenizer.texts_to_sequences(chat_logs)


def _pad_sequences(sequences):
    max_chat_length = 73
    return sequence.pad_sequences(sequences, maxlen=max_chat_length)


# sequences = _generate_chat_log_sequences(chat_logs)
# print(sequences)
#
# print(_pad_sequences(sequences))


def get_chat_log_sequences(log_ids):
    chat_logs = _get_summarized_chat_logs(CHAT_LOGS_FILENAME, log_ids)
    sequences = _generate_chat_log_sequences(chat_logs)
    return _pad_sequences(sequences)

