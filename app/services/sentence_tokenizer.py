import linecache

from app.services.paths import CHAT_LOGS_FILENAME
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence

MAX_CHAT_LENGTH = 100
TOP_WORDS = 10000


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
        num_words=TOP_WORDS,
        filters="\n",
        lower=False,
        split=' '
    )
    tokenizer.fit_on_texts(chat_logs)
    return tokenizer.texts_to_sequences(chat_logs)


def _pad_sequences(sequences, sequence_size):
    return sequence.pad_sequences(sequences, maxlen=sequence_size)


# sequences = _generate_chat_log_sequences(chat_logs)
# print(sequences)
#
# print(_pad_sequences(sequences))


def get_chat_log_sequences_and_chat_logs(log_ids, sequence_size=MAX_CHAT_LENGTH):
    """
    Get chat log sequences and chat logs given log ids
    :param sequence_size: MAXIMUM size of the sequence
    :param log_ids: list of log ids
    :return: dict containing chat data in the shape:
        {
        "sequences": [123, 21, 35, 4],
        "chat_logs": ["hello", "hi mark"]
    }
    """
    chat_logs = get_chat_logs(log_ids)
    sequences = _generate_chat_log_sequences(chat_logs)
    chat_data = {
        "sequences": _pad_sequences(sequences, sequence_size),
        "chat_logs": chat_logs
    }
    return chat_data


def get_chat_logs(log_ids):
    return _get_summarized_chat_logs(CHAT_LOGS_FILENAME, log_ids)
