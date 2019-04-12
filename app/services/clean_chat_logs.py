import gzip
import re
import string

from app.services.data_preparation import CHAT_DATA_TYPE_FILES
from string import printable
from os.path import join

PUNCTUATION_SET = set(string.punctuation)
PRINTABLE = set(printable)


def remove_non_printable_characters_from_string(chars):
    return ''.join(char for char in chars if char in PRINTABLE)


def strip_leading_and_trailing_punctuation(word):
    return word.strip(string.punctuation)


def all_chars_in_word_are_punctuation(word):
    return all(char in PUNCTUATION_SET for char in word)


def pre_process_sentence(sentence):
    if not sentence:
        print("No words here")
    if type(sentence) is not str:
        try:
            sentence = sentence.decode('utf-8')
        except Exception:
            raise ValueError("Input must be a String or ByteString")

    return sentence


def get_words_in_sentence(sentence):
    sentence = pre_process_sentence(sentence)
    words = []
    for word in sentence.split():
        if not all_chars_in_word_are_punctuation(word):
            word = strip_leading_and_trailing_punctuation(word)
            if ',' in word:
                comma_split_words = word.split(',')
                for word in comma_split_words:
                    if not all_chars_in_word_are_punctuation(word):
                        word = strip_leading_and_trailing_punctuation(word)
                        words.append(word)
            else:
                words.append(word)
    return words


def preprocess_chat_log_file(input_file, output_file):
    with open(input_file) as in_file, open(output_file, "w") as out_file:
        for chat_line in in_file:
            chat_line = pre_process_sentence(chat_line)
            chat_line = remove_non_printable_characters_from_string(chat_line)
            # print(chat_line)
            out_file.write(chat_line)


def preprocess_chat_log_file_words_only(input_file, output_file):
    with open(input_file) as in_file, open(output_file, "w") as out_file:
        for chat_line in in_file:
            chat_line = pre_process_sentence(chat_line)
            chat_line = remove_non_printable_characters_from_string(chat_line)
            words = get_words_in_sentence(chat_line)
            out_file.write("{}\n".format(" ".join(words)))


def clean_chat_log_file(chat_log_input_filename, chat_log_output_filename, words_only=False):
    if words_only:
        preprocess_chat_log_file_words_only(chat_log_input_filename, chat_log_output_filename)
    else:
        preprocess_chat_log_file(chat_log_input_filename, chat_log_output_filename)


if __name__ == "__main__":
    data_types = ["train", "validation", "test"]

    for data_type in data_types:
        chat_log_input_filename = join(CHAT_DATA_TYPE_FILES, data_type + ".txt")
        chat_log_output_filename = join(CHAT_DATA_TYPE_FILES, data_type + "-processed.txt")
        clean_chat_log_file(chat_log_input_filename, chat_log_output_filename, words_only=False)
