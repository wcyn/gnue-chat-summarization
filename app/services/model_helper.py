import json
import os
import time
from os.path import join

from keras.callbacks import ModelCheckpoint

from app.services.paths import DATA_FILES_DIR

TIME_OFFSET = 1554206530


class ModelHelper:
    def __init__(self):
        self.time_diff = None
        self.checkpoints_dir = None
        self.model_details_filename = "model_details.json"

    def get_time_diff(self):
        time_now = time.time()
        self.time_diff = int(time_now-TIME_OFFSET)
        return self.time_diff

    def _get_checkpoints_dir(self, epochs, batch_size, time_diff, optimizer):
        self.checkpoints_dir = join(
            DATA_FILES_DIR,
            "{}_checkpoints_hybrid_{}e_{}b_{}".format(time_diff, epochs, batch_size, optimizer)
        )
        return self.checkpoints_dir

    def _validate_checkpoints_dir(self):
        if not self.checkpoints_dir:
            raise ValueError("Checkpoint Directory not set")

    def _create_checkpoints_dir(self):
        self._validate_checkpoints_dir()
        os.mkdir(self.checkpoints_dir)

    def _get_checkpointer(self):
        self._validate_checkpoints_dir()
        return ModelCheckpoint(
            filepath=self.checkpoints_dir + '/model-{epoch:02d}.hdf5',
            verbose=1,
            period=10
        )

    def fetch_checkpointer(self, epochs, batch_size, optimizer):
        time_diff = self.get_time_diff()
        checkpoints_dir = self._get_checkpoints_dir(epochs, batch_size, time_diff, optimizer)
        self._create_checkpoints_dir()
        return self._get_checkpointer()

    def save_model_details(self, model_details):
        """

        :type model_details: dict containing details about the model
        """
        model_details_filename = join(self.checkpoints_dir, self.model_details_filename)
        with open(model_details_filename, "w") as model_details_file:
            model_details_file.write(json.dumps(model_details, indent=2, sort_keys=True))

    def load_model_details(self, checkpoints_dir=None, model_details_filename=None):
        if not checkpoints_dir:
            checkpoints_dir = self.checkpoints_dir
        if not model_details_filename:
            model_details_filename = self.model_details_filename

        model_details_filename = join(checkpoints_dir, model_details_filename)

        with open(model_details_filename) as model_details_file:
            data = "".join(line.strip() for line in model_details_file.readlines())
            model_details = json.loads(data)
        assert isinstance(model_details, dict)
        return model_details


# m = ModelHelper()
# m_data = m.load_model_details(
#     "/Users/wcyn/venv-projects/gnue-irc/GNUeIRC/feature_extraction/data_files/1132802_checkpoints_hybrid_3e_128b_rmsprop/model_details.json")
# print(m_data)
# print(type(m_data))


"""
Elapsed Time: 0:16:42.478337
Total Dirty Plan identified: 221
Total Plans processed: 221
PlanSummaries updated: 221
PlanSkills updated: 8219
Plans Topic Performance Healed: 221
Teams updated: 3338
Users updated: 89
UserSkills processed: 391
UserSkills updated: 143
UserSkills updated msgs sent: 89
"""
