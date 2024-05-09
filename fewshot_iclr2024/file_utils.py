# Copyright 2024 Lawrence Livermore National Security, LLC and other
# LUAR Project Developers. 
#
# SPDX-License-Identifier: Apache-2.0

import configparser
import os
from datetime import datetime

def singleton(cls):
    """Decorator for defining a Singleton class.
    """
    instance = [None]

    def wrapper(*args, **kwargs):
        if instance[0] is None:
            instance[0] = cls(*args, **kwargs)
        return instance[0]

    return wrapper()

@singleton
class Utils(object):
    """Keeps track of all the paths required by the application and provides 
       basic utilities for reading JSON files.
    """
    def __init__(self):

        # Read project file configuration
        self._config_dir = "/".join(os.path.realpath(__file__).split("/")[:-1])
        self.config_filename = os.path.join(
            self._config_dir, "file_config.ini")
        self.config = configparser.ConfigParser()
        self.config.read(self.config_filename)

        # Set all the paths and create an experiment name
        self.set_paths()

    @property
    def config_dir(self):
        return self._config_dir

    @property
    def project_path(self):
        return self._project_path

    @property
    def aac_data_path(self):
        return self._aac_data_path

    @property
    def lwd_data_path(self):
        return self._lwd_data_path

    @property
    def fewshot_data_path(self):
        return self._fewshot_data_path

    @property
    def weights_path(self):
        return self._weights_path


    def create_project_name(self):
        """Creates a project name.
        """
        name = datetime.now().strftime("Experiment_%B_%Y")

        return name

    def set_paths(self):
        """Sets all of the project path variables that are relevant for reading pre-trained models, 
           saving output, etc.
        """
        Experiment_Paths = self.config["Experiment_Paths"]

        self._project_path = Experiment_Paths["project_root_path"]
        self._aac_data_path = Experiment_Paths["aac_data_path"]
        self._lwd_data_path = Experiment_Paths["lwd_data_path"]
        self._fewshot_data_path = Experiment_Paths["fewshot_data_path"]
        self._weights_path = Experiment_Paths["weights_path"]

        self.path_exists(self._aac_data_path)
        self.path_exists(self._lwd_data_path)
        self.path_exists(self._fewshot_data_path)
        self.path_exists(self._weights_path)


    def path_exists(self, path: str):
        """Checks if a path exist. If it does, it will return it.

        Args:
            path (str): Path to create or check existence for.
            If path doesn't exist, creates path.
        """
        if os.path.exists(path) == False:
            os.makedirs(path)
            