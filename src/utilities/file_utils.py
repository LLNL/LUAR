# Copyright 2023 Lawrence Livermore National Security, LLC and other
# LUAR Project Developers. 
#
# SPDX-License-Identifier: Apache-2.0

import configparser
import os
from datetime import datetime

from tqdm import tqdm 

from utilities import decorators as decorator

@decorator.singleton
class Utils(object):
    """Keeps track of all the paths required by the application and provides 
       basic utilities for reading JSON files.
    """
    def __init__(self):

        # Read project file configuration
        self._config_dir = '/'.join(os.path.realpath(__file__).split('/')[:-3])
        self.config_filename = os.path.join(
            self._config_dir, 'file_config.ini')
        self.config = configparser.ConfigParser()
        self.config.read(self.config_filename)

        # Set all the paths and create an experiment name
        self.set_paths()
        self._project_name = self.create_project_name()

    @property
    def config_dir(self):
        return self._config_dir

    @property
    def project_path(self):
        return self._project_path

    @property
    def output_path(self):
        return self._output_path

    @property
    def data_path(self):
        return self._data_path

    @property
    def transformer_path(self):
        return self._transformer_path

    @property
    def embedding_path(self):
        return self._embeddings_path

    @property
    def query_path(self):
        return self._query_path

    def create_project_name(self):
        """Creates a project name.
        """
        name = datetime.now().strftime('Experiment_%B_%Y')

        return name

    def set_paths(self):
        """Sets all of the project path variables that are relevant for reading pre-trained models, 
           saving output, etc.
        """
        Experiment_Paths = self.config['Experiment_Paths']

        self._project_path = Experiment_Paths['project_root_path']
        self._output_path = Experiment_Paths['output_path']
        self._data_path = Experiment_Paths['data_path']
        self._transformer_path = Experiment_Paths['transformer_path']

        self.path_exists(self._output_path, create_path=True)
        self.path_exists(self._data_path, create_path=True)
        self.path_exists(self._transformer_path, create_path=True)

    def path_exists(self, path: str, create_path=False):
        """Checks if a path exist. If it does, it will return it.

        Args:
            path (str): Path to create or check existence for.
            create_path (bool, optional): If True, will create the path if it doesn't exist. 
                Defaults to False.
        """
        if os.path.exists(path) == True:
            return path
        
        if os.path.exists(path) is False and create_path == True:
            os.makedirs(path)
            
            return path
        else:
            raise ValueError("The following path doesn\'t exist: {}".format(path))
    
    def dict2string(self, dictionary: dict, title: str):
        """Turns a dictionary into something pretty to print or write.

        Args:
            dictionary (dict): Dictionary with the data.
            title (str): Pretty title to give the collection of data.
        """
        message = '\n\t'+title+'\n'
        info = ['\t\t' + str(key)+'=' + str(value) + '\n' for key,
                value in list(dictionary.items())]
        info_str = ''.join(info)
        message += info_str

        return message