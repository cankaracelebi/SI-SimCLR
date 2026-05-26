"""
Authors: Wouter Van Gansbeke, Simon Vandenhende
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
Modifications by Elisa Messori: paths to astrocytes datasets added
"""
import os


class MyPath(object):
    @staticmethod
    def db_root_dir(database='', pairing_strategy=''):
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        local_annotations = os.path.join(project_root, 'data', 'crop16_EF_2405.csv')
        test_data = os.path.join(project_root, 'data', 'crop16_EF_2405_test.csv')
        
        database_paths = {
            'crop16_EF_2405': local_annotations,
            'crop16_EF_2405_adj_single': local_annotations,
            'debug': test_data
        }

        assert (database in database_paths.keys()), 'Database {} not implemented'.format(database) 

        return database_paths[database]

