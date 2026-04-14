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
        
        database_paths = {
            #'ctrl_allcond_both_variant': 'YOUR/PATH/control_untreated_A1low_A1high_all_11_DAPI_GFAP/annotations_completepath.csv',
            #'crop16_ac_complete': 'YOUR/PATH/fluoACS_crop16/control_als_untreated_A1low_A1high_all_11_DAPI_GFAP/annotations_completepath.csv',
            #'crop4_ac_replicate': 'YOUR/PATH/fluoACS_replicate_crop4/control_als_untreated_A1low_A1high_all_11_DAPI_GFAP/annotations_completepath.csv',
            #'crop16_ac_replicate': 'YOUR/PATH/fluoACS_replicate_crop16/control_als_untreated_A1low_A1high_all_11_DAPI_GFAP/annotations_completepath.csv',
            #'completeacs_unsplitted': 'YOUR/PATH/control_als_untreated_A1low_A1high_all_0_DAPI_GFAP/annotations_completepath.csv',
            #'crop4_EF_2405': 'YOUR/PATH/fluoacs_data/original/control_als_untreated_A1low_A1high_all_11_DAPI_GFAP/annotations_HDBSCAN_filter_nosingle.csv',
            'crop16_EF_2405': local_annotations,
            'crop16_EF_2405_adj_single': local_annotations
        }

        assert (database in database_paths.keys()), 'Database {} not implemented'.format(database) 

        return database_paths[database]

