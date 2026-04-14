"""
Authors: Wouter Van Gansbeke, Simon Vandenhende
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
Modifications by Elisa Messori: added the item base_dir to the EasyDict object.
"""
import os
import shutil
import yaml
from easydict import EasyDict
import errno
#from utils.utils import mkdir_if_missing


def mkdir_if_missing(directory):
    if not os.path.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise


def load_env_paths(config_file_env):
    """Load and resolve path settings from env.yml without side effects."""
    with open(config_file_env, 'r') as stream:
        env_cfg = yaml.safe_load(stream) or {}

    raw_root_dir = env_cfg.get('root_dir', '..')
    root_dir = os.path.abspath(os.path.join(os.path.dirname(config_file_env), raw_root_dir))

    def _resolve_path(raw_path, default_relative=None):
        if raw_path is None:
            raw_path = default_relative
        if raw_path is None:
            return None
        if os.path.isabs(raw_path):
            return raw_path
        return os.path.abspath(os.path.join(root_dir, raw_path))

    results_root_dir = _resolve_path(env_cfg.get('results_root_dir'), 'results')

    return {
        'root_dir': root_dir,
        'image_root_dir': _resolve_path(env_cfg.get('image_root_dir')),
        'embeddings_root_dir': _resolve_path(env_cfg.get('embeddings_root_dir'), 'embeddings'),
        'results_root_dir': results_root_dir,
        'analysis_results_root_dir': _resolve_path(
            env_cfg.get('analysis_results_root_dir'),
            os.path.join(results_root_dir, 'analysis')
        ),
        'temp_results_root_dir': _resolve_path(
            env_cfg.get('temp_results_root_dir'),
            os.path.join(results_root_dir, 'temp')
        ),
    }


def create_config(config_file_env, config_file_exp):
    env_paths = load_env_paths(config_file_env)
    root_dir = env_paths['root_dir']
    image_root_dir = env_paths['image_root_dir']
    embeddings_root_dir = env_paths['embeddings_root_dir']
    results_root_dir = env_paths['results_root_dir']
    analysis_results_root_dir = env_paths['analysis_results_root_dir']
    temp_results_root_dir = env_paths['temp_results_root_dir']

    exp_code = os.path.basename(os.path.dirname(os.path.abspath(config_file_exp)))
    if exp_code == 'configs':
        exp_code = os.path.splitext(os.path.basename(config_file_exp))[0]

    # Keep experiment-specific artifacts separated under the configured results root.
    base_dir = os.path.join(results_root_dir, exp_code)
   
    with open(config_file_exp, 'r') as stream:
        config = yaml.safe_load(stream)
    
    cfg = EasyDict()
   
    # Copy
    for k, v in config.items():
        cfg[k] = v

    # Set paths for pretext task (These directories are needed in every stage)
    pretext_dir = os.path.join(base_dir, 'pretext')
    target_dir = os.path.join(embeddings_root_dir, exp_code)
    emb_dir = target_dir
    mkdir_if_missing(base_dir)
    mkdir_if_missing(pretext_dir)
    mkdir_if_missing(target_dir)

    # Store a copy of the training config with experiment artifacts.
    shutil.copyfile(config_file_exp, os.path.join(base_dir, 'simclr_acs.yml'))

    # Added one key to the config dict to have path for saving embeddings (and saving predictions as well, I guess!)
    cfg['exp_code'] = exp_code
    cfg['root_dir'] = root_dir
    cfg['base_dir'] = base_dir
    cfg['target_dir'] = target_dir
    cfg['emb_dir'] = emb_dir
    cfg['image_root_dir'] = image_root_dir
    cfg['embeddings_root_dir'] = embeddings_root_dir
    cfg['results_root_dir'] = results_root_dir
    cfg['analysis_results_root_dir'] = analysis_results_root_dir
    cfg['temp_results_root_dir'] = temp_results_root_dir
    cfg['pretext_dir'] = pretext_dir
    cfg['pretext_checkpoint'] = os.path.join(pretext_dir, 'checkpoint.pth.tar')
    cfg['pretext_model'] = os.path.join(pretext_dir, 'model.pth.tar')

    
    return cfg 
