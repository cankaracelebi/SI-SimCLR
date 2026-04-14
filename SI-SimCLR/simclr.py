"""
Original version by Wouter Van Gansbeke, Simon Vandenhende
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
Modified by Elisa Messori
"""
import argparse
import os
import gc
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import tempfile                                     # These two lines are supposed to 'solve' the matplotlib warning
os.environ["MPLCONFIGDIR"] = tempfile.gettempdir()
import torch
from torchvision import transforms
import pickle
from utils.utils import experiments_summary
from utils.config import create_config
from utils.common_config import get_criterion, get_model, get_train_dataset,\
                                get_train_dataloader, get_train_transformations,\
                                get_optimizer,\
                                adjust_learning_rate, get_pairing_strategy,\
                                get_val_dataloader
from utils.train_utils import simclr_train
from utils.inference import get_latent_embeddings, PCA_latent_embeddings


# Parser
parser = argparse.ArgumentParser(description='SimCLR')
parser.add_argument('--config_env',
                    help='Config file for the environment')
parser.add_argument('--config_exp',
                    help='Config file for the experiment')
args = parser.parse_args()


def main():

    # Retrieve config file
    p = create_config(args.config_env, args.config_exp)
    print(p)
    
    print('')
    print(f'LAUNCHING EXPERIMENT {p["exp_code"]}')
    print('')
    exp_summary = experiments_summary(p)
    
    # Model
    print('Retrieve model')
    model = get_model(p)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device:\t{device}')
    print('Model is {}'.format(model.__class__.__name__))
    print('Model parameters: {:.2f}M'.format(sum(p.numel() for p in model.parameters()) / 1e6))
    print(model)
    model = model.to(device)

    print(f'Is CUDA available?\t{torch.cuda.is_available()}')
   
    # CUDNN
    print('Set CuDNN benchmark')
    torch.backends.cudnn.benchmark = (device.type == 'cuda')
    
    # Dataset
    print('Retrieve dataset')
    train_transforms = get_train_transformations(p)
    print('Train transforms:', train_transforms)
    #val_transforms = get_val_transformations(p)
    #print('Validation transforms:', val_transforms)
    train_dataset_pairing = get_pairing_strategy(p)
    print('THIS IS THE PAIRING STRATEGY')
    print(p['pairing_strategy'])
    train_dataset = get_train_dataset(p, train_transforms,
                                      to_augmented_dataset=train_dataset_pairing.get('augmentations', 0),
                                      to_adjacent_neighbor_dataset=train_dataset_pairing.get('adjacent_neighbors', 0))
    train_dataloader = get_train_dataloader(p, train_dataset)
    
    #val_dataset = get_val_dataset(p, val_transforms, to_cropped_neighbor_dataset=True)
    #val_dataloader = get_val_dataloader(p, val_dataset)
    print(f'Dataset contains {len(train_dataset)} train')
    print(f'In training there will be {train_dataset.repeated_pairs} repeated pairs')

    # Criterion
    print('Retrieve criterion')
    criterion = get_criterion(p)
    print('Criterion is {}'.format(criterion.__class__.__name__))
    criterion = criterion.to(device)

    # Optimizer and scheduler
    print('Retrieve optimizer')
    optimizer = get_optimizer(p, model)
    print(optimizer)
 
    # Checkpoint
    if os.path.exists(p['pretext_checkpoint']):
        print('Restart from checkpoint {}'.format(p['pretext_checkpoint']))
        checkpoint = torch.load(p['pretext_checkpoint'], map_location=device)
        optimizer.load_state_dict(checkpoint['optimizer'])
        model.load_state_dict(checkpoint['model'])
        model = model.to(device)
        start_epoch = checkpoint['epoch']

    else:
        print('No checkpoint file at {}'.format(p['pretext_checkpoint']))
        start_epoch = 0
        model = model.to(device)
    
    val = (p['train_db_name'] != p['val_db_name']) # val is True if it is present a validation set
    print(f'Presence of validation set: {val}')

    # Training
    print('Starting main loop')
    # If the train is re-starting from a checkpoint, we can just load the stored losses and update them
    if os.path.exists(os.path.join(p['base_dir'], 'losses_train.pickle')):
        epoch_losses_train = pickle.load(open(os.path.join(p['base_dir'], 'losses_train.pickle'), 'rb'))
    else:
        epoch_losses_train = []

    #if val:
    #    if os.path.exists(os.path.join(p['base_dir'], 'losses_val.pickle')):
    #        epoch_losses_val = pickle.load(open(os.path.join(p['base_dir'], 'losses_val.pickle'), 'rb'))
    #    else:
    #        epoch_losses_val = []

    for epoch in range(start_epoch, p['epochs']):
        print('Epoch %d/%d' %(epoch, p['epochs']))
        print('-'*15)

        # Adjust lr
        lr = adjust_learning_rate(p, optimizer, epoch)
        print('Adjusted learning rate to {:.5f}'.format(lr))
        
        # Train
        print('Train ...')
        #train_loss, train_dist, train_dist_cos = simclr_train(train_dataloader, model, criterion, optimizer, epoch, p['batch_size'], p['epochs'], p['base_dir'])
        
        train_loss = simclr_train(train_dataloader, model, criterion, optimizer, epoch, p['batch_size'], p['epochs'], p['base_dir'])
        epoch_losses_train.append(train_loss)
        #epoch_dist_train.append(train_dist)
        #epoch_dist_train_cos.append(train_dist_cos)
        with open(os.path.join(p['base_dir'], 'losses_train.pickle'), 'wb') as handle:
            pickle.dump(epoch_losses_train, handle, protocol=pickle.HIGHEST_PROTOCOL)

        #if val:
        #    #val_loss, val_dist, val_dist_cos = simclr_validation(val_dataloader, model, criterion, optimizer, epoch)
        #    val_loss = simclr_validation(val_dataloader, model, criterion, optimizer, epoch)
        #    epoch_losses_val.append(val_loss)
        #    with open(os.path.join(p['base_dir'], 'losses_val.pickle'), 'wb') as handle:
        #        pickle.dump(epoch_losses_val, handle, protocol=pickle.HIGHEST_PROTOCOL)
        # Checkpoint
        print('Checkpoint ...')
        torch.save({'optimizer': optimizer.state_dict(), 'model': model.state_dict(), 
                    'epoch': epoch + 1}, p['pretext_checkpoint'])

    # Save final model
    torch.save(model.state_dict(), p['pretext_model'])

    dataloader = get_val_dataloader(p, train_dataset)
    if not os.path.exists(os.path.join(p['target_dir'], 'latent_embeddings_train.pickle')):
        get_latent_embeddings(model, dataloader, p['target_dir'], p['batch_size']) #embeddings are computed on the training set by default
        print('Latent embeddings saved')

    #if val:
    #    print('Computing validation set embeddings')
    #    if not os.path.exists(os.path.join(p['target_dir'], 'latent_embeddings_val.pickle')):
    #        get_latent_embeddings(model, val_dataloader, p['target_dir'], p['batch_size'], 'val')
    #    if not os.path.exists(os.path.join(p['target_dir'], 'projected_embeddings_val.pickle')):    
    #        get_projected_embeddings(model, val_dataloader, p['target_dir'], p['batch_size'], 'val')

    exp_summary.loc[exp_summary['exp codes'] == p['exp_code'], 'trained'] = 1 # Update dataframe to say you finished training for the current experiment
    exp_summary.to_csv(os.path.join(p['results_root_dir'], 'exp_summary.csv'), index=False)
    #print('Computations completed!\nFor further analysis, please run results_analysis script (results.job)')

    TARGET_DIR = p['target_dir']
    EXP = p['exp_code']
    exp_summary = experiments_summary(p)

    if not os.path.exists(os.path.join(TARGET_DIR, 'PCA_embeddings_train.pickle')):
        PCA_latent_embeddings(TARGET_DIR, val)

    if not os.path.exists(os.path.join(TARGET_DIR, 'plots')):
        os.makedirs(os.path.join(TARGET_DIR, 'plots'))

if __name__ == '__main__':
    main()