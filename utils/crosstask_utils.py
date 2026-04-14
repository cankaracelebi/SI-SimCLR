import numpy as np
import pandas as pd
import os
import torch
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torchvision import transforms
import random
import torchvision.transforms.functional as TF
from typing import Sequence


def get_transforms():
    """Standard ImageNet normalization and resizing"""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def subsample_data(annotations):

    batches = annotations['experiment'].unique()
    counts = annotations.groupby(['condition', 'als_label', 'experiment']).size()
    
    #print(counts)
    counts_values = counts.values
    min_samples = np.min(counts_values[counts_values > 0])

    if counts[('untreated', 'control')].min() < 2 * min_samples:
        min_samples = counts[('untreated', 'control')].min() // 2

    #print(f'Each category had {min_samples} samples coming from each experiment')

    # Create two disjoint sets for control untreated samples
    ctrl1_indices = []
    ctrl2_indices = []

    # Create a dictionary to store indices for each category and experiment
    selected_indices = {('untreated', 'control'): [], ('untreated', 'als'): [], ('A1 high', 'control'): [], ('A1 low', 'control'): []}

    for condition, als_label in selected_indices.keys():
        n_samples = min_samples
        if condition == 'untreated' and als_label == 'control':
            n_samples = min_samples * 2
        for experiment in batches:
            indices = annotations[(annotations['condition'] == condition) & (annotations['als_label'] == als_label) & (annotations['experiment'] == experiment)].index
            if indices.size == 0:
                continue
            selected = np.random.choice(indices, n_samples, replace=False)
            selected_indices[(condition, als_label)].extend(selected)

    np.random.shuffle(selected_indices[('untreated', 'control')], )
    ctrl1_indices = selected_indices[('untreated', 'control')][:min_samples]
    ctrl2_indices = selected_indices[('untreated', 'control')][min_samples:]

    # Create dataframes for each category
    als = annotations.loc[selected_indices[('untreated', 'als')]]
    als['class'] = 'ALS'
    ctrl1 = annotations.loc[ctrl1_indices]
    ctrl2 = annotations.loc[ctrl2_indices]
    ctrl1['class'] = 'CTRL'
    ctrl2['class'] = 'CTRL'
    tichigh = annotations.loc[selected_indices[('A1 high', 'control')]]
    ticlow = annotations.loc[selected_indices[('A1 low', 'control')]]
    tichigh['class'] = 'TIChigh'
    ticlow['class'] = 'TIClow'

    als_dataset = pd.concat([als, ctrl2])
    tichigh_dataset = pd.concat([tichigh, ctrl1])
    ticlow_dataset = pd.concat([ticlow, ctrl1])

    # Verify that als_dataset does not have any 'img_code' value in common with tichigh_dataset or ticlow_dataset
    assert not als_dataset['img_code'].isin(tichigh_dataset['img_code']).any(), "ALS data and TIC high data have common images"
    assert not als_dataset['img_code'].isin(ticlow_dataset['img_code']).any(), "ALS dataset and TIC low dataset have common 'img_code' values"
    assert not tichigh_dataset['img_code'].isin(als_dataset['img_code']).any(), "TIC high dataset and ALS dataset have common 'img_code' values"
    assert not ticlow_dataset['img_code'].isin(als_dataset['img_code']).any(), "TIC low dataset and ALS dataset have common 'img_code' values"

    subsampled_data = {
        'ALS': als_dataset,
        'TIChigh': tichigh_dataset,
        'TIClow': ticlow_dataset
    }

    return subsampled_data


def train_MLP_classifier(classifier_name: str, data: pd.DataFrame, embeddings_col: str, saving_dir: str):

    os.makedirs(saving_dir, exist_ok=True)

    label = 'condition' if 'TIC' in classifier_name else 'als_label'
    if 'high' in classifier_name:
        pos_label = 'A1 high'
    elif 'low' in classifier_name:
        pos_label = 'A1 low'
    else:
        pos_label = 'ALS'
    # Extract labels and embeddings from the DataFrame
    labels = data[label].map(lambda x: 1 if x == pos_label else 0).values
    embeddings = np.stack(data[embeddings_col].values)
    ann = data.drop(columns=[embeddings_col])

    emb_name = {'pretrained_resnet_embedding': 'ResNet_pretr', 'simclr_embedding': 'SimCLR'}[embeddings_col]

    scores = []
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    best_auc = 0
    best_model = None
    best_fold = None

    # Dictionaries to store train and test indices and labels
    train_indices = {}
    test_indices = {}
    train_labels = {}
    test_labels = {}

    for fold, (train_index, test_index) in enumerate(skf.split(embeddings, labels)):
        X_train, X_test = embeddings[train_index], embeddings[test_index]
        y_train, y_test = labels[train_index], labels[test_index]

        # Store indices and labels
        train_indices[fold] = train_index
        test_indices[fold] = test_index
        train_labels[fold] = y_train
        test_labels[fold] = y_test

        clf = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, random_state=42,)
        clf.fit(X_train, y_train)
        # Validation
        y_pred = clf.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_pred)
        auc_score = auc(fpr, tpr)
        scores.append(auc_score)

        # Save the model if it has the best AUC so far
        if auc_score > best_auc:
            best_auc = auc_score
            best_model = clf
            best_fold = fold
            model_path = os.path.join(saving_dir, f'{classifier_name}_F{fold}.pt')
            # Delete existing models in the saving_dir
            for file in os.listdir(saving_dir):
                if file.startswith(f'{classifier_name}_F') and file.endswith('.pt'):
                    os.remove(os.path.join(saving_dir, file))
            torch.save(clf, model_path)

        # Save the test annotations for each fold
        test_annotations = ann.iloc[test_index]
        test_annotations[f'predicted_probability_{emb_name}'] = y_pred
        test_annotations['fold'] = fold
        #test_annotations.to_csv(os.path.join(saving_dir, f'{classifier_name}_fold_{fold}_test.csv'), index=False)
        if fold == 0:
            all_folds_annotations = test_annotations
        else:
            all_folds_annotations = pd.concat([all_folds_annotations, test_annotations], axis=0)

        print(f'Fold {fold} was trained for {clf.n_iter_} iterations')

    all_folds_annotations.to_csv(os.path.join(saving_dir, f'{classifier_name}_all_folds_test.csv'), index=False)

    return best_model, best_fold, best_auc, scores, all_folds_annotations


def get_resnet_model(device, partial_freeze=False):
    # 1. Load Pretrained
    try:
        weights = models.ResNet18_Weights.DEFAULT
        model = models.resnet18(weights=weights)
    except:
        model = models.resnet18(pretrained=True)

    if partial_freeze:
        for param in model.parameters():
            param.requires_grad = False

        # 3. Unfreeze the last block (layer4)
        # This allows the model to adapt high-level features to your specific cells
        for param in model.layer4.parameters():
            param.requires_grad = True

    # 4. Replace the Head
    # New layers are always trainable (requires_grad=True) by default
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    
    return model.to(device)


def train_image_classifier(classifier_name, data, saving_dir, batch_size=32, n_epochs=25, lr=10e-4):

    os.makedirs(saving_dir, exist_ok=True)

    if 'embedding' in data.columns:
        ann = data.drop(columns=['embedding'])
    else:
        ann = data.copy()

    label_col = 'condition' if 'TIC' in classifier_name else 'als_label'
    if 'high' in classifier_name:
        pos_label = 'A1 high'
    elif 'low' in classifier_name:
        pos_label = 'A1 low'
    else:
        pos_label = 'ALS'

    print(f'Classifier is {label_col} and positive label is {pos_label}')

    # Data transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Prepare data
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    binary_labels = data[label_col].map(lambda x: 1 if x == pos_label else 0).values
    
    best_auc = 0
    best_fold = None
    scores = []
    
    # Container for all predictions
    all_folds_annotations = pd.DataFrame()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    for fold, (train_idx, test_idx) in enumerate(skf.split(data, binary_labels)):
        train_data = data.iloc[train_idx]
        test_data = data.iloc[test_idx]

        train_dataset = ImageDataset(train_data, label_col, pos_label, transform=transform)
        test_dataset = ImageDataset(test_data, label_col, pos_label, transform=transform)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

        # Load pre-trained ResNet18
        model = get_resnet_model(device, partial_freeze=True)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

        # Training loop
        for epoch in range(n_epochs):
            model.train()
            for imgs, lbls in train_loader:
                imgs = imgs.to(device)
                lbls = lbls.to(device).long()
                
                optimizer.zero_grad()
                out = model(imgs)
                loss = criterion(out, lbls)
                loss.backward()
                optimizer.step()
                
        # Validation
        model.eval()
        y_true = []
        y_pred = []
        with torch.no_grad():
            for imgs, lbls in test_loader:
                imgs = imgs.to(device)
                out = model(imgs)
                probs = torch.softmax(out, dim=1)[:, 1].cpu().numpy()
                # Handle edge case if batch size is 1
                #if probs.ndim == 0: probs = [probs] 
                y_pred.extend(probs)
                y_true.extend(lbls.numpy())
                
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        auc_score = auc(fpr, tpr)
        scores.append(auc_score)

        # Save the model if it has the best AUC so far
        #print('Selecting best model...')
        if auc_score > best_auc:
            best_auc = auc_score
            best_fold = fold
            # Clean up old models for this classifier
            for f in os.listdir(saving_dir):
                if f.startswith(f'{classifier_name}_F') and f.endswith('.pt'):
                    os.remove(os.path.join(saving_dir, f))
            torch.save(model.state_dict(), os.path.join(saving_dir, f'{classifier_name}_F{fold}.pt'))
            
        # Store predictions
        fold_res = ann.iloc[test_idx].copy()
        fold_res['predicted_probability'] = y_pred
        fold_res['fold'] = fold
        all_folds_annotations = pd.concat([all_folds_annotations, fold_res])

    # Save CSV
    all_folds_annotations.to_csv(os.path.join(saving_dir, f'{classifier_name}_all_folds_test.csv'), index=False)
    
    return None, best_fold, best_auc, scores, all_folds_annotations


def load_resnet_model(model_path, device):
    """Loads a saved ResNet18 model state dict"""
    # Initialize architecture
    try:
        weights = models.ResNet18_Weights.DEFAULT
        model = models.resnet18(weights=weights)
    except:
        model = models.resnet18(pretrained=True)
    
    # Adjust head for binary classification
    model.fc = nn.Linear(model.fc.in_features, 2)
    
    # Load weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def inference_resnet(model, data, classifier_name_target, batch_size=32):
    """
    Runs inference using a loaded model (2-class output).
    """
    device = next(model.parameters()).device
    
    # Determine labels
    label_col = 'condition' if 'TIC' in classifier_name_target else 'als_label'
    if 'high' in classifier_name_target:
        pos_label = 'A1 high'
    elif 'low' in classifier_name_target:
        pos_label = 'A1 low'
    else:
        pos_label = 'ALS'
        
    ds = ImageDataset(data, label_col, pos_label, transform=get_transforms())
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4)
    
    y_true = []
    y_pred = []
    
    model.eval()
    with torch.no_grad():
        for imgs, lbls in loader:
            imgs = imgs.to(device)
            out = model(imgs) # Output shape is (Batch, 2)
            
            # --- FIX: Use Softmax and select index 1 (Positive Class) ---
            probs = torch.softmax(out, dim=1)[:, 1].cpu().numpy()
            
            y_pred.extend(probs)
            y_true.extend(lbls.numpy())
            
    return np.array(y_true), np.array(y_pred)


# 1. Dataset class to load the images and their labels
class ImageDataset(Dataset):
    """
    Dataset class that loads images from the 'img_code' column in the dataframe.
    """
    def __init__(self, df, label_col, pos_label, transform=None):
        self.df = df
        self.label_col = label_col
        self.pos_label = pos_label
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row['img_code']
        # Binary label: 1 if matches positive label, else 0
        label = 1 if row[self.label_col] == self.pos_label else 0
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # Return black image on failure to prevent crash
            image = Image.new('RGB', (224, 224))
            
        if self.transform:
            image = self.transform(image)
        return image, label


# 2. Class for extracting embeddings from a pre-trained model (e.g., ResNet)
class FeatureExtractor(nn.Module):
    def __init__(self, pretrained_model_name='resnet18'):
        super().__init__()
        # Load the pre-trained model
        model = models.__dict__[pretrained_model_name](pretrained=True)
        # Remove the last classification layer (we want the feature extractor)
        self.backbone = nn.Sequential(*list(model.children())[:-1])  # Remove the last FC layer
        self.backbone_out_dim = model.fc.in_features  # Get the input dimension of the last layer

    def forward(self, x):
        x = self.backbone(x)  # Extract features from the backbone
        x = torch.flatten(x, 1)  # Flatten the output to 1D vector
        return x


# 3. Function to obtain embeddings for each image in the dataset
def obtain_embeddings(data, labels_mutation, labels_condition, transform=None, batch_size=64, pretrained_model_name='resnet18'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create dataset and dataloader for batching
    dataset = ImageDataset(data, labels_mutation, labels_condition, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Initialize the model and move it to the device (GPU/CPU)
    feature_extractor = FeatureExtractor(pretrained_model_name=pretrained_model_name)
    feature_extractor = feature_extractor.to(device)
    feature_extractor.eval()  # Set the model to evaluation mode

    embeddings = {}

    with torch.no_grad():
        for images, _, _, indices in dataloader:
            images = images.to(device)  # Move the images to the device

            # Extract embeddings from the feature extractor
            embeddings_batch = feature_extractor(images)
            embeddings_batch = embeddings_batch.cpu().numpy()  # Convert to NumPy array for storage

            # Store embeddings with corresponding indices
            for idx, emb in zip(indices, embeddings_batch):
                embeddings[idx] = emb

    return embeddings

