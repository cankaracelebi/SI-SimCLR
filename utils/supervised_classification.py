import random
import yaml
import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
#from sklearn.linear_model import LogisticRegression

# Optional PyTorch implementation
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False



class PyTorchLogisticRegression(nn.Module):
    def __init__(self, input_dim, lr=1):
        super(PyTorchLogisticRegression, self).__init__()
        self.linear = nn.Linear(input_dim, 1)
        self.sigmoid = nn.Sigmoid()
        self.lr = lr
        #self.weight_decay = weight_decay
        self.classes_ = None  # Will be set in fit()
        self.encode_mode = "none"

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
        
        # Initialize weights equivalent to sklearn's defaults
        nn.init.normal_(self.linear.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.linear.bias, 0.0)

    def forward(self, x):
        return self.linear(x)

    def fit(self, X, y, max_iter=100, seed=42):
        torch.manual_seed(seed)
        y = np.array(y) # Ensure numpy
        
        # --- CUSTOM LABEL HANDLING ---
        # Check if labels are strings
        if y.dtype.kind in {'U', 'S', 'O'}: 
            unique_labels = np.unique(y)
            
            # 1. Search for 'control' or 'untreated' (case-insensitive)
            neg_label = None
            for label in unique_labels:
                if str(label).lower() in ['control', 'untreated']:
                    neg_label = label
                    break
            
            if neg_label is not None:
                # --- MANUAL MAPPING ---
                self.encode_mode = "manual"
                pos_labels = [l for l in unique_labels if l != neg_label]
                pos_label = pos_labels[0] if len(pos_labels) > 0 else "treated"
                
                self.classes_ = np.array([neg_label, pos_label])
                y_numeric = np.where(y == neg_label, 0.0, 1.0)
                
            else:
                # --- STANDARD SKLEARN BEHAVIOR ---
                self.encode_mode = "generic"
                le = LabelEncoder()
                y_numeric = le.fit_transform(y).astype(float)
                self.classes_ = le.classes_
        else:
            # Already numeric
            self.encode_mode = "none"
            y_numeric = y.astype(np.float32)
            self.classes_ = np.unique(y)

        # -----------------------------
        
        # FIX 1: Use y_numeric, not y
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y_numeric, dtype=torch.float32).view(-1, 1).to(self.device)
        
        criterion = nn.BCEWithLogitsLoss()

        # FIX 2: Pass weight_decay and use Strong Wolfe line search for stability
        optimizer = optim.LBFGS(self.parameters(), 
                                lr=self.lr, 
                                max_iter=20, 
                                history_size=10, 
                                line_search_fn='strong_wolfe',)

        def closure():
            optimizer.zero_grad()
            outputs = self.forward(X_tensor)
            loss = criterion(outputs, y_tensor)
            loss.backward()
            return loss
        
        self.train()
        for epoch in range(max_iter):
            # L-BFGS step requires the closure
            optimizer.step(closure)
            
            # Optional: Print loss to check convergence
            # FIX 3: .item() needs parentheses
            # with torch.no_grad():
            #     loss_val = closure().item()
            #     if (epoch + 1) % 10 == 0:
            #         print(f"Epoch {epoch+1}, Loss: {loss_val:.6f}")

    def predict_proba(self, X):
        self.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
            logits = self.forward(X_tensor)
            pos_probs = self.sigmoid(logits).cpu().numpy()
            neg_probs = 1 - pos_probs
            return np.hstack((neg_probs, pos_probs))

    def predict(self, X, threshold=0.5):
        probs = self.predict_proba(X)[:, 1]
        preds_numeric = (probs >= threshold).astype(int)
        
        # Restore original labels if we encoded them
        if self.encode_mode == "manual" or self.encode_mode == "generic":
            return self.classes_[preds_numeric]
            
        return preds_numeric



def build_torch_mlp(input_dim, hidden_sizes=(100,), dropout=0.0):
    layers = []
    in_dim = input_dim
    for h in hidden_sizes:
        layers.append(nn.Linear(in_dim, h))
        layers.append(nn.ReLU())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        in_dim = h
    layers.append(nn.Linear(in_dim, 2))
    return nn.Sequential(*layers)



def train_torch_model(X_train, y_train, X_val, device, hidden_sizes=(100,), lr=1e-3, max_epochs=100, seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    X_tr = torch.from_numpy(X_train.astype(np.float32)).to(device)
    y_tr = torch.from_numpy(y_train.astype(np.int64)).to(device)
    X_te = torch.from_numpy(X_val.astype(np.float32)).to(device)

    model = build_torch_mlp(X_train.shape[1], hidden_sizes=hidden_sizes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.train()
    for _ in range(max_epochs):
        optimizer.zero_grad()
        logits = model(X_tr)
        loss = criterion(logits, y_tr)
        loss.backward()
        
        # ADDED SAFETY: Clip gradients to prevent exploding gradients (NaNs)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
    
    model.eval()
    with torch.no_grad():
        logits_te = model(X_te)
        probs = torch.softmax(logits_te, dim=1).cpu().numpy()
    return model, probs



def balance_classes(ann_df, embeddings, label_col, random_state=0):
    rng = np.random.RandomState(random_state)
    labels = ann_df[label_col].unique()
    min_count = ann_df[label_col].value_counts().min()

    balanced_indices = []
    for lab in labels:
        lab_indices = ann_df[ann_df[label_col] == lab].index.values
        chosen = rng.choice(lab_indices, min_count, replace=False)
        balanced_indices.extend(chosen)

    balanced_indices = np.array(sorted(balanced_indices))
    return ann_df.loc[balanced_indices].reset_index(drop=True), embeddings[balanced_indices], min_count



def run_classifier(name, batch, classifier_name, ann_df, embeddings, labels, 
                   positive_class, cv_seeds, model_seeds, args, device):
    """
    Core execution function. 
    Uses args.model_type to branch between MLP (sklearn/torch) and LogReg.
    """
    results = []
    model_type = args.model_type # Taken from the args namespace

    for cv_seed in cv_seeds:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=cv_seed)
        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(embeddings, labels)):
            X_train, X_test = embeddings[train_idx], embeddings[test_idx]
            y_train, y_test = labels.iloc[train_idx].values, labels.iloc[test_idx].values
            img_code_test = ann_df['img_code'].iloc[test_idx]

            for m_seed in model_seeds:
                if model_type == 'mlp':
                    if args.use_torch:
                        unique_labels = np.unique(y_train)
                        label_to_int = {str(l): i for i, l in enumerate(unique_labels)}
                        y_train_int = np.array([label_to_int[str(x)] for x in y_train])
                        _, probs = train_torch_model(
                            X_train, y_train_int, X_test, device,
                            hidden_sizes=tuple(args.hidden_layer_sizes),
                            lr=args.lr, max_epochs=args.max_epochs, seed=m_seed
                        )
                        # Ensure we grab the correct column for the positive class
                        pos_index = list(unique_labels).index(positive_class) if positive_class in unique_labels else 1
                        pos_proba = probs[:, pos_index]
                        pred_classes = [unique_labels[int(p.argmax())] for p in probs]
                    else:
                        clf = MLPClassifier(hidden_layer_sizes=tuple(args.hidden_layer_sizes),
                                            max_iter=args.max_iter, random_state=m_seed)
                        clf.fit(X_train, y_train)
                        y_proba = clf.predict_proba(X_test)
                        pos_index = list(clf.classes_).index(positive_class) if positive_class in clf.classes_ else 1
                        pos_proba = y_proba[:, pos_index]
                        pred_classes = clf.predict(X_test)
                
                else: # Logistic Regression
                    input_dim = X_train.shape[1]
                    clf = PyTorchLogisticRegression(input_dim=input_dim, lr=1) # optimized with lbfgs: lr=1 is okay
                    clf.fit(X_train, y_train, max_iter=args.max_iter, seed=m_seed)
                    y_proba = clf.predict_proba(X_test)
                    pos_index = list(clf.classes_).index(positive_class) if positive_class in clf.classes_ else 1
                    pos_proba = y_proba[:, pos_index]
                    pred_classes = clf.predict(X_test)

                res_temp = pd.DataFrame({
                    'exp_code': name,
                    'batch': batch,
                    'classifier': classifier_name,
                    'cv_seed': cv_seed,
                    'fold': fold_idx,
                    'model_seed': m_seed,
                    'model_type': model_type,
                    'true': list(y_test),
                    'pred_class': list(pred_classes),
                    'pred_prob': list(pos_proba),
                    'img_code': list(img_code_test)
                })
                results.append(res_temp)
    return results



def balance_all_tasks(global_annotations, features, cv_seeds, model_seeds, args, device):
    results = []
    

    for name, embeddings in tqdm(features.items()):
        subset_sizes = {"ALS": [], "TIC high": [], "TIC low": []}
        temp_storage = {}

        for batch in global_annotations['experiment'].unique():
            # Get integer indices to safely slice the NumPy array
            batch_mask = global_annotations['experiment'] == batch
            batch_indices = np.where(batch_mask)[0] 
            
            batch_ann = global_annotations.iloc[batch_indices].reset_index(drop=True)
            batch_embeddings = embeddings[batch_indices] # Safe slice

            # ALS Task
            als_mask = batch_ann['condition'] == 'untreated'
            als_indices = np.where(als_mask)[0]
            als_ann = batch_ann.iloc[als_indices].reset_index(drop=True)
            
            if als_ann['als_label'].nunique() == 2:
                _, _, size = balance_classes(als_ann, batch_embeddings[als_indices], 'als_label')
                subset_sizes["ALS"].append(size)
                temp_storage[(name, batch, "ALS")] = (als_ann, batch_embeddings[als_indices])

            # TIC Tasks (High/Low)
            for tic_key, cond_to_exclude in [("TIC high", "A1 low"), ("TIC low", "A1 high")]:
                tic_mask = (batch_ann['als_label'] == 'control') & (batch_ann['condition'] != cond_to_exclude)
                tic_indices = np.where(tic_mask)[0]
                tic_ann = batch_ann.iloc[tic_indices].reset_index(drop=True)
                
                if len(tic_ann['condition'].unique()) == 2:
                    _, _, size = balance_classes(tic_ann, batch_embeddings[tic_indices], 'condition')
                    subset_sizes[tic_key].append(size)
                    temp_storage[(name, batch, tic_key)] = (tic_ann, batch_embeddings[tic_indices])

        # Find global minimum for strict balancing safely
        valid_sizes = [min(sizes) for sizes in subset_sizes.values() if len(sizes) > 0]
        if not valid_sizes: 
            continue
        global_min = min(valid_sizes)

        for (name_, batch_, task), (ann, emb) in temp_storage.items():
            label_col = "als_label" if task == "ALS" else "condition"
            ann_bal, emb_bal, _ = balance_classes(ann, emb, label_col)
            
            if len(ann_bal) > 2 * global_min:
                # Sample the dataframe but KEEP the old index to slice embeddings
                ann_bal_sampled = ann_bal.groupby(label_col, group_keys=False).apply(
                    lambda x: x.sample(global_min, random_state=0)
                )
                # Slice embeddings using the preserved index values
                emb_bal = emb_bal[ann_bal_sampled.index.values]
                
                # NOW it is safe to reset the index
                ann_bal = ann_bal_sampled.reset_index(drop=True)

            # Call run_classifier with aligned data
            results += run_classifier(
                name=name_, 
                batch=batch_, 
                classifier_name=task, 
                ann_df=ann_bal,
                embeddings=emb_bal, 
                labels=ann_bal[label_col],
                positive_class='als' if task=="ALS" else ('A1 high' if task=="TIC high" else 'A1 low'),
                cv_seeds=cv_seeds, 
                model_seeds=model_seeds, 
                args=args, 
                device=device
            )
    return results