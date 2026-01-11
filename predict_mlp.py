import random
import numpy as np
import os
import argparse
from tqdm import trange, tqdm
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import matplotlib.pyplot as plt
import copy
import gc

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

# --- Improved Model Architecture: SE-ResNet for Tabular Data ---
# Incorporating Squeeze-and-Excitation blocks and GELU activation
class SEBlock(nn.Module):
    def __init__(self, dim, reduction=8):
        super().__init__()
        self.se = nn.Sequential(
            nn.Linear(dim, dim // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(dim // reduction, dim, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.se(x)

class ResNetBlock(nn.Module):
    def __init__(self, dim, dropout=0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm1d(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.BatchNorm1d(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.se = SEBlock(dim)

    def forward(self, x):
        return x + self.se(self.block(x))

class ResNetModel(nn.Module):
    def __init__(self, input_dims, hidden_dim=256, num_blocks=4, dropout=0.2):
        super().__init__()
        
        self.input_proj = nn.Sequential(
            nn.BatchNorm1d(input_dims),
            nn.Linear(input_dims, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.0)
        )
        
        self.blocks = nn.Sequential(
            *[ResNetBlock(hidden_dim, dropout) for _ in range(num_blocks)]
        )
        
        self.head = nn.Sequential(
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, 1)
        )
        
        self.sigmoid = nn.Sigmoid()
        self.BCEloss = nn.BCEWithLogitsLoss()

    def forward(self, inputs, labels=None):
        x = self.input_proj(inputs)
        x = self.blocks(x)
        prediction_scores = self.head(x)
        logits = self.sigmoid(prediction_scores)

        if labels is None:
            return logits
        else:
            loss = self.BCEloss(prediction_scores, labels.unsqueeze(1).float())
            return logits, loss

class MyDataset(Dataset):
    def __init__(self, dataframe):
        self.features = dataframe.drop(['win'], axis=1).values.astype(np.float32)
        self.labels = dataframe['win'].values.astype(np.float32)
        
    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        return {"inputs": self.features[index], "label": self.labels[index]}

def create_data_loader(dataset, batch_size, shuffle=True):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=2, pin_memory=True)

def evaluate(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for data in dataloader:
            inputs = data['inputs'].to(device)
            labels = data['label'].to(device)
            logits, _ = model(inputs, labels)
            preds = (logits > 0.5).int().cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            
    return accuracy_score(all_labels, all_preds)

def train_fold(fold_idx, train_df, val_df, input_dims, args, device):
    print(f"\n=== Training Fold {fold_idx+1} ===")
    
    train_dataset = MyDataset(train_df)
    val_dataset = MyDataset(val_df)
    
    train_loader = create_data_loader(train_dataset, args.bsz, shuffle=True)
    val_loader = create_data_loader(val_dataset, args.bsz * 2, shuffle=False)
    
    # Initialize Model for this fold
    model = ResNetModel(input_dims=input_dims, hidden_dim=256, num_blocks=3, dropout=0.2).to(device)
    
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    
    # Cosine Annealing with Warm Restarts
    # T_0=10 means restart every 10 epochs initially
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)
    
    best_val_acc = 0.0
    best_state = None
    patience_cnt = 0
    
    for epoch in range(args.epochs):
        model.train()
        train_loss_sum = 0
        for data in train_loader:
            inputs = data['inputs'].to(device)
            labels = data['label'].to(device)
            
            optimizer.zero_grad()
            _, loss = model(inputs, labels)
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item()
            
            # Step scheduler per batch (for WarmRestarts)
            scheduler.step(epoch + len(train_loader) / len(train_loader)) 

        # Validation
        val_acc = evaluate(model, val_loader, device)
        
        # Simple logging
        if (epoch + 1) % 5 == 0 or epoch == 0:
            avg_loss = train_loss_sum / len(train_loader)
            print(f"Epoch {epoch+1}/{args.epochs} | Loss: {avg_loss:.4f} | Val Acc: {val_acc:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())
            patience_cnt = 0
        else:
            patience_cnt += 1
            
        if patience_cnt >= args.patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
            
    print(f"Fold {fold_idx+1} Best Val Acc: {best_val_acc:.4f}")
    
    # Save best model for this fold
    save_path = f'model_fold_{fold_idx}.pth'
    torch.save(best_state, save_path)
    return best_val_acc

def main(args):
    set_seed(2025)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load All Data
    full_df = pd.read_csv('./data/train_processed.csv')
    test_df = pd.read_csv('./data/test_processed.csv')
    
    input_dims = full_df.shape[1] - 1
    print(f"Input features: {input_dims}")

    # K-Fold Cross Validation
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=2025)
    
    fold_scores = []
    
    # Training Loop
    for fold, (train_idx, val_idx) in enumerate(kfold.split(full_df, full_df['win'])):
        train_data = full_df.iloc[train_idx].reset_index(drop=True)
        val_data = full_df.iloc[val_idx].reset_index(drop=True)
        
        score = train_fold(fold, train_data, val_data, input_dims, args, device)
        fold_scores.append(score)
        
        # Clean up to save memory
        gc.collect()
        torch.cuda.empty_cache()
        
    print(f"\nAverage CV Score: {np.mean(fold_scores):.4f}")

    # --- Inference Ensemble ---
    print("\nStarting Ensemble Inference...")
    test_dataset = MyDataset(test_df) # test_processed has a dummy 'win' column usually
    test_loader = create_data_loader(test_dataset, batch_size=4096, shuffle=False)
    
    # Array to store sum of predictions
    ensemble_preds = np.zeros(len(test_df))
    
    for fold in range(5):
        print(f"Loading Fold {fold+1} model...")
        model = ResNetModel(input_dims=input_dims, hidden_dim=256, num_blocks=3, dropout=0.2).to(device)
        model.load_state_dict(torch.load(f'model_fold_{fold}.pth'))
        model.eval()
        
        fold_preds = []
        with torch.no_grad():
            for data in test_loader:
                inputs = data['inputs'].to(device)
                logits = model(inputs) # logits are actually probabilities because of sigmoid in forward
                fold_preds.extend(logits.cpu().numpy().flatten())
        
        ensemble_preds += np.array(fold_preds)
    
    # Average predictions
    avg_preds = ensemble_preds / 5.0
    final_predictions = (avg_preds > 0.5).astype(int)
    
    submission = pd.DataFrame({'win': final_predictions})
    submission.to_csv('submission_ensemble.csv', index=False)
    print("Ensemble submission saved to 'submission_ensemble.csv'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bsz", type=int, default=2048, help="batch size")
    parser.add_argument("--lr", type=float, default=1.5e-3, help="learning rate")
    parser.add_argument("--epochs", type=int, default=100, help="max epochs")
    parser.add_argument("--patience", type=int, default=15, help="early stopping")
    args = parser.parse_args()

    main(args)
