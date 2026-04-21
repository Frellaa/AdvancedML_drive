import torch
import torch.nn as nn
from torch.optim import Adam
from sklearn.metrics import accuracy_score

def train_model(model, train_loader, val_loader, epochs=10, lr=1e-4, device='cpu'):
    # BCEWithLogitsLoss combines Sigmoid and Binary Cross Entropy
    criterion = nn.BCEWithLogitsLoss() 
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-5) # L2 regularization
    
    model.to(device)
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        # Validation Phase
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for X_val, y_val in val_loader:
                X_val, y_val = X_val.to(device), y_val.to(device)
                preds = model(X_val)
                loss = criterion(preds, y_val)
                val_loss += loss.item()
                
                # Convert logits to binary predictions (0 or 1)
                binary_preds = (torch.sigmoid(preds) > 0.5).float()
                all_preds.extend(binary_preds.cpu().numpy())
                all_targets.extend(y_val.cpu().numpy())
                
        val_acc = accuracy_score(all_targets, all_preds)
        
        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {train_loss/len(train_loader):.4f} | "
              f"Val Loss: {val_loss/len(val_loader):.4f} | "
              f"Val Accuracy: {val_acc:.4f}")