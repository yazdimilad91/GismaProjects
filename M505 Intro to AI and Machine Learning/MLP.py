import os
import numpy as np
import pandas as pd
import torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns


# 1) Load & prep data

X = pd.read_csv("X_tcga.csv", index_col=0)
meta = pd.read_csv("meta_tcga.csv")

common_ids = X.index.intersection(meta["sample"])
X = X.loc[common_ids]
meta = meta.set_index("sample").loc[common_ids]

label_map = {"Normal Tissue":0, "Solid Tissue Normal":0, "Primary Tumor":1, "Metastatic":1}
y = meta["_sample_type"].map(label_map).astype(int)

print("X shape:", X.shape)
print("y distribution:\n", y.value_counts())

# Stratified train/val split (no CV)
X_train, X_val, y_train, y_val = train_test_split(
    X.values, y.values, test_size=0.2, random_state=42, stratify=y.values
)

# Standardize features (fit on train, transform val)
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_val   = scaler.transform(X_val)


# 2) Torch tensors/Loaders

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(42); np.random.seed(42)

X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.long)
X_val_t   = torch.tensor(X_val,   dtype=torch.float32)
y_val_t   = torch.tensor(y_val,   dtype=torch.long)

train_ds = TensorDataset(X_train_t, y_train_t)
val_ds   = TensorDataset(X_val_t, y_val_t)

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=256, shuffle=False)


# 3) Simple MLP model

class MLP(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2)  # logits for 2 classes
        )
    def forward(self, x):
        return self.net(x)

model = MLP(in_dim=X.shape[1]).to(device)

# Class weights to handle imbalance 
n0 = (y_train == 0).sum()
n1 = (y_train == 1).sum()
N  = n0 + n1
w0 = N / (2.0 * n0)
w1 = N / (2.0 * n1)
class_weights = torch.tensor([w0, w1], dtype=torch.float32, device=device)

criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)


# 4) Train loop 

epochs = 40
train_losses, val_losses, val_accs, val_aucs = [], [], [], []

os.makedirs("results", exist_ok=True)

for epoch in range(1, epochs+1):
    # ---- train ----
    model.train()
    epoch_loss = 0.0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * xb.size(0)
    epoch_loss /= len(train_loader.dataset)
    train_losses.append(epoch_loss)

    # ---- validate ----
    model.eval()
    with torch.no_grad():
        val_loss = 0.0
        all_probs, all_preds, all_true = [], [], []
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            val_loss += loss.item() * xb.size(0)

            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            preds = logits.argmax(dim=1).cpu().numpy()
            all_probs.append(probs); all_preds.append(preds); all_true.append(yb.cpu().numpy())

        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)

        all_probs = np.concatenate(all_probs)
        all_preds = np.concatenate(all_preds)
        all_true  = np.concatenate(all_true)

        acc = accuracy_score(all_true, all_preds)
        try:
            auc = roc_auc_score(all_true, all_probs)
        except ValueError:
            auc = np.nan

        val_accs.append(acc)
        val_aucs.append(auc)

    if epoch % 5 == 0 or epoch == 1 or epoch == epochs:
        print(f"Epoch {epoch:02d} | train_loss={epoch_loss:.4f} | val_loss={val_loss:.4f} | val_acc={acc:.3f} | val_auc={auc:.3f}")


# 5) Plots & metrics
# Training curves
plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.plot(train_losses, label="train loss")
plt.plot(val_losses, label="val loss")
plt.title("Loss"); plt.legend()

plt.subplot(1,2,2)
plt.plot(val_accs, label="val acc")
plt.plot(val_aucs, label="val auc")
plt.title("Validation Acc/AUC"); plt.legend()
plt.tight_layout()
plt.savefig("results/training_curves.png", dpi=150)
plt.show(); plt.close()

# ROC curve
fpr, tpr, _ = roc_curve(all_true, all_probs)
plt.figure(figsize=(5,5))
plt.plot(fpr, tpr, label=f"AUC={val_aucs[-1]:.3f}")
plt.plot([0,1],[0,1],"k--")
plt.xlabel("FPR"); plt.ylabel("TPR")
plt.title("ROC (Validation)")
plt.legend()
plt.tight_layout()
plt.savefig("results/roc_val.png", dpi=150)
plt.show(); plt.close()

# Confusion Matrix
cm = confusion_matrix(all_true, all_preds)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Normal","Tumor"],
            yticklabels=["Normal","Tumor"])
plt.xlabel("Predicted"); plt.ylabel("True")
plt.title("Confusion Matrix (Validation)")
plt.tight_layout()
plt.savefig("results/confusion_matrix_val.png", dpi=150)
plt.show(); plt.close()

print("\nFinal validation metrics:")
print(f"Accuracy: {val_accs[-1]:.3f}")
print(f"AUC:      {val_aucs[-1]:.3f}")