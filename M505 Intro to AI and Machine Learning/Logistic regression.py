import os
import pandas as pd, numpy as np, torch, torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns

# load data
X = pd.read_csv("X_tcga.csv", index_col=0)
meta = pd.read_csv("meta_tcga.csv")

common_ids = X.index.intersection(meta["sample"])
X = X.loc[common_ids]
meta = meta.set_index("sample").loc[common_ids]

label_map = {"Normal Tissue":0, "Solid Tissue Normal":0, "Primary Tumor":1, "Metastatic":1}
y = meta["_sample_type"].map(label_map)

print("X shape:", X.shape)
print("y distribution:\n", y.value_counts())

# model
class LogReg(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.fc = nn.Linear(in_dim, 2)
    def forward(self, x):
        return self.fc(x)

device = "cuda" if torch.cuda.is_available() else "cpu"

# cross validation
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold_accs, fold_aucs = [], []
best_fold, best_auc, best_model_state, best_preds, best_y_val = None, -1, None, None, None


os.makedirs("results", exist_ok=True)

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y), 1):
    print(f"\n--- Fold {fold} ---")
    
    # split
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    # tensors
    X_train = torch.tensor(X_train.values, dtype=torch.float32)
    y_train = torch.tensor(y_train.values, dtype=torch.long)
    X_val   = torch.tensor(X_val.values,   dtype=torch.float32)
    y_val   = torch.tensor(y_val.values,   dtype=torch.long)

    # init
    model = LogReg(X.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    best_val_loss, patience, best_model = 1e9, 0, None
    train_losses, val_losses, val_accs, val_aucs = [], [], [], []

    for epoch in range(50):
        model.train()
        optimizer.zero_grad()
        out = model(X_train.to(device))
        loss = criterion(out, y_train.to(device))
        loss.backward(); optimizer.step()

        model.eval()
        with torch.no_grad():
            out_val = model(X_val.to(device))
            val_loss = criterion(out_val, y_val.to(device)).item()
            preds = out_val.argmax(1).cpu().numpy()
            probs = torch.softmax(out_val, dim=1)[:,1].cpu().numpy()
            acc = accuracy_score(y_val, preds)
            auc = roc_auc_score(y_val, probs)

        train_losses.append(loss.item())
        val_losses.append(val_loss)
        val_accs.append(acc)
        val_aucs.append(auc)

        if best_val_loss - val_loss > 0.05:
            best_val_loss = val_loss
            best_model = model.state_dict().copy()
            patience = 0
        else:
            patience += 1
        if patience > 5:
            print("  early stopping")
            break

        if (epoch+1) % 5 == 0:
            print(f"epoch {epoch+1:02d} | train={loss.item():.3f} | "
                  f"val={val_loss:.3f} | acc={acc:.3f} | auc={auc:.3f}")

    model.load_state_dict(best_model)

    fold_accs.append(val_accs[-1])
    fold_aucs.append(val_aucs[-1])
    print(f"fold {fold} | acc={val_accs[-1]:.3f} | auc={val_aucs[-1]:.3f}")

    if val_aucs[-1] > best_auc:
        best_auc = val_aucs[-1]
        best_fold = fold
        best_model_state = model.state_dict().copy()
        best_preds = preds
        best_y_val = y_val.cpu().numpy()

# plots
    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    plt.plot(train_losses, label="train loss")
    plt.plot(val_losses, label="val loss")
    plt.title(f"loss (fold {fold})"); plt.legend()

    plt.subplot(1,2,2)
    plt.plot(val_accs, label="val acc")
    plt.plot(val_aucs, label="val auc")
    plt.title(f"val acc/auc (fold {fold})"); plt.legend()
    plt.tight_layout()
    plt.savefig(f"results/training_fold{fold}.png", dpi=150)
    plt.close()

    fpr, tpr, _ = roc_curve(y_val, probs)
    plt.figure(figsize=(5,5))
    plt.plot(fpr, tpr, label=f"auc={auc:.3f}")
    plt.plot([0,1],[0,1],"k--")
    plt.title(f"roc (fold {fold})"); plt.legend()
    plt.savefig(f"results/roc_fold{fold}.png", dpi=150)
    plt.close()

# summary
print("\n===== CV Results =====")
print(f"mean acc: {np.mean(fold_accs):.3f} ± {np.std(fold_accs):.3f}")
print(f"mean auc: {np.mean(fold_aucs):.3f} ± {np.std(fold_aucs):.3f}")
print(f"\n>>> Best fold was Fold {best_fold} with AUC={best_auc:.3f}")

best_model = LogReg(X.shape[1]).to(device)
best_model.load_state_dict(best_model_state)

cm = confusion_matrix(best_y_val, best_preds)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Normal","Tumor"],
            yticklabels=["Normal","Tumor"])
plt.xlabel("Predicted"); plt.ylabel("True")
plt.title(f"Confusion Matrix (Best Fold {best_fold})")
plt.tight_layout()
plt.savefig(f"results/confusion_matrix_best_fold.png", dpi=150)
plt.close()

weights = best_model.fc.weight.detach().cpu().numpy()[1]
top_idx = np.argsort(np.abs(weights))[::-1][:10]
print("\nTop 10 influential genes/features (Best Fold):")
for i in top_idx:
    print(f"  {X.columns[i]} | weight={weights[i]:.4f}")

plt.figure(figsize=(6,4))
sns.barplot(x=np.abs(weights[top_idx]), y=X.columns[top_idx])
plt.title("Top 10 Genes (Best Fold)")
plt.xlabel("Weight magnitude"); plt.ylabel("Gene")
plt.tight_layout()
plt.savefig("results/top_genes_best_fold.png", dpi=150)
plt.close()

