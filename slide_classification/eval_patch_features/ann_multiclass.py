import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Optional, Tuple, List


class ANNMultiClassifier:
    def __init__(
        self,
        input_dim: int = 512,
        hidden_dim1: int = 512,
        hidden_dim2: int = 128,
        num_classes: int = 3,
        max_iter: int = 100,
        verbose: bool = True
    ):
        self.max_iter = max_iter
        self.verbose = verbose
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_dim = input_dim
        self.hidden_dim1 = hidden_dim1
        self.hidden_dim2 = hidden_dim2
        self.num_classes = num_classes

        # Model definition (NO softmax here)
        self.model = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim1),
            nn.ReLU(),
            nn.BatchNorm1d(self.hidden_dim1),
            nn.Dropout(0.3),
            nn.Linear(self.hidden_dim1, self.hidden_dim2),
            nn.ReLU(),
            nn.BatchNorm1d(self.hidden_dim2),
            nn.Dropout(0.3),
            nn.Linear(self.hidden_dim2, self.num_classes)
        ).to(self.device)

        self.loss_func = nn.CrossEntropyLoss()

    def compute_loss(self, preds: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return self.loss_func(preds, labels)

    def predict_proba(self, feats: torch.Tensor) -> torch.Tensor:
        feats = feats.to(self.device)
        self.model.eval()
        with torch.no_grad():
            logits = self.model(feats)
            return torch.softmax(logits, dim=1)  # Apply softmax here for probabilities

    def fit(
        self,
        train_feats: torch.Tensor,
        train_labels: torch.Tensor,
        val_feats: Optional[torch.Tensor] = None,
        val_labels: Optional[torch.Tensor] = None,
        combine_trainval: bool = False
    ) -> Tuple[List[float], List[float]]:
        train_feats, train_labels = train_feats.to(self.device), train_labels.to(self.device)

        if val_feats is not None:
            val_feats, val_labels = val_feats.to(self.device), val_labels.to(self.device)

        if combine_trainval and val_feats is not None:
            train_feats = torch.cat([train_feats, val_feats], dim=0)
            train_labels = torch.cat([train_labels, val_labels], dim=0)
            val_feats, val_labels = None, None

        optimizer = optim.Adam(self.model.parameters(), lr=1e-4, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.3, patience=5, verbose=self.verbose
        )

        train_loss_history, val_loss_history = [], []
        best_val_loss = float("inf")
        patience, epochs_no_improve = 20, 0

        for epoch in range(self.max_iter):
            # Training
            self.model.train()
            preds = self.model(train_feats)
            loss = self.compute_loss(preds, train_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss_history.append(loss.item())

            # Validation
            val_loss = None
            if val_feats is not None and not combine_trainval:
                self.model.eval()
                with torch.no_grad():
                    val_preds = self.model(val_feats)
                    val_loss = self.compute_loss(val_preds, val_labels)
                val_loss_history.append(val_loss.item())

                # Early stopping
                if val_loss.item() < best_val_loss:
                    best_val_loss = val_loss.item()
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= patience:
                        if self.verbose:
                            print(f"Early stopping at epoch {epoch}")
                        break

            # Scheduler step
            scheduler.step(val_loss.item() if val_loss is not None else loss.item())

            if self.verbose and epoch % 10 == 0:
                if val_loss is not None:
                    print(f"Epoch {epoch}: Train Loss = {loss:.4f}, Val Loss = {val_loss:.4f}")
                else:
                    print(f"Epoch {epoch}: Train Loss = {loss:.4f}")

        return train_loss_history, val_loss_history