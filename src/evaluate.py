import torch
from torch.utils.data import DataLoader
from torch import nn
from tqdm import tqdm
from typing import Dict, List, Any, Optional



def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    threshold: float = 0.5
    ) -> Dict[str, float]:
    model.eval()
    running_loss: float = 0.0
    correct_predictions: int = 0
    total_samples: int = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(data_loader):
            labels: torch.Tensor = batch.pop("label").to(device)
            inputs = {
                'seq': batch['seq'].to(device),
                'kinetics': batch['kinetics'].to(device)
            }

            logits = model(inputs)
            loss = criterion(logits, labels.float())
            running_loss += loss.item() * labels.size(0)

            probs = torch.sigmoid(logits)
            preds = probs>threshold
            
            total_samples += labels.size(0)

            correct_predictions += (preds == labels).sum().item()
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())


    epoch_loss: float = running_loss / total_samples
    epoch_acc: float = correct_predictions / total_samples
    return {"loss": epoch_loss, "accuracy": epoch_acc, 'preds': torch.cat(all_preds), 'labels': torch.cat(all_labels)}