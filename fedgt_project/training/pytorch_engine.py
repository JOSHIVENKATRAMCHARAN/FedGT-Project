from collections import defaultdict

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class SimpleClientNet(nn.Module):
    """Small MNIST-style classifier adapted to the 10-feature project dataset."""

    def __init__(self, input_dim=10, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def run_local_training(round_number=None, epochs=2, batch_size=32, lr=0.01):
    """
    Simulate federated local training with PyTorch.

    Each client trains an independent small neural network on its own samples.
    The resulting loss, accuracy, gradient norm, and gradient variance are
    stored as ModelUpdate records for the selected federated round.
    """
    from core.models import Client, ClientData, ModelUpdate

    if round_number is None:
        latest = ModelUpdate.objects.order_by('-round').values_list('round', flat=True).first()
        round_number = (latest or 0) + 1

    clients = list(Client.objects.order_by('client_id'))
    results = []
    for client in clients:
        rows = list(ClientData.objects.filter(client_id=client.client_id))
        if not rows:
            continue

        X = np.array([[
            r.feature_1, r.feature_2, r.feature_3, r.feature_4, r.feature_5,
            r.feature_6, r.feature_7, r.feature_8, r.feature_9, r.feature_10,
        ] for r in rows], dtype=np.float32)
        y = np.array([r.label for r in rows], dtype=np.int64)

        X = (X - X.mean(axis=0, keepdims=True)) / (X.std(axis=0, keepdims=True) + 1e-6)
        dataset = TensorDataset(torch.tensor(X), torch.tensor(y))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        model = SimpleClientNet()
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        last_loss = 0.0
        gradients = []
        for _ in range(epochs):
            for xb, yb in loader:
                optimizer.zero_grad()
                logits = model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                gradients.extend(_flat_gradients(model).tolist())
                optimizer.step()
                last_loss = float(loss.item())

        with torch.no_grad():
            logits = model(torch.tensor(X))
            pred = logits.argmax(dim=1).numpy()
            accuracy = float((pred == y).mean() * 100.0)

        grad_array = np.array(gradients or [0.0], dtype=np.float32)
        update_norm = float(np.linalg.norm(grad_array))
        gradient_variance = float(np.var(grad_array))

        ModelUpdate.objects.update_or_create(
            round=round_number,
            client_id=client.client_id,
            defaults={
                'loss': round(last_loss, 6),
                'accuracy': round(accuracy, 4),
                'update_norm': round(update_norm, 6),
                'gradient_variance': round(gradient_variance, 6),
            },
        )
        results.append({
            'client_id': client.client_id,
            'round': round_number,
            'loss': round(last_loss, 6),
            'accuracy': round(accuracy, 4),
            'update_norm': round(update_norm, 6),
            'gradient_variance': round(gradient_variance, 6),
        })

    return {
        'round': round_number,
        'client_count': len(results),
        'results': results,
        'summary': _summary(results),
    }


def _flat_gradients(model):
    parts = []
    for param in model.parameters():
        if param.grad is not None:
            parts.append(param.grad.detach().reshape(-1))
    if not parts:
        return torch.tensor([0.0])
    return torch.cat(parts)


def _summary(results):
    grouped = defaultdict(list)
    for item in results:
        grouped['accuracy'].append(item['accuracy'])
        grouped['loss'].append(item['loss'])
    if not results:
        return {'accuracy': 0, 'loss': 0}
    return {
        'accuracy': round(float(np.mean(grouped['accuracy'])), 4),
        'loss': round(float(np.mean(grouped['loss'])), 6),
    }
