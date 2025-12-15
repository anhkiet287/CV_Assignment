
import torch

def accuracy_from_logits(logits, y):
    return (logits.argmax(1) == y).float().mean().item()
