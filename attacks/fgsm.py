
import torch
import torch.nn.functional as F

def fgsm(model, x, y, epsilon: float):
    model.eval()
    x_adv = x.clone().detach().requires_grad_(True)
    logits = model(x_adv)
    loss = F.cross_entropy(logits, y)
    model.zero_grad(set_to_none=True)
    loss.backward()
    grad_sign = x_adv.grad.detach().sign()
    x_adv = (x_adv + epsilon * grad_sign).clamp(0.0, 1.0).detach()
    return x_adv
