
import torch
import torch.nn.functional as F

def pgd(model, x, y, epsilon=8/255, step=2/255, iters=10):
    x0 = x.clone().detach()
    x_adv = (x0 + torch.empty_like(x0).uniform_(-epsilon, epsilon)).clamp(0, 1)
    model.eval()
    for _ in range(iters):
        x_adv.requires_grad_(True)
        logits = model(x_adv)
        loss = F.cross_entropy(logits, y)
        grad = torch.autograd.grad(loss, x_adv, only_inputs=True)[0]
        with torch.no_grad():
            x_adv = x_adv + step * grad.sign()
            x_adv = torch.max(torch.min(x_adv, x0 + epsilon), x0 - epsilon)
            x_adv.clamp_(0, 1)
        x_adv = x_adv.detach()
    return x_adv.detach()
