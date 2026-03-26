import torch
import torch.nn as nn

class PolicyWrapper(torch.nn.Module):
    def __init__(self, matgcn, T, device):
        super().__init__()
        self.matgcn = matgcn
        self.T = T
        self.device = device
        self.buffer = None

    def reset(self):
        self.buffer = None

    def forward(self, states, laplacian):
        # states: [B, N, F]

        states = states.unsqueeze(1)  # [B, 1, N, F]

        if self.buffer is None:
            self.buffer = states.repeat(1, self.T, 1, 1)
        else:
            self.buffer = torch.cat(
                [self.buffer[:, 1:], states],
                dim=1
            )

        # no external features in simulator
        external = torch.zeros(states.size(0), self.T, 0).to(self.device)

        logits = self.matgcn(self.buffer, external, laplacian)

        return logits  # [B, N, 3]
    
def sample_actions(logits, padding_mask):
    probs = torch.softmax(logits, dim=-1)

    actions = torch.multinomial(
        probs.view(-1, 3), 1
    ).view(logits.size(0), logits.size(1))

    actions[padding_mask] = 0
    return actions

def compute_laplacian(adj):
    D = torch.diag(torch.sum(adj, dim=1))
    L = D - adj
    D_inv_sqrt = torch.diag(1.0 / torch.sqrt(torch.sum(adj, dim=1) + 1e-6))
    return D_inv_sqrt @ L @ D_inv_sqrt