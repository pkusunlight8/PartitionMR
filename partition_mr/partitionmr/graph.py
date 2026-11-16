"""Optional equivariant graph architecture, extracted from the research implementation."""
import torch


class OddGraph(torch.nn.Module):
    def __init__(self, width=48, layers=4):
        super().__init__()
        self.start = torch.nn.Linear(2, width, bias=False)
        self.local = torch.nn.ModuleList([torch.nn.Linear(width, width, bias=False) for _ in range(layers)])
        self.message = torch.nn.ModuleList([torch.nn.Linear(width, width, bias=False) for _ in range(layers)])
        self.gates = torch.nn.ModuleList([torch.nn.Sequential(torch.nn.Linear(5, 32), torch.nn.SiLU(),
                                                             torch.nn.Linear(32, width), torch.nn.Sigmoid()) for _ in range(layers)])
        self.finish = torch.nn.Linear(width, 1, bias=False)

    def forward(self, z, R, mu, sigma):
        m = z.shape[1]
        I = torch.eye(m, device=z.device)[None]
        inverse = torch.linalg.inv(R + sigma[:, None, None] ** 2 * I)
        base = sigma[:, None] ** 2 * torch.einsum('nij,nj->ni', inverse, z)
        field = torch.einsum('nij,nj->ni', inverse, z)
        v = torch.tanh(self.start(torch.stack([z / 5, field / 5], dim=-1)))
        invariant = torch.stack([z ** 2 / 25, mu[:, None].expand_as(z) / 4,
                                 sigma[:, None].expand_as(z), inverse.diagonal(dim1=1, dim2=2)], dim=-1)
        adjacency = inverse / (inverse.abs().sum(dim=-1, keepdim=True) + 1e-8)
        for local, message, gate in zip(self.local, self.message, self.gates):
            g = gate(torch.cat([invariant, (v ** 2).mean(-1, keepdim=True)], dim=-1))
            v = v + g * torch.tanh(local(v) + message(torch.bmm(adjacency, v)))
        magnetization = torch.tanh(self.finish(v).squeeze(-1))
        return base + mu[:, None] * torch.einsum('nij,nj->ni', I - sigma[:, None, None] ** 2 * inverse, magnetization)

