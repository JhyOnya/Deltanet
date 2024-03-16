import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class p_zu_x_func(nn.Module):
    def __init__(self, dim_in=5 + 15, nh=2, dim_h=20, dim_out=128):
        super().__init__()
        self.nh = nh
        self.dim_in = dim_in
        self.dim_out = dim_out

        self.att = torch.nn.Parameter(torch.zeros(dim_in), requires_grad=True)

        self.input_z = nn.Linear(dim_in, dim_h)
        self.hidden_z = nn.ModuleList([nn.Linear(dim_h, dim_h) for _ in range(nh)])
        self.output_z = nn.Linear(dim_h, dim_out)

        self.input_u = nn.Linear(dim_in, dim_h)
        self.hidden_u = nn.ModuleList([nn.Linear(dim_h, dim_h) for _ in range(nh)])
        self.output_u = nn.Linear(dim_h, dim_out)
        self.eye = torch.eye(dim_in).cuda()

    def forward(self, x):
        att = torch.diag(torch.sigmoid(self.att))
        att_ = self.eye - att

        x_z = torch.mm(x, att)
        pre_z = F.elu(self.input_z(x_z))
        for i in range(self.nh):
            pre_z = F.elu(self.hidden_z[i](pre_z))
        z = self.output_z(pre_z)

        x_u = torch.mm(x, att_)
        pre_u = F.elu(self.input_u(x_u))
        for i in range(self.nh):
            pre_u = F.elu(self.hidden_u[i](pre_u))
        u = self.output_u(pre_u)

        return z, u, self.att.cpu().detach().numpy()


class p_t_z_func(nn.Module):
    def __init__(self, dim_in=20, nh=2, dim_h=20, dim_out=1):
        super().__init__()
        self.nh = nh
        self.dim_out = dim_out

        self.input = nn.Linear(dim_in, dim_h)
        self.hidden = nn.ModuleList([nn.Linear(dim_h, dim_h) for _ in range(nh)])
        self.output = nn.Linear(dim_h, dim_out)

    def forward(self, z):
        pre = F.elu(self.input(z))
        for i in range(self.nh):
            pre = F.elu(self.hidden[i](pre))
        pre = self.output(pre)
        return torch.sigmoid(pre)


class p_y_tu_func(nn.Module):

    def __init__(self, dim_in=20, nh=2, dim_h=20, dim_out=1):
        super().__init__()
        self.nh = nh
        self.dim_out = dim_out

        self.input = nn.Linear(dim_in, dim_h)
        self.hidden = nn.ModuleList([nn.Linear(dim_h, dim_h) for _ in range(nh)])

        self.mu_t0 = nn.Linear(dim_h, dim_out)
        self.mu_t1 = nn.Linear(dim_h, dim_out)

    def forward(self, t, u):
        pre = F.elu(self.input(u))
        for i in range(self.nh):
            pre = F.elu(self.hidden[i](pre))

        mu_t0 = F.elu(self.mu_t0(pre))
        mu_t1 = F.elu(self.mu_t1(pre))
        y = (1 - t) * mu_t0 + t * mu_t1

        return y
