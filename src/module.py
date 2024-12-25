import torch
import torch.nn as nn
import torch.nn.functional as F

SQRT_CONSTL = 1e-10
SQRT_CONSTR = 1e10

def safe_sqrt(x, lbound=SQRT_CONSTL, rbound=SQRT_CONSTR):
    return torch.sqrt(torch.clamp(x, lbound, rbound))

class SimpleNN(nn.Module):
    def __init__(self, x_dim, hidden_dim=32):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(x_dim, hidden_dim) 
        self.fc2 = nn.Linear(hidden_dim, 1) 

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = self.fc2(x)
        return x
    
class FNetwork(nn.Module):
    def __init__(self, x_dim, hidden_dim=32):
        super(FNetwork, self).__init__()
        self.fc1 = nn.Linear(x_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim) 
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.elu(self.fc1(x))
        x = F.elu(self.fc2(x))
        x = self.fc3(x)
        output = torch.sigmoid(x) * 2 - 1
        return output
    
class Net(nn.Module):
    def __init__(self, n, x_dim, dropout, args):
        super(Net, self).__init__()

        dim_in = args.dim_in
        dim_out = args.dim_out

        activation=nn.ELU()

        self.rep_net = nn.Sequential(nn.Linear(x_dim, dim_in),
                                    activation,
                                    nn.Dropout(dropout),
                                    nn.Linear(dim_in, dim_in),
                                    activation,
                                    nn.Dropout(dropout))

        self.f_cost = SimpleNN(x_dim, dim_out)

        self.y0_net = nn.Sequential(nn.Linear(dim_in, dim_out),
                                    activation,
                                    nn.Dropout(dropout),
                                    nn.Linear(dim_out, 1),
                                    nn.Sigmoid())

        self.y1_net = nn.Sequential(nn.Linear(dim_in, dim_out),
                                    activation,
                                    nn.Dropout(dropout),
                                    nn.Linear(dim_out, 1),
                                    nn.Sigmoid())
    def output(self,x):
        h_rep = self.rep_net(x)
        h_rep_norm = h_rep / safe_sqrt(torch.sum(torch.square(h_rep), dim=1, keepdim=True))

        y0 = self.y0_net(h_rep_norm)
        y1 = self.y1_net(h_rep_norm)

        return y0, y1

    def forward(self, x, t):
        h_rep = self.rep_net(x)
        h_rep_norm = h_rep / safe_sqrt(torch.sum(torch.square(h_rep), dim=1, keepdim=True))

        y0 = self.y0_net(h_rep_norm)
        y1 = self.y1_net(h_rep_norm)
        y = t * y1 + (1-t) * y0

        f = self.f_cost(x)

        return h_rep_norm, y0, y1, y, f
