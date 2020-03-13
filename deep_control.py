import os
import numpy as np
import torch
import torch.nn as nn
import time
from numpy.linalg import norm
import copy
import math
import argparse


class Net(nn.Module):
    
    def __init__(self, dim, nOut, n_layers, vNetWidth, activation = "relu"):
        super(Net, self).__init__()
        self.dim = dim
        self.nOut = nOut
        
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation=="tanh":
            self.activation = nn.Tanh()
        else:
            raise ValueError("unknown activation function {}".format(activation))
        
        self.i_h = self.hiddenLayerT1(dim, vNetWidth)
        self.h_h = nn.ModuleList([self.hiddenLayerT1(vNetWidth, vNetWidth) for l in range(n_layers-1)])
        self.h_o = self.outputLayer(vNetWidth, nOut)
        
    def hiddenLayerT1(self, nIn, nOut):
        layer = nn.Sequential(nn.Linear(nIn,nOut,bias=True),
                              self.activation)   
        return layer
    
    
    def outputLayer(self, nIn, nOut):
        layer = nn.Sequential(nn.Linear(nIn, nOut,bias=True))#,
        return layer
    
    def forward(self, S):
        h = self.i_h(S)
        for l in range(len(self.h_h)):
            h = self.h_h[l](h)
        output = self.h_o(h)
        return output


class control_solver(nn.Module):
    
    def __init__(self, dim, timegrid, n_layers, vNetWidth = 100, gradNetWidth=100):
        super(control_solver, self).__init__()
        self.dim = dim
        self.timegrid = torch.Tensor(timegrid).to(device)
        
        # Network for gradient
        self.net_timegrid_u = Net(dim=dim+1, nOut=dim, n_layers=n_layers, vNetWidth=vNetWidth)
            
    def forward(self, S0):        
        # in this case, error is the same as the objective functional!
        error = torch.zeros_like(S0)
        S_now = S0
        timegrid_mat = self.timegrid.view(1,-1).repeat(S0.shape[0],1)
        h = self.timegrid[1]-self.timegrid[0]
        dW_mat = math.sqrt(h)*torch.randn(S_now.data.size()[0], \
            S_now.data.size()[1]*len(self.timegrid), device=device)
        for i in range(1,len(self.timegrid)):
            # get the optimal u from the network
            input_val = torch.cat([timegrid_mat[:,(i-1)].view(-1,1), S_now],1)
            u_val = self.net_timegrid_u(input_val)
            dW = dW_mat[:, (i-1)*self.dim: i*self.dim]
            # the running cost
            error += ff_fun(timegrid_mat[:,i-1], S_now, u_val)*h
            # update the SDE
            S_now = S_now + b_fun(timegrid_mat[:,i-1], S_now, u_val)*h \
                    + sigma_fun(timegrid_mat[:,i-1], S_now, u_val)*dW
        # the terminal cost
        error += g_fun(S_now, u_val)
            
        return error

def g_fun(x, u):
    return(x*x)
    # # this is an implementation for Merton problem
    # return(torch.exp(-gamma*x))

def ff_fun(t, x, u):
    return(u*u)
    # # this is an implementation for Merton problem
    # return(0)

def b_fun(t, x, u):
    return(u)
    # # this is an implementation for Merton problem
    # return(u*(mu-r)+r*x)

def sigma_fun(t, x, u):
    return(u)
    # # this is an implementation for Merton problem
    # return(u*sigma)


def train():
    n_iter = 1000   
    for it in range(n_iter):
        model.train()
        lr = base_lr * (0.1**(it//n_iter))
        for param_group in optimizer.state_dict()['param_groups']:
            param_group['lr']=lr
        optimizer.zero_grad()
        z = torch.randn([batch_size, dim]).to(device)
        # initialized to be 0
        input = torch.ones(batch_size, dim, device=device)*x0
        init_time = time.time()
        error = model(input) 
        time_forward = time.time() - init_time
        loss = (1.0/batch_size)*(torch.sum(error))
        init_time = time.time()
        loss.backward()
        time_backward = time.time() - init_time
        optimizer.step()

        with open(file_log_path, 'a') as f:
            f.write('{}, {}, {}, {}\n'.format(it, loss.item(), time_forward, time_backward))

# we initialise weights using Xavier initialisation
def weight_initialise(m):
    if isinstance(m, nn.Linear):
        gain=nn.init.calculate_gain('relu')
        nn.init.xavier_uniform_(m.weight, gain)

if __name__ == '__main__':
    np.random.seed(1)
    # we read the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--vNetWidth', action="store", type=int, default=22, help="network width")
    parser.add_argument('--n-layers', action="store", type=int, default=2, help="number of layers")
    parser.add_argument('--timestep', action="store", type=float, default=0.01, help="timestep")
    parser.add_argument('--dim', action="store", type=int, default=1, help="dimension of the PDE")

    args = parser.parse_args()
    vNetWidth = args.vNetWidth
    n_layers = args.n_layers
    timestep = args.timestep
    dim = args.dim

    if torch.cuda.is_available():
        device = "cuda:0"
    else:
        device = "cpu"

    PATH_RESULTS = os.getcwd()
    print(os.getcwd())

    file_log_path = os.path.join(PATH_RESULTS, 'control_log.txt')
    with open(file_log_path, 'a') as f:
        f.write('iteration, loss, time forward pass, time backward pass\n')


    if torch.cuda.is_available():
        device = "cuda:0"
    else:
        device = "cpu"
    
    
    ##################
    # Problem setup ##
    ##################
    init_t, T = 0,1
    timegrid = np.arange(init_t, T+timestep/2, timestep)
    x0 = 0

    # # this is an implementation for Merton problem
    # gamma = 1
    # mu = 0.05
    # r = 0.03
    # sigma = 0.1
    # x0 = 1
    
    #########################
    # Network instantiation #
    #########################
    model = control_solver(dim=dim, timegrid=timegrid, n_layers=n_layers, vNetWidth=vNetWidth, gradNetWidth=vNetWidth)      
    model.apply(weight_initialise)
    model.to(device)
    
    ##################################
    # Network weights initialisation #
    ##################################
    model.apply(weight_initialise)
    
    #######################
    # training parameters #
    #######################
    batch_size = 5000
    base_lr = 0.001
    optimizer = torch.optim.Adam(model.parameters(),lr=base_lr, betas=(0.9, 0.999))
    train()