import os
import argparse
import time
import numpy as np
import imageio

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F

import animator

parser = argparse.ArgumentParser('Spiral Example 2')
parser.add_argument('--method', type=str, choices=['dopri5', 'adams'], default='dopri5')
parser.add_argument('--data_size', type=int, default=1000)
parser.add_argument('--batch_time', type=int, default=2)
parser.add_argument('--batch_size', type=int, default=20)
parser.add_argument('--niters', type=int, default=2000)
parser.add_argument('--test_freq', type=int, default=20)
parser.add_argument('--viz', action='store_true')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--adjoint', action='store_true')
args = parser.parse_args()

working_dir = "Rossler_results"
def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)

makedirs(working_dir)
import matplotlib.pyplot as plt

if args.adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint

device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')

true_u0 = torch.tensor([0.1, 0.0, -0.1]).to(device)
t = torch.linspace(0.0, 10.0, args.data_size).to(device)

class Lambda(nn.Module):

    def __init__(self):
        super(Lambda, self).__init__()
        self.a = nn.Parameter(torch.tensor([0.2]))
        self.b = nn.Parameter(torch.tensor([0.2]))
        self.c = nn.Parameter(torch.tensor([5.7]))

    def forward(self, t, u):
        x, y, z = u[0], u[1], u[2]
        du1 = -y - z
        du2 = x + (self.a[0] * y)
        du3 = self.b[0] + z * (x - self.c[0])
        return torch.stack([du1, du2, du3])


with torch.no_grad():
    true_fU = odeint(Lambda(), true_u0, t, method='rk4')


def get_batch():
    s = torch.from_numpy(np.random.choice(np.arange(args.data_size - args.batch_time, dtype=np.int64), args.batch_size, replace=False))
    batch_u0 = true_fU[s]  # (M, D)
    batch_t = t[:args.batch_time]  # (T)
    batch_u = torch.stack([true_fU[s + i] for i in range(args.batch_time)], dim=0)  # (T, M, D)
    return batch_u0.to(device), batch_t.to(device), batch_u.to(device)



def visualize_3d(true_fU=None, pred_fU=None, size=(10, 10)):

    fig = plt.figure(figsize=size)
    ax = fig.add_subplot(1, 1, 1, projection='3d')    


    if pred_fU != None:
        z = pred_fU.cpu().numpy()
        z = np.reshape(z, [-1, 3])
        for i in range(len(z)):
            ax.plot(z[i:i + 10, 0], z[i:i + 10, 1], z[i:i + 10, 2], color=plt.cm.jet(i / len(z) / 1.6))
                
    if true_fU != None:
        z = true_fU.cpu().numpy()
        z = np.reshape(z, [-1, 3])
        ax.scatter(z[:, 0], z[:, 1], z[:, 2], marker='.', color='k', alpha=0.5, linewidths=0, s=45)
            
        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.savefig(str(working_dir) + '/{:03d}'.format(itr), dpi=200, pad_inches=0.1)



class ODEFunc(nn.Module):

    def __init__(self, u_dim=3, n_hidden=256):
        super(ODEFunc, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(u_dim, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, u_dim)
        )

    def forward(self, t, u):
        return self.net(u)


class RunningAverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.val = None
        self.avg = 0

    def update(self, val):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val


if __name__ == '__main__':

    ii = 0
    scheme = 'rk4'
    if (os.path.exists(str(working_dir) + "/output.txt")):
        os.remove(str(working_dir) + "/output.txt")
    prog_out = open(str(working_dir) + "/output.txt", "a")
    prog_out.write("Operating with " + str(scheme) + "\n")

    func = ODEFunc().to(device)
    
    optimizer = optim.Adam(func.parameters(), lr=1e-2)
    end = time.time()

    time_meter = RunningAverageMeter(0.97)
    loss_meter = RunningAverageMeter(0.97)
    loss_arr = []
    time_arr = []

    for itr in range(1, args.niters + 1):
        start = time.time()
        optimizer.zero_grad()
        batch_u0, batch_t, batch_u = get_batch()
        pred_fU = odeint(func, batch_u0, batch_t, method=scheme).to(device)
        loss = F.mse_loss(pred_fU, batch_u)
        loss.backward()
        optimizer.step()
        time_arr.append(time.time()-start)

        time_meter.update(time.time() - end)
        loss_meter.update(loss.item())



        if itr % args.test_freq == 0:
            with torch.no_grad():
                pred_fU = odeint(func, true_fU[0], t, method=scheme)
                loss = F.mse_loss(pred_fU, true_fU)
                loss_arr.append(loss.item())
                prog_out.write('Iter {:04d} | Total Loss {:.6f}\n\n'.format(itr, loss.item()))
                visualize_3d(true_fU, pred_fU)
                ii += 1



        end = time.time()
    loss_arr = np.log(loss_arr)
    plt.clf()
    plt.plot(np.array(range(len(loss_arr))) * args.test_freq, loss_arr)
    plt.savefig(str(working_dir) + '/loss.png')
    prog_out.write("Average time per iteration = " + str(np.mean(np.array(time_arr))))
    prog_out.close()
    animator.make_gif(working_dir,scheme)


