# -*- coding: utf-8 -*-
"""
sourced from: http://pytorch.org/tutorials/beginner/pytorch_with_examples.html#pytorch-variables-and-autograd
"""
import torch
from torch.autograd import Variable
from torch.nn import Parameter

dtype = torch.FloatTensor
# dtype = torch.cuda.FloatTensor # Uncomment this to run on GPU

N, D_in, H, D_out = 64, 1000, 100, 10

x = Variable(torch.randn(N, D_in).type(dtype), requires_grad=False)
y = Variable(torch.randn(N, D_out).type(dtype), requires_grad=False)

w1 = Parameter(torch.randn(D_in, H).type(dtype))
w2 = Parameter(torch.randn(H, D_out).type(dtype))

learning_rate = 1e-6
optimizer=torch.optim.Adam([w1,w2],lr=learning_rate)
for t in range(500):
    y_pred = x.mm(w1).clamp(min=0).mm(w2)

    loss = (y_pred - y).pow(2).sum()
    print(t, loss.data[0])

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()