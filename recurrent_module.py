# -*- coding: utf-8 -*-
"""
sourced from: me, myself, n I
"""
import torch
from torch.autograd import Variable
from torch import nn

import numpy as np

dtype = torch.FloatTensor
emb_dtype = torch.LongTensor
# dtype = torch.cuda.FloatTensor # Uncomment this to run on GPU
# emb_dtype = torch.cuda.LongTensor

data_size=256
emb_size=64
hidden_size=128
output_size=data_size
batch_size=128
seq_len=32



h_0 = Variable(torch.zeros(batch_size, hidden_size).type(dtype), requires_grad=False)
c_0 = Variable(torch.zeros(batch_size, hidden_size).type(dtype), requires_grad=False)
hidden_init = (h_0,c_0)

loss_fn=nn.CrossEntropyLoss()

class simple_lstm(nn.Module):
	def __init__(self):
		super(simple_lstm,self).__init__()
		self.add_module('emb',nn.Embedding(data_size,emb_size))
		self.add_module('rnn',nn.LSTMCell(emb_size,hidden_size))
		self.add_module('h2o',nn.Linear(hidden_size,output_size))

	def forward(self,inputs):
		hidden=hidden_init
		#resize hidden state batch size if we input a different size
		if hidden[0].size(0)!=inputs.size(0):
			hidden=tuple([x.narrow(0,0,inputs.size(0)) for x in hidden])
		loss=0
		# @ngimel it should run out of cuda memory part way through this loop on the second call to forward
		for t in range(inputs.size(1)-1):
			data=inputs[:,t]
			emb=self.emb(data)
			hidden=self.rnn(emb,hidden)
			out=self.h2o(hidden[0])
			loss+=loss_fn(out,inputs[:,t+1])

		return loss/((inputs.size(1)-1))


model = simple_lstm()

def eval_char_model(string,volatile=True):
	text=Variable(torch.ByteTensor(np.fromstring(string.encode('utf-8'),dtype=np.uint8)).type(emb_dtype),volatile=volatile).unsqueeze(0)
	return model(text).data.cpu()[0]

if __name__=='__main__':
	x=Variable(torch.from_numpy(np.random.randint(data_size,size=(batch_size,seq_len+1))).type(emb_dtype),requires_grad=False)

	learning_rate = 1e-6
	optimizer=torch.optim.Adam(model.parameters(),lr=learning_rate)
	for t in range(500):
	    loss = model(x)

	    print(t, loss.data.cpu()[0])

	    optimizer.zero_grad()

	    loss.backward()

	    optimizer.step()
