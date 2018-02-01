import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np

def linear(input, weight, bias=None):
	if bias is None:
		return F.linear(input, weight.cuda())
	else:
		return F.linear(input, weight.cuda(), bias.cuda())


def conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
	return F.conv2d(input, weight.cuda(), bias.cuda(), stride, padding, dilation, groups)


def relu(input):
	return F.threshold(input, 0, 0, inplace=True)


def maxpool(input, kernel_size, stride=None):
	return F.max_pool2d(input, kernel_size, stride)


def batchnorm(input, weight=None, bias=None, running_mean=None, running_var=None, training=True, eps=1e-5,
              momentum=0.1):
	''' momentum = 1 restricts stats to the current mini-batch '''
	# This hack only works when momentum is 1 and avoids needing to track running stats
	# by substuting dummy variables
	running_mean = torch.zeros(int(np.prod(np.array(input.data.size()[1])))).cuda()
	running_var = torch.ones(int(np.prod(np.array(input.data.size()[1])))).cuda()
	return F.batch_norm(input, running_mean, running_var, weight, bias, training, momentum, eps)


def bilinear_upsample(in_, factor):
	return F.upsample(in_, None, factor, 'bilinear')


def log_softmax(input):
	return F.log_softmax(input)


def count_correct(pred, target):
	''' count number of correct classification predictions in a batch '''
	pairs = [int(x == y) for (x, y) in zip(pred, target)]
	return sum(pairs)


def forward_pass(net, in_, target, weights=None):
	''' forward in_ through the net, return loss and output '''
	input_var = Variable(in_).cuda(async=True)
	target_var = Variable(target).cuda(async=True)
	out = net.net_forward(input_var, weights)
	loss = net.loss_fn(out, target_var)
	return loss, out


def evaluate(net, loader, weights=None):
	''' evaluate the net on the data in the loader '''
	num_correct = 0
	loss = 0
	for i, (in_, target) in enumerate(loader):
		batch_size = in_.numpy().shape[0]
		l, out = forward_pass(net, in_, target, weights)
		loss += l.data.cpu().numpy()[0]
		num_correct += count_correct(np.argmax(out.data.cpu().numpy(), axis=1), target.numpy())
	return float(loss) / len(loader), float(num_correct) / (len(loader) * batch_size)