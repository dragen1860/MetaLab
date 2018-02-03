import os, math, random
from collections import OrderedDict
import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
from utils import *
from omni import *
from omniglotNShot import OmniglotNShot

class OmniglotTask:

	def __init__(self, n_way, k_shot, mode='train'):
		root = '../data/omniglot/'
		# Alphabet_of_the_Magi/chapter01/1.jpg
		self.root = '{}/images_background'.format(root) if mode == 'train' else '{}/images_evaluation'.format(root)
		self.n_way = n_way
		self.k_shot = k_shot
		self.k_query = k_shot

		# Sample num_cls characters and num_inst instances of each
		languages = os.listdir(self.root)
		chars = []
		for l in languages:
			chars += [os.path.join(l, x) for x in os.listdir(os.path.join(self.root, l))]
		classes = random.sample(chars, n_way)
		labels = np.arange(n_way)
		path2label = dict(zip(classes, labels))

		# Now sample from the chosen classes to create class-balanced train and val sets
		self.train_ids = []
		self.val_ids = []
		for c in classes:
			# First get all isntances of that class
			all_imgs = [os.path.join(c, x) for x in os.listdir(os.path.join(self.root, c))]
			random.shuffle(all_imgs)

			# Sample num_inst instances randomly each for train and val
			self.train_ids += all_imgs[ : k_shot]
			self.val_ids += all_imgs[k_shot : k_shot + self.k_query]

		# Keep instances separated by class for class-balanced mini-batches
		self.train_labels = [path2label[self.get_class(x)] for x in self.train_ids]
		self.val_labels = [path2label[self.get_class(x)] for x in self.val_ids]

	def get_class(self, path):
		return os.path.join( * path.split('/')[:-1])


class OmniglotNet(nn.Module):
	'''
	The base model for few-shot learning on Omniglot
	'''

	def __init__(self, num_classes):
		super(OmniglotNet, self).__init__()
		# Define the network
		self.features = nn.Sequential(OrderedDict([
			('conv1', nn.Conv2d(3, 64, 3)),
			('bn1', nn.BatchNorm2d(64, momentum=1, affine=True)),
			('relu1', nn.ReLU(inplace=True)),
			('pool1', nn.MaxPool2d(2, 2)),
			('conv2', nn.Conv2d(64, 64, 3)),
			('bn2', nn.BatchNorm2d(64, momentum=1, affine=True)),
			('relu2', nn.ReLU(inplace=True)),
			('pool2', nn.MaxPool2d(2, 2)),
			('conv3', nn.Conv2d(64, 64, 3)),
			('bn3', nn.BatchNorm2d(64, momentum=1, affine=True)),
			('relu3', nn.ReLU(inplace=True)),
			('pool3', nn.MaxPool2d(2, 2))
		]))
		self.add_module('fc', nn.Linear(64, num_classes))

		# Define loss function
		self.loss_fn = nn.CrossEntropyLoss()

		# Initialize weights
		self._init_weights()

	def forward(self, x, weights = None):

		if weights == None:
			x = self.features(x)
			x = x.view(x.size(0), 64)
			x = self.fc(x)
		else:
			x = conv2d(x, weights['features.conv1.weight'], weights['features.conv1.bias'])
			x = batchnorm(x, weight=weights['features.bn1.weight'], bias=weights['features.bn1.bias'], momentum=1)
			x = relu(x)
			x = maxpool(x, kernel_size=2, stride=2)
			x = conv2d(x, weights['features.conv2.weight'], weights['features.conv2.bias'])
			x = batchnorm(x, weight=weights['features.bn2.weight'], bias=weights['features.bn2.bias'], momentum=1)
			x = relu(x)
			x = maxpool(x, kernel_size=2, stride=2)
			x = conv2d(x, weights['features.conv3.weight'], weights['features.conv3.bias'])
			x = batchnorm(x, weight=weights['features.bn3.weight'], bias=weights['features.bn3.bias'], momentum=1)
			x = relu(x)
			x = maxpool(x, kernel_size=2, stride=2)
			x = x.view(x.size(0), 64)
			x = linear(x, weights['fc.weight'], weights['fc.bias'])
		return x

	def net_forward(self, x, weights=None):
		return self.forward(x, weights)

	def _init_weights(self):
		''' Set weights to Gaussian, biases to zero '''
		torch.manual_seed(1337)
		torch.cuda.manual_seed(1337)
		torch.cuda.manual_seed_all(1337)

		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
				m.weight.data.normal_(0, math.sqrt(2. / n))
				if m.bias is not None:
					m.bias.data.zero_()
			elif isinstance(m, nn.BatchNorm2d):
				m.weight.data.fill_(1)
				m.bias.data.zero_()
			elif isinstance(m, nn.Linear):
				n = m.weight.size(1)
				m.weight.data.normal_(0, 0.01)
				# m.bias.data.zero_() + 1
				m.bias.data = torch.ones(m.bias.data.size())

	def copy_weights(self, net):
		''' Set this module's weights to be the same as those of 'net' '''
		# TODO: breaks if nets are not identical
		# TODO: won't copy buffers, e.g. for batch norm
		for m_from, m_to in zip(net.modules(), self.modules()):
			if isinstance(m_to, nn.Linear) or isinstance(m_to, nn.Conv2d) or isinstance(m_to, nn.BatchNorm2d):
				m_to.weight.data = m_from.weight.data.clone()
				if m_to.bias is not None:
					m_to.bias.data = m_from.bias.data.clone()






class InnerLoop(OmniglotNet):
	'''
	This module performs the inner loop of MAML
	The forward method updates weights with gradient steps on training data,
	then computes and returns a meta-gradient w.r.t. validation data
	'''

	def __init__(self, num_classes, num_updates, step_size, meta_batch_size):
		super(InnerLoop, self).__init__(num_classes)

		self.num_updates = num_updates
		self.step_size = step_size
		self.meta_batch_size = meta_batch_size

	def net_forward(self, x, weights=None):
		return super(InnerLoop, self).forward(x, weights)

	def forward_pass(self, in_, target, weights=None):
		''' Run data through net, return loss and output '''
		input_var = torch.autograd.Variable(in_).cuda(async=True)
		target_var = torch.autograd.Variable(target).cuda(async=True)
		# Run the batch through the net, compute loss
		out = self.net_forward(input_var, weights)
		loss = self.loss_fn(out, target_var)
		return loss, out

	def forward(self, support_x, support_y, query_x, query_y):

		fast_weights = OrderedDict((name, param) for (name, param) in self.named_parameters())


		for i in range(self.num_updates):
			if i == 0:
				loss, _ = self.forward_pass(support_x, support_y)
				grads = torch.autograd.grad(loss, self.parameters(), create_graph=True)
			else:
				loss, _ = self.forward_pass(support_x, support_y, fast_weights)
				grads = torch.autograd.grad(loss, fast_weights.values(), create_graph=True)
			fast_weights = OrderedDict(
				(name, param - self.step_size * grad) for ((name, param), grad) in zip(fast_weights.items(), grads))


		# Compute the meta gradient and return it
		loss, _ = self.forward_pass(query_x, query_y, fast_weights)
		loss = loss / self.meta_batch_size  # normalize loss
		grads = torch.autograd.grad(loss, self.parameters())
		meta_grads = {name: g for ((name, _), g) in zip(self.named_parameters(), grads)}
		return meta_grads



class MetaLearner:
	def __init__(self, n_way,  k_shot,  meta_batchsz,  beta, alpha,  num_updates):
		super(self.__class__, self).__init__()
		self.n_way = n_way
		self.k_shot = k_shot
		self.meta_batchsz = meta_batchsz
		self.beta = beta
		self.alpha = alpha
		self.num_updates = num_updates

		self.db = OmniglotNShot('dataset', batchsz=meta_batchsz, n_way=n_way, k_shot=k_shot, k_query=k_shot, imgsz=28)

		self.net = OmniglotNet(n_way)
		self.net.cuda()
		self.fast_net = InnerLoop(n_way,num_updates,alpha,meta_batchsz)
		self.fast_net.cuda()
		self.opt = torch.optim.Adam(self.net.parameters(), lr=beta)


	def meta_update(self, task, ls):


		support_x, support_y, query_x, query_y = self.db.get_batch('test')
		support_x = torch.from_numpy(support_x).float().transpose(2, 4).transpose(3, 4).repeat(1, 1, 3, 1, 1)
		query_x = torch.from_numpy(query_x).float().transpose(2, 4).transpose(3, 4).repeat(1, 1, 3, 1, 1)
		support_y = torch.from_numpy(support_y).long()
		query_y = torch.from_numpy(query_y).long()

		# We use a dummy forward / backward pass to get the correct grads into self.net
		loss, out = forward_pass(self.net, support_x[0], support_y[0])
		# Unpack the list of grad dicts
		gradients = {k: sum(d[k] for d in ls) for k in ls[0].keys()}
		# Register a hook on each parameter in the net that replaces the current dummy grad
		# with our grads accumulated across the meta-batch
		hooks = []

		for (k, v) in self.net.named_parameters():
			def get_closure():
				key = k
				return lambda grad:gradients[key]

			hooks.append(v.register_hook(get_closure()))

		self.opt.zero_grad()
		loss.backward()
		self.opt.step()


		for h in hooks:
			h.remove()

	def test(self):

		test_net = OmniglotNet(self.n_way)
		mval_acc = 0.0

		support_x, support_y, query_x, query_y = self.db.get_batch('test')
		support_x = torch.from_numpy(support_x).float().transpose(2, 4).transpose(3, 4).repeat(1, 1, 3, 1, 1)
		query_x = torch.from_numpy(query_x).float().transpose(2, 4).transpose(3, 4).repeat(1, 1, 3, 1, 1)
		support_y = torch.from_numpy(support_y).long()
		query_y = torch.from_numpy(query_y).long()

		for meta_batchidx in range(support_y.size(0)):
			# Make a test net with same parameters as our current net
			test_net.copy_weights(self.net)
			test_net.cuda()

			test_opt = torch.optim.SGD(test_net.parameters(), lr=self.alpha)

			for i in range(self.num_updates):
				loss, _ = forward_pass(test_net, support_x[meta_batchidx], support_y[meta_batchidx])
				test_opt.zero_grad()
				loss.backward()
				test_opt.step()

			vloss, vacc = evaluate(test_net, query_x[meta_batchidx], query_y[meta_batchidx])
			mval_acc += vacc

		mval_acc = mval_acc / support_y.size(0)
		print(mval_acc)


	def train(self):

		for it in range(15000): # 15000
			self.test()

			support_x, support_y, query_x, query_y = self.db.get_batch('train')
			support_x = torch.from_numpy(support_x).float().transpose(2, 4).transpose(3, 4).repeat(1, 1, 3, 1, 1)
			query_x = torch.from_numpy(query_x).float().transpose(2, 4).transpose(3, 4).repeat(1, 1, 3, 1, 1)
			support_y = torch.from_numpy(support_y).long()
			query_y = torch.from_numpy(query_y).long()


			grads = []
			for i in range(self.meta_batchsz): # 32

				task = OmniglotTask(self.n_way, self.k_shot, mode='train')
				self.fast_net.copy_weights(self.net)

				grads.append(self.fast_net.forward(support_x[i], support_y[i], query_x[i], query_y[i]))

			self.meta_update(task, grads)


def main():
	learner = MetaLearner(n_way=5,k_shot=5,meta_batchsz=32,beta=1e-3,alpha=1e-1,num_updates=5)
	learner.train()


if __name__ == '__main__':
	main()
