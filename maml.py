import os, math, random
from collections import OrderedDict
import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
from utils import *
from omni import *

class OmniglotTask:

	def __init__(self, root, n_way, k_shot, mode='train'):
		self.dataset = 'omniglot'
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

	def __init__(self, num_classes, loss_fn, num_in_channels=3):
		super(OmniglotNet, self).__init__()
		# Define the network
		self.features = nn.Sequential(OrderedDict([
			('conv1', nn.Conv2d(num_in_channels, 64, 3)),
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
		self.loss_fn = loss_fn

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

	def __init__(self, num_classes, loss_fn, num_updates, step_size, batch_size, meta_batch_size, num_in_channels=3):
		super(InnerLoop, self).__init__(num_classes, loss_fn, num_in_channels)
		# Number of updates to be taken
		self.num_updates = num_updates

		# Step size for the updates
		self.step_size = step_size

		# PER CLASS Batch size for the updates
		self.batch_size = batch_size

		# for loss normalization
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

	def forward(self, task):
		train_loader = get_data_loader(task, self.batch_size)
		val_loader = get_data_loader(task, self.batch_size, split='val')
		##### Test net before training, should be random accuracy ####
		tr_pre_loss, tr_pre_acc = evaluate(self, train_loader)
		val_pre_loss, val_pre_acc = evaluate(self, val_loader)
		fast_weights = OrderedDict((name, param) for (name, param) in self.named_parameters())
		for i in range(self.num_updates):
			in_, target = train_loader.__iter__().next()
			if i == 0:
				loss, _ = self.forward_pass(in_, target)
				grads = torch.autograd.grad(loss, self.parameters(), create_graph=True)
			else:
				loss, _ = self.forward_pass(in_, target, fast_weights)
				grads = torch.autograd.grad(loss, fast_weights.values(), create_graph=True)
			fast_weights = OrderedDict(
				(name, param - self.step_size * grad) for ((name, param), grad) in zip(fast_weights.items(), grads))
		##### Test net after training, should be better than random ####
		tr_post_loss, tr_post_acc = evaluate(self, train_loader, fast_weights)
		val_post_loss, val_post_acc = evaluate(self, val_loader, fast_weights)
		# print('train', tr_pre_loss, tr_post_loss, tr_pre_acc, tr_post_acc)
		# print('val', val_pre_loss, val_post_loss, val_pre_acc, val_post_acc)

		# Compute the meta gradient and return it
		in_, target = val_loader.__iter__().next()
		loss, _ = self.forward_pass(in_, target, fast_weights)
		loss = loss / self.meta_batch_size  # normalize loss
		grads = torch.autograd.grad(loss, self.parameters())
		meta_grads = {name: g for ((name, _), g) in zip(self.named_parameters(), grads)}
		metrics = (tr_post_loss, tr_post_acc, val_post_loss, val_post_acc)
		return metrics, meta_grads








class MetaLearner(object):
	def __init__(self,
	             dataset, # ominiglot
	             num_classes, # n-way
	             num_inst, # k-shot
	             meta_batch_size, # 32
 	             meta_step_size, #  lr
	             inner_batch_size, # 1
	             inner_step_size, # lr
	             num_updates, #  15000
	             num_inner_updates, # 5
	             loss_fn): # crossentropy
		super(self.__class__, self).__init__()
		self.dataset = dataset
		self.num_classes = num_classes
		self.num_inst = num_inst
		self.meta_batch_size = meta_batch_size
		self.meta_step_size = meta_step_size
		self.inner_batch_size = inner_batch_size
		self.inner_step_size = inner_step_size
		self.num_updates = num_updates
		self.num_inner_updates = num_inner_updates
		self.loss_fn = loss_fn

		# Make the nets
		# TODO: don't actually need two nets
		num_input_channels = 1 if self.dataset == 'mnist' else 3

		self.net = OmniglotNet(num_classes, self.loss_fn, num_input_channels)
		self.net.cuda()
		self.fast_net = InnerLoop(num_classes, self.loss_fn, self.num_inner_updates, self.inner_step_size,
		                          self.inner_batch_size, self.meta_batch_size, num_input_channels)
		self.fast_net.cuda()
		self.opt = torch.optim.Adam(self.net.parameters(), lr=meta_step_size)

	def get_task(self, root, n_cl, n_inst, split='train'):
		return OmniglotTask(root, n_cl, n_inst, split)

	def meta_update(self, task, ls):

		loader = get_data_loader(task, self.inner_batch_size, split='val')
		in_, target = loader.__iter__().next()
		# We use a dummy forward / backward pass to get the correct grads into self.net
		loss, out = forward_pass(self.net, in_, target)
		# Unpack the list of grad dicts
		gradients = {k: sum(d[k] for d in ls) for k in ls[0].keys()}
		# Register a hook on each parameter in the net that replaces the current dummy grad
		# with our grads accumulated across the meta-batch
		hooks = []
		for (k, v) in self.net.named_parameters():
			def get_closure():
				key = k

				def replace_grad(grad):
					return gradients[key]

				return replace_grad

			hooks.append(v.register_hook(get_closure()))
		# Compute grads for current step, replace with summed gradients as defined by hook
		self.opt.zero_grad()
		loss.backward()
		# Update the net parameters with the accumulated gradient according to optimizer
		self.opt.step()
		# Remove the hooks before next training phase
		for h in hooks:
			h.remove()

	def test(self):
		num_in_channels = 1 if self.dataset == 'mnist' else 3

		test_net = OmniglotNet(self.num_classes, self.loss_fn, num_in_channels)
		mtr_loss, mtr_acc, mval_loss, mval_acc = 0.0, 0.0, 0.0, 0.0

		# Select ten tasks randomly from the test set to evaluate on
		for _ in range(10):
			# Make a test net with same parameters as our current net
			test_net.copy_weights(self.net)
			test_net.cuda()
			test_opt = torch.optim.SGD(test_net.parameters(), lr=self.inner_step_size)
			task = self.get_task('../data/{}'.format(self.dataset), self.num_classes, self.num_inst, split='test')
			# Train on the train examples, using the same number of updates as in training
			train_loader = get_data_loader(task, self.inner_batch_size, split='train')
			for i in range(self.num_inner_updates):
				in_, target = train_loader.__iter__().next()
				loss, _ = forward_pass(test_net, in_, target)
				test_opt.zero_grad()
				loss.backward()
				test_opt.step()
			# Evaluate the trained model on train and val examples
			tloss, tacc = evaluate(test_net, train_loader)
			val_loader = get_data_loader(task, self.inner_batch_size, split='val')
			vloss, vacc = evaluate(test_net, val_loader)
			mtr_loss += tloss
			mtr_acc += tacc
			mval_loss += vloss
			mval_acc += vacc

		mtr_loss = mtr_loss / 10
		mtr_acc = mtr_acc / 10
		mval_loss = mval_loss / 10
		mval_acc = mval_acc / 10

		print('Meta train:', mtr_loss, mtr_acc)
		print('Meta val:', mval_loss, mval_acc)
		del test_net
		return mtr_loss, mtr_acc, mval_loss, mval_acc

	def _train(self, exp):
		''' debugging function: learn two tasks '''
		task1 = self.get_task('../data/{}'.format(self.dataset), self.num_classes, self.num_inst)
		task2 = self.get_task('../data/{}'.format(self.dataset), self.num_classes, self.num_inst)
		for it in range(self.num_updates):
			grads = []
			for task in [task1, task2]:
				# Make sure fast net always starts with base weights
				self.fast_net.copy_weights(self.net)
				_, g = self.fast_net.forward(task)
				grads.append(g)
			self.meta_update(task, grads)

	def train(self, exp):
		tr_loss, tr_acc, val_loss, val_acc = [], [], [], []
		mtr_loss, mtr_acc, mval_loss, mval_acc = [], [], [], []
		for it in range(self.num_updates): # 15000
			# Evaluate on test tasks
			mt_loss, mt_acc, mv_loss, mv_acc = self.test()
			mtr_loss.append(mt_loss)
			mtr_acc.append(mt_acc)
			mval_loss.append(mv_loss)
			mval_acc.append(mv_acc)

			# Collect a meta batch update
			grads = []
			tloss, tacc, vloss, vacc = 0.0, 0.0, 0.0, 0.0
			for i in range(self.meta_batch_size): # 32
				task = self.get_task('../data/{}'.format(self.dataset), self.num_classes, self.num_inst)
				self.fast_net.copy_weights(self.net)
				metrics, g = self.fast_net.forward(task)
				(trl, tra, vall, vala) = metrics
				grads.append(g)
				tloss += trl
				tacc += tra
				vloss += vall
				vacc += vala

			# Perform the meta update
			print('Meta update:', it)
			self.meta_update(task, grads)

			# Save a model snapshot every now and then
			if it % 500 == 0:
				torch.save(self.net.state_dict(), '../output/{}/train_iter_{}.pth'.format(exp, it))

			# Save stuff
			tr_loss.append(tloss / self.meta_batch_size)
			tr_acc.append(tacc / self.meta_batch_size)
			val_loss.append(vloss / self.meta_batch_size)
			val_acc.append(vacc / self.meta_batch_size)


def main():
	exp='maml-omniglot-5way-1shot-TEST'
	dataset='omniglot'
	num_cls=20
	num_inst=5
	batch=5
	m_batch=32
	num_updates=15000
	num_inner_updates=5
	lr=1e-1
	meta_lr=1e-3

	# make output dir
	output = '../output/{}'.format(exp)
	try:
		os.makedirs(output)
	except:
		pass

	loss_fn = nn.CrossEntropyLoss()
	learner = MetaLearner(dataset, num_cls, num_inst, m_batch, meta_lr, batch, lr, num_updates,num_inner_updates, loss_fn)
	learner.train(exp)


if __name__ == '__main__':
	main()
