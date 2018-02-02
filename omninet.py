import torch
from torch import nn
from torch import optim
from torch import autograd
from torch.autograd import Variable
from torch.nn import functional as F


class OmniNet(nn.Module):

	def __init__(self, n_way):
		super(OmniNet, self).__init__()

		self.net = nn.Sequential(nn.Conv2d(3, 64, kernel_size=3),
		                         nn.AvgPool2d(kernel_size=2),
		                         nn.BatchNorm2d(64),
		                         nn.ReLU(inplace=True),

		                         nn.Conv2d(64, 64, kernel_size=3),
		                         nn.AvgPool2d(kernel_size=2),
		                         nn.BatchNorm2d(64),
		                         nn.ReLU(inplace=True),

		                         nn.Conv2d(64, 64, kernel_size=3),
		                         nn.BatchNorm2d(64),
		                         nn.ReLU(inplace=True),

		                         nn.Conv2d(64, 64, kernel_size=3),
		                         nn.BatchNorm2d(64),
		                         nn.ReLU(inplace=True),
		                         )
		self.fc = nn.Linear(64, n_way)

		self.criteon = nn.CrossEntropyLoss()

	def forward(self, x, target):
		x = self.net(x)
		x = x.view(-1, 64)
		pred = self.fc(x)
		loss = self.criteon(pred, target)

		return loss, pred