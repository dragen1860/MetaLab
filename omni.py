import random
import numpy as np
from PIL import Image

import torch
import torch.utils.data as data
from torch.utils.data import DataLoader
from torch.utils.data.sampler import Sampler
import torchvision.transforms as transforms


class FewShotDataset(data.Dataset):
	"""
	Load image-label pairs from a task to pass to Torch DataLoader
	Tasks consist of data and labels split into train / val splits
	"""

	def __init__(self, task, split='train', transform=None, target_transform=None):
		self.transform = transform  # Torch operations on the input image
		self.target_transform = target_transform
		self.task = task
		self.root = self.task.root
		self.split = split
		self.img_ids = self.task.train_ids if self.split == 'train' else self.task.val_ids
		self.labels = self.task.train_labels if self.split == 'train' else self.task.val_labels

	def __len__(self):
		return len(self.img_ids)

	def __getitem__(self, idx):
		raise NotImplementedError("This is an abstract class. Subclass this class for your particular dataset.")


class Omniglot(FewShotDataset):
	def __init__(self, *args, **kwargs):
		super(Omniglot, self).__init__(*args, **kwargs)

	def load_image(self, idx):
		''' Load image '''
		im = Image.open('{}/{}'.format(self.root, idx)).convert('RGB')
		im = im.resize((28, 28), resample=Image.LANCZOS)  # per Chelsea's implementation
		im = np.array(im, dtype=np.float32)
		return im

	def __getitem__(self, idx):
		img_id = self.img_ids[idx]
		im = self.load_image(img_id)
		if self.transform is not None:
			im = self.transform(im)
		label = self.labels[idx]
		if self.target_transform is not None:
			label = self.target_transform(label)
		return im, label



class ClassBalancedSampler(Sampler):
	'''
	Samples class-balanced batches from 'num_cl' pools each
	of size 'num_inst'
	If 'batch_cutoff' is None, indices for iterating over batches
	of the entire dataset will be returned
	Otherwise, indices for the number of batches up to the batch_cutoff
	will be returned
	(This is to allow sampling with replacement across training iterations)
	'''

	def __init__(self, num_cl, num_inst, batch_cutoff=None):
		self.num_cl = num_cl
		self.num_inst = num_inst
		self.batch_cutoff = batch_cutoff

	def __iter__(self):
		'''return a single list of indices, assuming that items will be grouped by class '''
		# First construct batches of 1 instance per class
		batches = [[i + j * self.num_inst for i in torch.randperm(self.num_inst)] for j in range(self.num_cl)]
		batches = [[batches[j][i] for j in range(self.num_cl)] for i in range(self.num_inst)]
		# Shuffle within each batch so that classes don't always appear in same order
		for sublist in batches:
			random.shuffle(sublist)

		if self.batch_cutoff is not None:
			random.shuffle(batches)
			batches = batches[:self.batch_cutoff]

		batches = [item for sublist in batches for item in sublist]

		return iter(batches)

	def __len__(self):
		return 1


def get_data_loader(task, split='train'):
	# NOTE: batch size here is # instances PER CLASS

	dset = Omniglot(task, transform=transforms.Compose([
					transforms.ToTensor(),
					transforms.Normalize(mean=[0.92206, 0.92206, 0.92206], std=[0.08426, 0.08426, 0.08426])]),
				                split=split)

	sampler = ClassBalancedSampler(task.n_way, task.k_shot, batch_cutoff=(None if split != 'train' else 1))
	loader = DataLoader(dset, batch_size=task.n_way*task.k_shot)
	return loader