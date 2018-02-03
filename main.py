from omniglotNShot import OmniglotNShot
from meta import MetaLearner
from omninet import OmniNet
from MiniImagenet import MiniImagenet
import torch
from torch.autograd import Variable
import numpy as np
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader





def main():
	meta_batchsz = 256
	n_way = 5
	k_shot = 5
	k_query = 5
	meta_lr = 1e-3
	num_updates = 5
	dataset = 'omniglot' # 'mini-imagenet'


	meta = MetaLearner(OmniNet, n_way=n_way, k_shot=k_shot, meta_batchsz=meta_batchsz, beta=meta_lr, num_updates=num_updates).cuda()

	if dataset == 'omniglot':
		db = OmniglotNShot('dataset', batchsz=meta_batchsz, n_way=n_way, k_shot=k_shot, k_query=k_query, imgsz=28)

	elif dataset == 'mini-imagenet':
		mini = MiniImagenet('../mini-imagenet/', mode='test', n_way=n_way, k_shot=k_shot, k_query=k_query,
		                        batchsz=600, resize=84)
		db = DataLoader(mini, meta_batchsz, shuffle=True, num_workers=2, pin_memory=True)

	tb = SummaryWriter('runs')

	for episode_num in range(200000):

		# 1. train
		if dataset == 'omniglot':
			support_x, support_y, query_x, query_y = db.get_batch('test')
			support_x = Variable(torch.from_numpy(support_x).float().transpose(2,4).transpose(3, 4).repeat(1,1,3,1,1)).cuda()
			query_x = Variable(torch.from_numpy(query_x).float().transpose(2,4).transpose(3, 4).repeat(1,1,3,1,1)).cuda()
			support_y = Variable(torch.from_numpy(support_y).long()).cuda()
			query_y = Variable(torch.from_numpy(query_y).long()).cuda()
		elif dataset == 'mini-imagenet':
			batch_test = iter(db).next()
			support_x = Variable(batch_test[0]).cuda()
			support_y = Variable(batch_test[1]).cuda()
			query_x = Variable(batch_test[2]).cuda()
			query_y = Variable(batch_test[3]).cuda()

		# backprop has been embeded in forward func.
		accs = meta(support_x, support_y, query_x, query_y)
		acc = np.array(accs).mean()

		# 2. test
		if episode_num % 10 == 0:
			test_accs = []
			for i in range(10):
				if dataset == 'omniglot':
					support_x, support_y, query_x, query_y = db.get_batch('test')
					support_x = Variable(torch.from_numpy(support_x).float().transpose(2,4).transpose(3, 4).repeat(1,1,3,1,1)).cuda()
					query_x = Variable(torch.from_numpy(query_x).float().transpose(2,4).transpose(3, 4).repeat(1,1,3,1,1)).cuda()
					support_y = Variable(torch.from_numpy(support_y).long()).cuda()
					query_y = Variable(torch.from_numpy(query_y).long()).cuda()
				elif dataset == 'mini-imagenet':
					batch_test = iter(db).next()
					support_x = Variable(batch_test[0]).cuda()
					support_y = Variable(batch_test[1]).cuda()
					query_x = Variable(batch_test[2]).cuda()
					query_y = Variable(batch_test[3]).cuda()

				test_acc = meta.pred(support_x, support_y, query_x, query_y)
				test_accs.append(test_acc)
			test_acc = np.array(test_accs).mean()
			print('episode:', episode_num, '\ttrain acc:%.6f'%acc, '\t\ttest acc:%.6f'%test_acc)
			tb.add_scalar('test-acc', test_acc)
			tb.add_scalar('train-acc', acc)



















if __name__ == '__main__':
	main()