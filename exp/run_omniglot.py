import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from helper.omniglotNShot import OmniglotNShot
from meta.reptile import MetaLearner
from mdl.naive import Naive

import torch
from torch.autograd import Variable
import numpy as np
from tensorboardX import SummaryWriter


def main():
	meta_batchsz = 32
	n_way = 5
	k_shot = 1
	k_query = k_shot
	meta_lr = 1e-3
	num_updates = 5


	imgsz = 28
	db = OmniglotNShot('dataset', batchsz=meta_batchsz, n_way=n_way, k_shot=k_shot, k_query=k_query, imgsz=imgsz)



	meta = MetaLearner(Naive, (n_way, imgsz), n_way=n_way, k_shot=k_shot, meta_batchsz=meta_batchsz, beta=meta_lr,
	                   num_updates=num_updates).cuda()

	tb = SummaryWriter('runs')


	# main loop
	for episode_num in range(200000):

		# 1. train
		support_x, support_y, query_x, query_y = db.get_batch('test')
		support_x = Variable( torch.from_numpy(support_x).float().transpose(2, 4).transpose(3, 4).repeat(1, 1, 3, 1, 1)).cuda()
		query_x = Variable( torch.from_numpy(query_x).float().transpose(2, 4).transpose(3, 4).repeat(1, 1, 3, 1, 1)).cuda()
		support_y = Variable(torch.from_numpy(support_y).long()).cuda()
		query_y = Variable(torch.from_numpy(query_y).long()).cuda()


		# backprop has been embeded in forward func.
		accs = meta(support_x, support_y, query_x, query_y)
		train_acc = np.array(accs).mean()

		# 2. test
		if episode_num % 30 == 0:
			test_accs = []
			for i in range(min(episode_num // 5000 + 3, 10)): # get average acc.

				support_x, support_y, query_x, query_y = db.get_batch('test')
				support_x = Variable( torch.from_numpy(support_x).float().transpose(2, 4).transpose(3, 4).repeat(1, 1, 3, 1, 1)).cuda()
				query_x = Variable( torch.from_numpy(query_x).float().transpose(2, 4).transpose(3, 4).repeat(1, 1, 3, 1, 1)).cuda()
				support_y = Variable(torch.from_numpy(support_y).long()).cuda()
				query_y = Variable(torch.from_numpy(query_y).long()).cuda()

 

				# get accuracy
				test_acc = meta.pred(support_x, support_y, query_x, query_y)
				test_accs.append(test_acc)

			test_acc = np.array(test_accs).mean()
			print('episode:', episode_num, '\tfinetune acc:%.6f' % train_acc, '\t\ttest acc:%.6f' % test_acc)
			tb.add_scalar('test-acc', test_acc)
			tb.add_scalar('finetune-acc', train_acc)


if __name__ == '__main__':
	main()
