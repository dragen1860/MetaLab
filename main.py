from omniglotNShot import OmniglotNShot
from meta import MetaLearner
from omninet import OmniNet

import torch
from torch.autograd import Variable
import numpy as np



def main():


	meta_batchsz = 32
	n_way = 5
	k_shot = 1
	k_query = 15
	meta_lr = 1e-3
	num_updates = 5


	meta = MetaLearner(OmniNet, n_way=n_way, k_shot=k_shot, meta_batchsz=meta_batchsz, beta=meta_lr, num_updates=num_updates).cuda()

	db = OmniglotNShot('dataset', batchsz=meta_batchsz, n_way=n_way, k_shot=k_shot, k_query=k_query, imgsz=28)

	for episode_num in range(20000):
		support_x, support_y, query_x, query_y = db.get_batch('train')
		support_x = Variable(torch.from_numpy(support_x).float().transpose(2,4).transpose(3, 4).repeat(1,1,3,1,1)).cuda()
		query_x = Variable(torch.from_numpy(query_x).float().transpose(2,4).transpose(3, 4).repeat(1,1,3,1,1)).cuda()
		support_y = Variable(torch.from_numpy(support_y).long()).cuda()
		query_y = Variable(torch.from_numpy(query_y).long()).cuda()

		# support_x: [meta_batchsz, setsz, 3, imgsz, imgsz]
		# query_x: [meta_batchsz, querysz, 3, imgsz, imgsz]
		accs = meta(support_x, support_y, query_x, query_y)
		acc = np.array(accs).mean()
		print(episode_num, acc)





















if __name__ == '__main__':
	main()