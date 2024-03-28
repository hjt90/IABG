import os
import os.path as osp
import time
import numpy as np 
import pandas as pd 
import scipy.sparse as sp
import torch.utils.data as data
import random
from random import sample
import argparse
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.backends.cudnn as cudnn

from iabg.config import data_dir
from iabg.utils import read_json, read_npy, read_csv

from typing import (
    Union,
)

conf = dict({
	'dataset': 'clothing',
	'model_path': './save/trms/'   
})

def load_all_embeddings(conf):
    upath = osp.join(data_dir, conf['dataset'],'pretrain','embeddings','user_embd.pt')
    ipath = osp.join(data_dir, conf['dataset'],'pretrain','embeddings','item_embd.pt')
    new_item_id_map_path = osp.join(data_dir, conf['dataset'],'processed','new_item_ID_map.json')
    item_id_map_path = osp.join(data_dir, conf['dataset'],'pretrain','pretrain_item_ID_map.json')
    new_item_id_nlp_embedding_path = osp.join(data_dir, conf['dataset'],'processed','item_nlp_info_new_ID_sorted_embeddings.npy')
    bundle_nlp_embedding_path = osp.join(data_dir, conf['dataset'],'processed','bundle_intent_nlp.npy')
    user_id_map_path = osp.join(data_dir, conf['dataset'],'pretrain','pretrain_user_ID_map.json')
    new_user_id_map_path = osp.join(data_dir, conf['dataset'],'user_idx_mapping.csv')

    user_embedding = torch.load(upath, map_location=torch.device('cpu')).detach().numpy()
    item_embedding = torch.load(ipath, map_location=torch.device('cpu')).detach().numpy()
    new_item_id_map = read_json(new_item_id_map_path)
    item_id_map = read_json(item_id_map_path)
    new_item_id_nlp_embedding = read_npy(new_item_id_nlp_embedding_path)
    bundle_nlp_embedding = read_npy(bundle_nlp_embedding_path)
    user_id_map = read_json(user_id_map_path)
    new_user_id_map = read_csv(new_user_id_map_path,skip_header=True)

    # ItemID to 10code
    item_id2keys = list(item_id_map.keys())

    # ItemID to 10code to NewItemID
    item_id2new_item_id = [new_item_id_map[key] if key in new_item_id_map else -1 for key in item_id2keys]
    assert len([item for item in item_id2new_item_id if item != -1]) == len(new_item_id_map)

    # ItemIDEmbedding to NewItemIDEmbedding
    item_id2new_item_id_index = [item_id2new_item_id.index(i) for i in range(len(new_item_id_map))]
    new_item_embedding = item_embedding[item_id2new_item_id_index]

    # UserID to 14code
    user_id2keys = list(user_id_map.keys())

    # UserID to 14code to NewUserID
    keys_new_user_id = [int(item[0]) for item in new_user_id_map]
    values_14code = [item[1] for item in new_user_id_map]
    new_user_id_map = dict(zip(values_14code, keys_new_user_id))
    user_id2new_user_id = [new_user_id_map[key] if key in new_user_id_map else -1 for key in user_id2keys]
    assert len([item for item in user_id2new_user_id if item != -1]) == len(new_user_id_map)

    # UserIDEmbedding to NewUserIDEmbedding
    user_id2new_user_id_index = [user_id2new_user_id.index(i) for i in range(len(new_user_id_map))]
    new_user_embedding = user_embedding[user_id2new_user_id_index]

    return new_user_embedding, new_item_embedding, new_item_id_nlp_embedding, bundle_nlp_embedding

def ng_samples(user, bundle_id, item_list, max_item_id, accept_per = 0.5, is_test = False, sample_size = 99):
	'''
	Generate negative samples
	[user, bundle, item_list]: data from train_data
	max_item_id: max item id in train_data
	accept_per: the percentage of items that are finally in item_list
	sample_size: the number of negative samples for each positive sample

	if is_test=True return [[user, bundle_id, list(item_id), pos_item_id], sample_size * [user, bundle_id, list(item_id), neg_item_id]]
	else return [sample_size * [user, bundle_id, list(item_id), pos_item_id, neg_item_id]]
	'''
	# Determine user, bundle_id, accept_item, pos_item_id
	assert len(item_list) >= 1
	zero_accept = random.random() < 0.5
	accept_num = int(len(item_list) * accept_per) if not zero_accept else 0
	if accept_num == len(item_list):
		accept_num = len(item_list) - 1
	accept_item = sample(item_list, accept_num)
	unaccept_item = list(set(item_list) - set(accept_item))
	pos_item_id = np.random.choice(unaccept_item, 1)[0]
	accept_item = [accept_item[idx] if idx < len(accept_item) else 0 for idx in range(10)]


	# Generate negative samples
	if is_test:
		res = []
		pos_len = len(unaccept_item)
		neg_len = sample_size + 1 - pos_len
		for item in unaccept_item:
			res.append([user, bundle_id, accept_item, item, pos_len])
		for t in range(neg_len):
			j = np.random.randint(max_item_id)
			while j in item_list:
				j = np.random.randint(max_item_id)
			res.append([user, bundle_id, accept_item, j, pos_len])
	else:
		res = []
		for t in range(sample_size):
			j = np.random.randint(max_item_id)
			while j in item_list:
				j = np.random.randint(max_item_id)
			res.append([user, bundle_id, accept_item, pos_item_id, j])
	
	return res

def load_all_data():
	'''
	Load data from files
	train_data: list of [user, bundle, item]
	test_data: list of [user, bundle, item_i, item_i] (positive) and 99 * [user, bundle, item_i, item_i] (negative)
	'''
	user_bundle_path = osp.join(data_dir, conf['dataset'],'processed','user_bundle.csv')
	bundle_item_new_id_path = osp.join(data_dir, conf['dataset'],'processed','bundle_item_new_ID.csv')
	user_bundle = pd.read_csv(user_bundle_path)
	bundle_item_new_id = pd.read_csv(bundle_item_new_id_path)

	user_bundle = user_bundle.drop(columns=['timestamp'])
	bundle_item_new_id = bundle_item_new_id.groupby('bundle ID')['new item ID'].apply(list).reset_index()
	user_bundle_item_new_id = pd.merge(user_bundle, bundle_item_new_id, how='left', on='bundle ID')
	user_bundle_item_new_id = user_bundle_item_new_id.values.tolist()

	train_data, test_data = train_test_split(user_bundle_item_new_id, test_size=0.1, random_state=42)

	max_item_id = 0
	for user, bundle, item_list in user_bundle_item_new_id:
		for item in item_list:
			if item > max_item_id:
				max_item_id = item

	# test_data = sample(train_data, int(len(train_data) * 0.01))
	test_data_neg = []
	for user, bundle, item_list in test_data:
		test_data_neg += ng_samples(user, bundle, item_list, max_item_id, accept_per = random.random(), is_test = True, sample_size = 99)
	assert len(test_data_neg) == len(test_data) * 100
	
	return train_data, test_data_neg, max_item_id

def hit(gt_item, pred_items):
	res = 0
	for item in gt_item:
		if item in pred_items:
			res += 1
	return res

def ndcg(gt_item, pred_items):
	res = 0
	for item in gt_item:
		if item in pred_items:
			index = pred_items.index(item)
			res += np.reciprocal(np.log2(index+2))
	return res

def metrics(model, test_loader, top_k):
	HR, NDCG = [], []

	for user, bundle_id, accept_item, item_i, item_j, pos_len in test_loader:
		# import ipdb
		# ipdb.set_trace()
		pos_len = pos_len[0].item()
		prediction_i, prediction_j = model(user, bundle_id, accept_item, item_i, item_j)
		_, indices = torch.topk(prediction_i.cpu(), top_k)
		recommends = torch.take(
				item_i, indices).cpu().numpy().tolist()

		gt_item = [item.item() for item in item_i[0:pos_len]]
		HR.append(hit(gt_item, recommends)/pos_len)
		NDCG.append(ndcg(gt_item, recommends)/pos_len)

	return np.mean(HR), np.mean(NDCG)

class BPRData(data.Dataset):
	def __init__(self, features, 
				num_item, num_ng=0, is_training=None):
		super(BPRData, self).__init__()
		"""
		Note that the labels are only useful when training, we thus 
		add them in the ng_sample() function.
		"""
		self.features = features
		self.num_item = num_item
		self.num_ng = num_ng
		self.is_training = is_training

	def ng_sample(self):
		assert self.is_training, 'no need to sampling when testing'

		self.features_fill = []
		for user, bundle_id, item_list in self.features:
			self.features_fill += ng_samples(user, bundle_id, item_list, self.num_item, accept_per = random.random(), is_test = False, sample_size = self.num_ng)
		
	def __len__(self):
		return self.num_ng * len(self.features) if \
				self.is_training else len(self.features)

	def __getitem__(self, idx):
		features = self.features_fill if \
				self.is_training else self.features

		user, bundle_id, accept_item, item_i = features[idx][0], features[idx][1], features[idx][2], features[idx][3]
		
		item_j = features[idx][4] if \
				self.is_training else features[idx][3]
		
		if self.is_training:
			return user, bundle_id, accept_item, item_i, item_j
		else:
			return user, bundle_id, accept_item, item_i, item_j, features[idx][4]

class BPR(nn.Module):
	def __init__(
			self,
			device: Union[str, int, torch.device] = "cpu",
			onetrm: bool = False,
		  ):
		super(BPR, self).__init__()
		self.device = device
		self.onetrm = onetrm

		self.user_embedding, self.item_embedding, self.item_nlp_embedding, self.bundle_nlp_embedding = load_all_embeddings(conf)
		self.user_embedding = torch.tensor(self.user_embedding).to(device)
		self.item_embedding = torch.tensor(self.item_embedding).to(device)
		self.item_nlp_embedding = torch.tensor(self.item_nlp_embedding).to(device)
		self.bundle_nlp_embedding = torch.tensor(self.bundle_nlp_embedding).to(device)

		self.w1 = nn.Parameter(torch.Tensor([0.5]))
		self.encoder_layer = nn.TransformerEncoderLayer(d_model=320, nhead=8)
		self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=1)

		self.nlp_encoder_layer = nn.TransformerEncoderLayer(d_model=768, nhead=8)
		self.nlp_transformer_encoder = nn.TransformerEncoder(self.nlp_encoder_layer, num_layers=1)

		self.id_nlp_encoder_layer = nn.TransformerEncoderLayer(d_model=768+320, nhead=8)
		self.id_nlp_transformer_encoder = nn.TransformerEncoder(self.id_nlp_encoder_layer, num_layers=1)

	def forward(self, user, bundle_id, accept_item, item_i, item_j):
		# import ipdb
		# ipdb.set_trace()
		user = user.to(self.device)
		bundle = torch.cat(accept_item, dim=0).reshape(len(accept_item),-1).transpose(0,1).to(self.device)
		item_i = item_i.to(self.device)
		item_j = item_j.to(self.device)
		pool = torch.cat((item_i.unsqueeze(1), item_j.unsqueeze(1)), dim=1)	# pool_len = 2
		bid = bundle_id.to(self.device)

		user_embed = self.user_embedding[user].unsqueeze(1) # (B,1,E)
		pool_embed = self.item_embedding[pool] # (B,pool_len,E)
		bundle_seq_embed = self.item_embedding[bundle] # (B,bundle_len,E)
		pool_nlp_embed = self.item_nlp_embedding[pool] # (B,pool_len,NE)
		bundle_seq_nlp_embed = self.item_nlp_embedding[bundle] # (B,bundle_len,NE)
		bundle_nlp_embed = self.bundle_nlp_embedding[bid] # (B,NE)
		bundle_nlp_embed = torch.unsqueeze(bundle_nlp_embed, dim=1) # (B,1,NE)

		if self.onetrm:
			pool_id_nlp_embed = torch.cat([pool_embed,pool_nlp_embed],dim=2) # (B,pool_len,E+NE)
			bundle_mask = torch.unsqueeze(bundle==0, dim=2) # (B,bundle_len,1)
			bundle_mask = torch.cat([torch.tensor([False]).to(self.device).expand(bundle_mask.shape[0],1,1), bundle_mask], dim=1) # (B,bundle_len+1,1)
			user_id_embed = torch.cat([user_embed,bundle_seq_embed],dim=1) # (B,bundle_len+1,E)
			user_nlp_embed = torch.cat([bundle_nlp_embed,bundle_seq_nlp_embed],dim=1) # (B,bundle_len+1,NE)
			user_id_nlp_embed = torch.cat([user_id_embed,user_nlp_embed],dim=2) # (B,bundle_len+1,E+NE)

			user_id_nlp_embed = self.id_nlp_transformer_encoder(user_id_nlp_embed.transpose(0,1),src_key_padding_mask=bundle_mask.squeeze(dim=2)) # (bundle_len+1,B,E+NE)
			user_id_nlp_embed = user_id_nlp_embed.transpose(0,1).masked_fill(bundle_mask, value=0) # (B,bundle_len+1,E+NE) mask掉bundle中空的位置
			user_id_nlp_embed = user_id_nlp_embed.sum(dim=1)/(torch.logical_not(bundle_mask).sum(dim=1)) # (B,E)
			logits = torch.cosine_similarity(user_id_nlp_embed.unsqueeze(dim=1),pool_id_nlp_embed,dim=-1).squeeze(dim=1) # (B,pool_len)
		else:
			bundle_mask = torch.unsqueeze(bundle==0, dim=2) # (B,bundle_len,1)
			bundle_mask = torch.cat([torch.tensor([False]).to(self.device).expand(bundle_mask.shape[0],1,1), bundle_mask], dim=1) # (B,bundle_len+1,1)
			user_id_embed = torch.cat([user_embed,bundle_seq_embed],dim=1) # (B,bundle_len+1,E)
			user_nlp_embed = torch.cat([bundle_nlp_embed,bundle_seq_nlp_embed],dim=1) # (B,bundle_len+1,NE)

			user_id_embed = self.transformer_encoder(user_id_embed.transpose(0,1),src_key_padding_mask=bundle_mask.squeeze(dim=2)) # (bundle_len+1,B,E)
			user_nlp_embed = self.nlp_transformer_encoder(user_nlp_embed.transpose(0,1),src_key_padding_mask=bundle_mask.squeeze(dim=2)) # (bundle_len+1,B,NE)
			# import ipdb
			# ipdb.set_trace()
			user_id_embed = user_id_embed.transpose(0,1).masked_fill(bundle_mask, value=0) # (B,bundle_len+1,E) mask掉bundle中空的位置
			user_nlp_embed = user_nlp_embed.transpose(0,1).masked_fill(bundle_mask, value=0) # (B,bundle_len+1,NE)
			user_id_embed = user_id_embed.sum(dim=1)/(torch.logical_not(bundle_mask).sum(dim=1)) # (B,E)
			user_nlp_embed = user_nlp_embed.sum(dim=1)/(torch.logical_not(bundle_mask).sum(dim=1)) # (B,NE)
		
			logits1 = torch.cosine_similarity(user_id_embed.unsqueeze(dim=1),pool_embed,dim=-1).squeeze(dim=1) # (B,pool_len)
			logits2 = torch.cosine_similarity(user_nlp_embed.unsqueeze(dim=1),pool_nlp_embed,dim=-1).squeeze(dim=1) # (B,pool_len)
			logits = (logits1*self.w1 + logits2) / (1 + self.w1)
			# logits = logits1

		prediction_i, prediction_j = logits[:,0], logits[:,1]
		return prediction_i, prediction_j


parser = argparse.ArgumentParser()
parser.add_argument("--lr", 
	type=float, 
	default=5e-5, 
	help="learning rate")
parser.add_argument("--lamda", 
	type=float, 
	default=0.1, 
	help="model regularization rate")
parser.add_argument("--batch_size", 
	type=int, 
	default=256, 
	help="batch size for training")
parser.add_argument("--epochs", 
	type=int,
	default=100000,  
	help="training epoches")
parser.add_argument("--top_k", 
	type=int, 
	default=5, 
	help="compute metrics@top_k")
parser.add_argument("--factor_num", 
	type=int,
	default=32, 
	help="predictive factors numbers in the model")
parser.add_argument("--num_ng", 
	type=int,
	default=4, 
	help="sample negative items for training")
parser.add_argument("--test_num_ng", 
	type=int,
	default=99, 
	help="sample part of negative items for testing")
parser.add_argument("--out", 
	default=True,
	help="save model or not")
parser.add_argument("--gpu", 
	type=str,
	default="5",  
	help="gpu card ID")
parser.add_argument('--bundle_size', type=int, default=10, help='bundle size (default: 10)')
parser.add_argument("--pretrain_path", type=str, default=None)
parser.add_argument("--onetrm", default = False, action='store_true')


if __name__ == "__main__":
	args = parser.parse_args()
	print(args)

	# os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
	cudnn.benchmark = True
	device = torch.device("cuda:"+str(args.gpu) if args.gpu is not None and torch.cuda.is_available() else "cpu")

	############################## PREPARE DATASET ##########################
	train_data, test_data, item_num = load_all_data()

	# construct the train and test datasets
	train_dataset = BPRData(
			train_data, item_num, args.num_ng, True)
	test_dataset = BPRData(
			test_data, item_num, 0, False)
	train_loader = data.DataLoader(train_dataset,
			batch_size=args.batch_size, shuffle=True, num_workers=0)
	test_loader = data.DataLoader(test_dataset,
			batch_size=args.test_num_ng+1, shuffle=False, num_workers=0)

	########################### CREATE MODEL #################################
	model = BPR(device, onetrm=args.onetrm).to(device)

	if args.pretrain_path:
		model.load_state_dict(torch.load(args.pretrain_path, map_location=device), strict=False)
		print("Loaded net from: ", args.pretrain_path)

	optimizer = optim.SGD(
				model.parameters(), lr=args.lr, weight_decay=args.lamda)
	# writer = SummaryWriter() # for visualization

	########################### TRAINING #####################################
	count, best_hr, best_epoch = 0, 0, 0
	for epoch in range(args.epochs):
		model.train() 
		start_time = time.time()
		train_loader.dataset.ng_sample()

		for user, bundle_id, accept_item, item_i, item_j in train_loader:
			model.zero_grad()
			prediction_i, prediction_j = model(user, bundle_id, accept_item, item_i, item_j)
			loss = - (prediction_i - prediction_j).sigmoid().log().sum()
			loss.backward()
			optimizer.step()
			# writer.add_scalar('data/loss', loss.item(), count)
			count += 1

		model.eval()
		HR, NDCG = metrics(model, test_loader, args.top_k)

		elapsed_time = time.time() - start_time
		print("The time elapse of epoch {:03d}".format(epoch) + " is: " + 
				time.strftime("%H: %M: %S", time.gmtime(elapsed_time)))
		print("HR: {:.3f}\tNDCG: {:.3f}\tBest epoch {:03d}\tBest HR: {:.3f}".format(np.mean(HR), np.mean(NDCG), best_epoch, best_hr))

		if HR > best_hr:
			best_hr, best_ndcg, best_epoch = HR, NDCG, epoch
			if args.out:
				if not os.path.exists(conf['model_path']):
					os.mkdir(conf['model_path'])
				torch.save(model.state_dict(),'{}net_params_14.pth'.format(conf['model_path']))

	print("End. Best epoch {:03d}: HR = {:.3f}, \
		NDCG = {:.3f}".format(best_epoch, best_hr, best_ndcg))