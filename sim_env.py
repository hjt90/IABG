import os
import os.path as osp
import time
import math
import itertools
import random
import json

import numpy as np
import pandas as pd
import torch

import gym
from gym import spaces, logger
from gym.utils import seeding

from iabg.config import data_dir, model_dir, DATA_CONFIG, MODEL_CONFIG, DEFAULT_CONFIG
from iabg.models.baseline_bpr import ItemBPRModel, BundleBPRModel
from iabg.models.baseline_ncf import ItemNCFModel, BundleNCFModel
from iabg.models.item2vec import CBOWModel, SkipGramModel
from iabg.data_utils import setup_dataset_test
from iabg.utils import read_csv, write_csv, read_pickle, write_pickle, read_npy, read_json
from iabg.metrics import bundle_metrics


def load_data(conf):

    pkl_file = osp.join(data_dir, conf['dataset'], conf['user_vocab'])
    print("load user vocab:", pkl_file)
    user_vocab = read_pickle(pkl_file)
    pkl_file = osp.join(data_dir, conf['dataset'], conf['item_vocab'])
    print("load item vocab:", pkl_file)
    item_vocab = read_pickle(pkl_file)

    user_set = set([user_vocab[tok] for tok in user_vocab.freqs])
    item_set = set([item_vocab[tok] for tok in item_vocab.freqs])

    # print(len(user_set), min(user_set), max(user_set))
    # print(len(item_set), min(item_set), max(item_set))

    csv_file = osp.join(data_dir, conf['dataset'], conf['seq_file'])
    print("load train data:", csv_file)
    data = read_csv(csv_file)
    # print(len(data), data[0])
    train_ds = dict()
    for user, _, click, buy in data:
        if len(buy) == 0:
            continue
        user = int(user)
        click = list(map(int, click.split('|')))
        buy = list(map(int, buy.split('|')))
        if len(buy) < conf['bundle_size']:
            continue
        train_ds[user] = (click, buy)
    # print(len(train_ds), min(train_ds), max(train_ds))

    csv_file = osp.join(data_dir, conf['dataset'], 'test_bundle_%d_%d.csv' % (conf['pool_size'], conf['bundle_size']))
    print("load test data:", csv_file)
    data = read_csv(csv_file)
    # print(len(data), data[0])
    test_ds = dict()
    for user, pos, pool, seq in data:
        user = int(user)
        pos = list(map(int, pos.split('|')))
        assert len(pos) == conf['bundle_size']
        pool = list(map(int, pool.split('|')))
        seq = list(map(int, seq.split('|')))
        test_ds[user] = (seq, pool, pos)
    # print(len(test_ds), min(test_ds), max(test_ds))

    return train_ds, test_ds, user_set, item_set

def load_data_amazon(conf):

    pkl_file = osp.join(data_dir, conf['dataset'], conf['user_vocab'])
    print("load user vocab:", pkl_file)
    user_vocab = read_pickle(pkl_file)
    pkl_file = osp.join(data_dir, conf['dataset'], conf['item_vocab'])
    print("load item vocab:", pkl_file)
    item_vocab = read_pickle(pkl_file)
    json_file = osp.join(data_dir, conf['dataset'], conf['candidate_items'])
    print("load candidate items:", json_file)
    with open(json_file, 'r') as f:
        candidate_items = json.load(f)
    user_set = set([user_vocab[tok] for tok in user_vocab.freqs])
    item_set = set([item_vocab[tok] for tok in item_vocab.freqs])
    candidate_set = set([item_vocab[tok] for tok in candidate_items])
    candidate_set.remove(None)

    # print(len(user_set), min(user_set), max(user_set))
    # print(len(item_set), min(item_set), max(item_set))

    csv_file = osp.join(data_dir, conf['dataset'], conf['seq_file'])
    print("load train data:", csv_file)
    data = read_csv(csv_file)
    # print(len(data), data[0])
    train_ds = dict()
    for user,bid,bitems,seq in data:
        if len(bitems) == 0:
            continue
        user = int(user)
        bitems = list(map(int, bitems.split('|')))
        #debug here
        # import ipdb
        # ipdb.set_trace()
        if(seq == ''):
            continue
        seq = list(map(int, seq.split('|')))
        if len(bitems) <= 2:
            continue
        train_ds[user] = (seq, bitems, bid)

    csv_file = osp.join(data_dir, conf['dataset'], conf['test_file'])
    print("load test data:", csv_file)
    data = read_csv(csv_file)
    # print(len(data), data[0])
    test_ds = dict()
    for user, bid, bitems, seq, pool in data:
        user = int(user)
        bitems = list(map(int, bitems.split('|')))
        # if(len(bitems) < conf['bundle_size']):
        #     continue
        # elif(len(bitems) > conf['bundle_size']):
        #     bitems = bitems[:conf['bundle_size']]
        # assert len(bitems) == conf['bundle_size']
        pool = list(map(int, pool.split('|')))
        if(seq == ''):
          seq = []
        else:
          seq = list(map(int, seq.split('|')))
        test_ds[user] = (seq, pool, bitems, bid)
    # print(len(test_ds), min(test_ds), max(test_ds))
    a = 1
    return train_ds, test_ds, user_set, item_set, candidate_set


def load_model(conf):
    conf.update(MODEL_CONFIG[conf['item_model']])
    if conf['item_model'] == 'BPR':
        item_model = ItemBPRModel(conf).to(conf['device'])
    else:
        item_model = ItemNCFModel(conf).to(conf['device'])
    file_name = '%s-%s-10.pt' % (conf['dataset'], type(item_model).__name__)
    path = osp.join(model_dir, file_name)
    print("load item model:", path)
    item_model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    item_model.eval()

    conf.update(MODEL_CONFIG[conf['bundle_model']])
    if conf['bundle_model'] == 'BPR':
        bundle_model = BundleBPRModel(conf).to(conf['device'])
    else:
        bundle_model = BundleNCFModel(conf).to(conf['device'])
    file_name = '%s-%s-10.pt' % (conf['dataset'], type(bundle_model).__name__)
    path = osp.join(model_dir, file_name)
    print("load bundle model:", path)
    bundle_model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    bundle_model.eval()

    conf.update(MODEL_CONFIG[conf['compat_model']])
    compat_model = SkipGramModel(conf).to(conf['device'])
    file_name = '%s-%s-10.pt' % (conf['dataset'], type(compat_model).__name__)
    path = osp.join(model_dir, file_name)
    print("load compat model:", path)
    compat_model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    compat_model.eval()

    return item_model, compat_model, bundle_model

def load_model_amazon(conf):
    # new_id： 长度11764 有NLP标注
    # item_id： 长度38912
    # 可以靠十位码转换
    # 期望输出new_id下的embeddings,item_embeddings和item_nlp_embeddings分别输出
    # 返回的user_embedding和item_embedding要进行排序
    # user也要map
    conf.update(MODEL_CONFIG[conf['item_model']])
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


    # file_name = '%s-%s-10.pt' % (conf['dataset'], type(item_model).__name__)
    # path = osp.join(model_dir, file_name)
    # print("load item model:", path)
    # item_model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    # item_model.eval()

    # conf.update(MODEL_CONFIG[conf['bundle_model']])
    # if conf['bundle_model'] == 'BPR':
    #     bundle_model = BundleBPRModel(conf).to(conf['device'])
    # else:
    #     bundle_model = BundleNCFModel(conf).to(conf['device'])
    # file_name = '%s-%s-10.pt' % (conf['dataset'], type(bundle_model).__name__)
    # path = osp.join(model_dir, file_name)
    # print("load bundle model:", path)
    # bundle_model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    # bundle_model.eval()

    conf.update(MODEL_CONFIG[conf['compat_model']])
    compat_model = SkipGramModel(conf).to(conf['device'])
    file_name = '%s-%s-10.pt' % (conf['dataset'], type(compat_model).__name__)
    path = osp.join(model_dir, file_name)
    print("load compat model:", path)
    compat_model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    compat_model.eval()

    return new_user_embedding, new_item_embedding, compat_model, new_item_id_nlp_embedding, bundle_nlp_embedding

class BundleEnv(gym.Env):
    """
    Description:
        Bundle Composition Environment.

    Observation:
        Type: Box()
        Num	Observation    Min  Max
        u	User Features  -Inf  Inf
        G	Candidate Items  -Inf  Inf
        ...

    Actions:
        Type: Discrete(N)
        Num	Action
        0	Choose Item 0
        1	Choose Item 1
        ...

    Reward:
        Reward is evaluate at each step.

    Starting State:
        The bundle is empty.

    Episode Termination:
        The whole bundle are formed.
    """

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50,
    }

    # pool_size = 20  # number of candidate items
    # bundle_size = 10  # number of items per bundle
    # num_features = 1  # number of user/item features

    def __init__(self, conf):

        self.num_users = conf['n_user']  # number of all users
        self.num_items = conf['n_item']  # number of all items
        self.start_idx = conf['start_idx']  # start index of users/items
        self.seq_len = conf['seq_len']  # length of behaviour sequence
        self.pool_size = conf['pool_size']
        self.bundle_size = conf['bundle_size']
        self.num_features = conf['num_features']
        self.env_mode = conf.get('env_mode', 'train')  # train/test
        self.rew_mode = conf.get('rew_mode', 'item')  # item/compat/bundle/metric
        self.metric = conf.get('metric', 'recall')  # precision/precision_plus/recall
        self.dataset = conf['dataset']  # dataset name
        if self.dataset == 'movielens':
            self.train_ds, self.test_ds, self.user_set, self.item_set = load_data(conf)
            self.item_model, self.compat_model, self.bundle_model = load_model(conf)
        else:
            self.train_ds, self.test_ds, self.user_set, self.item_set, self.candidate_set = load_data_amazon(conf)
            self.user_embedding, self.item_embedding, self.compat_model, _, _ = load_model_amazon(conf)
            # self.item_embedding = np.concatenate((self.item_embedding, item_nlp_embedding),axis=1)
        # assert self.num_users == self.start_idx + len(self.train_ds)
        

        # feature array for item pool (pool_size, num_features)
        self.x = None
        # whether sort items by feature or not to ensure consistent input order
        self.sort = True

        high = 999999
        self.action_space = spaces.Discrete(self.pool_size)
        # self.observation_space = spaces.Box(-1.0, 1.0, shape=(2 * self.pool_size * self.num_features,))
        self.observation_space = spaces.Dict({
            "user": spaces.Box(-high, high, shape=(1,), dtype=np.int32),
            "seq": spaces.Box(-high, high, shape=(self.seq_len,), dtype=np.int32),
            "pool": spaces.Box(-high, high, shape=(self.pool_size,), dtype=np.int32),
            "pos": spaces.Box(-high, high, shape=(self.bundle_size,), dtype=np.int32),
            "bid": spaces.Box(-high, high, shape=(1,), dtype=np.int32),
            "blen": spaces.Box(-high, high, shape=(1,), dtype=np.int32),
            "mask": spaces.Box(0, 1, shape=(self.pool_size,), dtype=np.int32),
            "bundle": spaces.Box(-high, high, shape=(self.bundle_size,), dtype=np.int32),
            "state": spaces.Box(-high, high, shape=(self.pool_size + self.bundle_size,), dtype=np.int32),
        })
        self.reward_range = (-float('inf'), float('inf'))

        self.state = None
        self.seed()

        self.max_episode_steps = self.pool_size
        self.elapsed_steps = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _reward(self, state, action):
        assert self.elapsed_steps <= self.bundle_size
        user, seq, pool, pos, bid,blen, mask, bundle = state
        # print(state)
        # seq = seq[np.newaxis, :]  # (1, L)
        num_selected = np.count_nonzero(bundle)
        # item preference reward
        # ------------------------------------------------------------------------
        r1 = 0.0
        if 'item' in self.rew_mode:
            item = np.array([pool[action]])  # (1,)
            if self.dataset == 'movielens':
                r1 = self.item_model.predict((user[np.newaxis], item[np.newaxis], seq[np.newaxis]))
            else:
                # import ipdb
                # ipdb.set_trace()
                u_e = self.user_embedding[user[np.newaxis]]
                i_e = self.item_embedding[item[np.newaxis]]
                r1 = np.dot(u_e[0],(i_e[0]).T)
            # print(r1.shape, r1, np.ravel(r1))
            r1 = np.ravel(r1)[0]
        # item compatibility reward
        # ------------------------------------------------------------------------
        r2 = 0.0
        if 'compat' in self.rew_mode:
            if num_selected > 0:
                inp = np.array([pool[action]] * num_selected).reshape(-1, 1)
                out = bundle[bundle > 0].reshape(-1, 1)
                assert inp.shape == out.shape
                r2 = self.compat_model.predict((inp, out))
                # print(r2.shape, r2, np.ravel(r2))
                r2 = np.ravel(r2.sum())[0]
        # bundle preference reward
        # ------------------------------------------------------------------------
        r3 = 0.0
        if 'bundle' in self.rew_mode:
            if num_selected >= blen - 1:
                pred = np.array(bundle)  # (K,)
                pred[-1] = pool[action]
                r3 = self.bundle_model.predict((user[np.newaxis], pred[np.newaxis], seq[np.newaxis]))
                # print(r3.shape, r3, np.ravel(r3))
                r3 = np.ravel(r3)[0]
        # ------------------------------------------------------------------------
        # metric reward
        # ------------------------------------------------------------------------
        r4 = 0.0
        recall = 0.0
        precision = 0.0
        precision_plus = 0.0
        if 'metric' in self.rew_mode:
            if num_selected >= blen - 1:
                pred = np.array(bundle)  # (K,)
                pred[-1] = pool[action]  # (K,)
                r4 = bundle_metrics(pos[np.newaxis], pred[np.newaxis])
                # print(pos, pred, r4)
                # r4 = r4['precision'] + r4['precision_plus'] + r4['recall']
                recall = r4['recall']
                precision = r4['precision']
                precision_plus = r4['precision_plus']
                r4 = r4[self.metric]
        # ------------------------------------------------------------------------
          # metric reward
        # ------------------------------------------------------------------------
        # print(r1, r2, r3, r4, num_selected, pred, pool[action])

        return r1 + r2 + r3 + r4, recall, precision, precision_plus

    def step(self, action):
        assert self.elapsed_steps is not None, "Cannot call env.step() before calling reset()"
        assert action in self.action_space, "%r (%s) invalid" % (action, type(action))
        self.elapsed_steps += 1
        # reward function
        # ------------------------------------------------------------------------
        start_time = time.time()
        reward,recall,precision,precision_plus = self._reward(self.state, action)
        elapsed = time.time() - start_time
        # transition dynamics
        # ------------------------------------------------------------------------
        user, seq, pool, pos, bid, blen, mask, bundle = self.state
        # 1 <= elapsed_steps <= bundle_size
        bundle[self.elapsed_steps - 1] = pool[action]
        # 0 <= n_chosed <= bundle_size - 1
        # n_chosed = self.elapsed_steps % self.bundle_size
        # bundle[n_chosed - 1] = pool[action]
        pool[action] = 0
        mask[action] = 0
        self.state = (user, seq, pool, pos, bid, blen, mask, bundle)
        # terminal state
        # ------------------------------------------------------------------------
        # done = True if self.elapsed_steps >= self.max_episode_steps else False
        # done = True if self.elapsed_steps >= self.bundle_size else False
        done = True if self.elapsed_steps >= blen else False
        if done:
            # print("bundle composition: ", self.elapsed_steps, bundle, recall,precision,precision_plus)
            pass
        # return np.array(self.state), reward, done, {}
        # return np.array(np.concatenate(self.state)), reward, done, {}
        # return list(map(np.array, self.state)), reward, done, {}
        return {
            "user": np.array(user),
            "seq": np.array(seq),
            "pool": np.array(pool),
            "pos": np.array(pos),
            "bid": np.array(bid),
            "blen": np.array(blen),
            "mask": np.array(mask),
            "bundle": np.array(bundle),
            "state": np.concatenate((pool, bundle))
        }, reward, done, {"elapsed": elapsed,"recall":recall,"precision":precision,"precision_plus":precision_plus}

    def _train_state(self):
        # user identity
        # -----------------------------------------------------------------
        # user = np.random.choice(list(self.user_set), 1)
        # import ipdb
        # ipdb.set_trace()
        user = np.random.choice(list(self.train_ds.keys()), 1)
        bid = -1 # 在movielens中无用
        blen = self.bundle_size
        if(self.dataset == 'movielens'):
            click, buy = self.train_ds[user.item()]
        else:
            his, bitems, bid  = self.train_ds[user.item()]
        # historical behaviors
        # -----------------------------------------------------------------
        # seq = random.sample(click, k=seq_len)
        if(self.dataset == 'movielens'):  
            idx = np.random.randint(0, len(click) - self.seq_len + 1)
            seq = click[idx:idx + self.seq_len]
        else:
            if len(his) < self.seq_len:
                # 用户历史数据长度不够，从candidate_setz中sample补齐
                # sample self.seq_len-len(his) supplement from whole candidate items
                supplement = np.random.choice(list(self.candidate_set), self.seq_len-len(his))
                seq = np.concatenate((his, supplement))
            else:
                # 用户历史数据长度够，直接从用户历史数据中sample seq
                idx = np.random.randint(0, len(his) - self.seq_len + 1)
                seq = his[idx:idx + self.seq_len]
        # positive bundle
        # -----------------------------------------------------------------
        # pos = random.sample(buy, k=K)
        # pos = np.random.choice(buy, size=K, replace=False)
        if(self.dataset == 'movielens'):
            idx = np.random.randint(0, len(buy) - self.bundle_size + 1)
            pos = buy[idx:idx + self.bundle_size]
            random.shuffle(pos)
        else:
            if(len(bitems) < self.bundle_size):
                blen = len(bitems)
                # bundle内物品数量不足 增加padding
                padding = [0 for _ in range(self.bundle_size - len(bitems))]
                # supplement = np.random.choice(list(self.candidate_set), self.bundle_size-len(bitems))
                random.shuffle(bitems)
                pos = bitems+padding
            else:
                # bundle内物品数量足够 直接选择
                idx = np.random.randint(0, len(bitems) - self.bundle_size + 1)
                pos = bitems[idx:idx + self.bundle_size]
                random.shuffle(pos)
            # if(len(bitems)<self.bundle_size):
            #     pos = bitems
            # else:
            #     idx = np.random.randint(0, len(bitems) - self.bundle_size + 1)
            #     pos = bitems#[idx:idx + self.bundle_size]
            
        # candidate items
        # -----------------------------------------------------------------
        # pool = np.random.choice(list(self.item_set), self.pool_size, replace=False)
        if(self.dataset == 'movielens'):
            neg_set = self.item_set - set(buy)
            neg = random.sample(neg_set, k=self.pool_size - self.bundle_size)
            pool = pos + neg  # candidate items
        else:
            neg_set = self.candidate_set - set(pos[:blen])
            neg = random.sample(neg_set, k=self.pool_size - blen)
            pool = pos[:blen] + neg  # candidate items
            # if none in pool, then debug
            if None in pool:
                import ipdb
                ipdb.set_trace()
        random.shuffle(pool)
        return map(np.array, (user, seq, pool, pos, bid, blen))

    def _test_state(self):
        user = np.random.choice(list(self.test_ds.keys()), 1)
        bid = -1 # 在movielens中无用
        blen = self.bundle_size
        if(self.dataset == 'movielens'):
            seq, pool, pos = self.test_ds[user.item()]
        else:
            seq, pool, pos, bid = self.test_ds[user.item()]
            #如果bundle的长度不够，添加padding                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        ``````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````
            if(len(pos)<self.bundle_size):
                blen = len(pos)
                padding = [0 for _ in range(self.bundle_size - len(pos))]
                pos = pos+padding
            else:
                idx = np.random.randint(0, len(pos) - self.bundle_size + 1)
                pos = pos[idx:idx + self.bundle_size]
            #如果seq长度不够，从candidate_set中sample补齐
            if(len(seq)<self.seq_len):
                supplement = np.random.choice(list(self.candidate_set), self.seq_len-len(seq)).tolist()
                seq = seq+supplement
            else:
                seq = seq[:self.seq_len]
            
            #如果pool的长度不够，从candidate_set中sample补齐
            if(len(pool)<self.pool_size):
                supplement = np.random.choice(list(self.candidate_set), self.pool_size-len(pool)).tolist()
                pool = pool+supplement
            else:
                pool = pool[:self.pool_size]

        return map(np.array, (user, seq, pool, pos, bid,blen))

    def reset(self):
        # import ipdb
        # ipdb.set_trace()
        if self.env_mode == 'train':
            user, seq, pool, pos, bid, blen = self._train_state()
            # user, seq, pool, pos = self._test_state()  # over-fitting
        else:
            user, seq, pool, pos, bid, blen = self._test_state()
        mask = np.array([1] * self.pool_size, dtype=np.int32)
        # bundle = np.zeros(shape=(self.bundle_size,), dtype=np.int32)
        bundle = np.zeros(shape=(self.bundle_size,), dtype=np.int32)
        # (1,) (20,) (20,) (3,) (20,) (3,) all the data type is  <class 'numpy.ndarray'>
        # print(type(user), type(seq), type(pool), type(pos), type(mask), type(bundle))
        # print(user.shape, seq.shape, pool.shape, pos.shape, mask.shape, bundle.shape)
        self.state = (user, seq, pool, pos, bid, blen, mask, bundle)
        self.elapsed_steps = 0
        # return np.array(self.state)
        # return np.array(np.concatenate(self.state))
        # return list(map(np.array, self.state))
        return {
            "user": np.array(user),
            "seq": np.array(seq),
            "pool": np.array(pool),
            "pos": np.array(pos),
            "bid": np.array(bid),
            "blen": np.array(blen),
            "mask": np.array(mask),
            "bundle": np.array(bundle),
            "state": np.concatenate((pool, bundle))
        }

    def render(self, mode='human'):
        pass

    def close(self):
        pass


if __name__ == '__main__':

    conf = {
        'dataset': 'clothing',
        'pool_size': 50,
        'bundle_size': 10,
        'item_model': 'NCF',
        'bundle_model': 'NCF',
        'compat_model': 'SG',
        'num_features': 1,
        'device': 'cpu'
    }
    conf.update(DATA_CONFIG[conf['dataset']])
    conf['vocab_size'] = conf['n_item']
    conf['env_mode'] = 'test'  # train/test
    conf['rew_mode'] = 'item'  # item/compat/bundle/metric
    conf['metric'] = 'recall'  # precision/precision_plus/recall
    print(conf)

    env = BundleEnv(conf)
    state = env.reset()
    print("Initial State: \n", state)

    pool_size = env.pool_size
    actions = np.random.permutation(range(0, pool_size))
    # actions = list(itertools.permutations(range(1, pool_size + 1), 1))
    print("Test actions: \n", actions)
    print('-' * 80)

    for i in range(pool_size):
        env.render()
        # action = np.random.randint(low=1, high=8 + 1)  # this takes random actions
        action = actions[i]
        next_state, reward, done, info = env.step(action)
        print("Env step %d:" % (i + 1), action, reward, done, info)
        # print("Env step %d:" % (i + 1), state, action, next_state, reward, done)
        # print("Env step %d:" % (i + 1), state, action, next_state, reward, done, sep='\n', end='\n')
        print('-' * 120)
        state = next_state
        if done:
            state = env.reset()
            break
    env.close()
