import os.path as osp
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import sys

from tianshou.data import Batch, to_torch
from tianshou.utils.net.common import MLP
from iabg.config import data_dir, model_dir, DATA_CONFIG, MODEL_CONFIG, DEFAULT_CONFIG
from iabg.models.baseline_bpr import ItemBPRModel, BundleBPRModel
from iabg.models.baseline_ncf import ItemNCFModel, BundleNCFModel
from iabg.models.item2vec import CBOWModel, SkipGramModel
from iabg.models.transformers import TransformerEncoder, TransformerEncoderLayer
from iabg.utils import read_json, read_npy, read_csv
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
    no_type_check,
)
ModuleType = Type[nn.Module]


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
    if conf['whitening']:
        new_item_id_nlp_embedding_path = osp.join(data_dir, conf['dataset'],'processed','item_nlp_info_new_ID_sorted_embeddings_whitening_320.npy')
        bundle_nlp_embedding_path = osp.join(data_dir, conf['dataset'],'processed','bundle_intent_nlp_whitening_320.npy')
    else:
        new_item_id_nlp_embedding_path = osp.join(data_dir, conf['dataset'],'processed','item_nlp_info_new_ID_sorted_embeddings.npy')
        bundle_nlp_embedding_path = osp.join(data_dir, conf['dataset'],'processed','bundle_intent_nlp.npy')
    user_id_map_path = osp.join(data_dir, conf['dataset'],'pretrain','pretrain_user_ID_map.json')
    new_user_id_map_path = osp.join(data_dir, conf['dataset'],'user_idx_mapping.csv')

    query_embedding = ''
    if conf['query_trm']:
        query_embedding_path = osp.join(data_dir, conf['dataset'],'processed','query_embeddings_384.npy')
        query_embedding = read_npy(query_embedding_path)
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
    # file_name = '%s-%s-10.pt' % (conf['dataset'], type(compat_model).__name__)
    # path = osp.join(model_dir, file_name)
    # print("load compat model:", path)
    # compat_model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    # compat_model.eval()

    return new_user_embedding, new_item_embedding, compat_model, new_item_id_nlp_embedding, bundle_nlp_embedding, query_embedding


class StateEncoder(nn.Module):
    """Wrapper of MLP to support more specific DRL usage.

    For advanced usage (how to customize the network), please refer to
    :ref:`build_the_network`.

    :param state_shape: int or a sequence of int of the shape of state.
    :param action_shape: int or a sequence of int of the shape of action.
    :param hidden_sizes: shape of MLP passed in as a list.
    :param norm_layer: use which normalization before activation, e.g.,
        ``nn.LayerNorm`` and ``nn.BatchNorm1d``. Default to no normalization.
        You can also pass a list of normalization modules with the same length
        of hidden_sizes, to use different normalization module in different
        layers. Default to no normalization.
    :param activation: which activation to use after each layer, can be both
        the same activation for all layers if passed in nn.Module, or different
        activation for different Modules if passed in a list. Default to
        nn.ReLU.
    :param device: specify the device when the network actually runs. Default
        to "cpu".
    :param bool softmax: whether to apply a softmax layer over the last layer's
        output.
    :param bool concat: whether the input shape is concatenated by state_shape
        and action_shape. If it is True, ``action_shape`` is not the output
        shape, but affects the input shape only.
    :param int num_atoms: in order to expand to the net of distributional RL.
        Default to 1 (not use).
    :param bool dueling_param: whether to use dueling network to calculate Q
        values (for Dueling DQN). If you want to use dueling option, you should
        pass a tuple of two dict (first for Q and second for V) stating
        self-defined arguments as stated in
        class:`~tianshou.utils.net.common.MLP`. Default to None.
    :param linear_layer: use this module as linear layer. Default to nn.Linear.

    .. seealso::

        Please refer to :class:`~tianshou.utils.net.common.MLP` for more
        detailed explanation on the usage of activation, norm_layer, etc.

        You can also refer to :class:`~tianshou.utils.net.continuous.Actor`,
        :class:`~tianshou.utils.net.continuous.Critic`, etc, to see how it's
        suggested be used.
    """

    def __init__(
        self,
        conf: dict,
        state_shape: Union[int, Sequence[int]],
        action_shape: Union[int, Sequence[int]] = 0,
        hidden_sizes: Sequence[int] = (),
        norm_layer: Optional[ModuleType] = None,
        activation: Optional[ModuleType] = nn.ReLU,
        device: Union[str, int, torch.device] = "cpu",
        softmax: bool = False,
        concat: bool = False,
        num_atoms: int = 1,
        dueling_param: Optional[Tuple[Dict[str, Any], Dict[str, Any]]] = None,
        linear_layer: Type[nn.Linear] = nn.Linear,
        onetrm: bool = False,
    ) -> None:
        super().__init__()
        self.device = device
        self.softmax = softmax
        self.datast = conf['dataset']
        self.onetrm = onetrm
        if(conf['dataset']=='movielens'):
            self.item_model, self.compat_model, self.bundle_model = load_model(conf)
            self.user_embedding = self.item_model.user_embed_mlp
            self.item_embedding = self.item_model.item_embed_mlp
        else:
            self.user_embedding,self.item_embedding, self.compat_model, self.item_nlp_embedding, self.bundle_nlp_embedding, self.query_embedding = load_model_amazon(conf)
            self.user_embedding = torch.tensor(self.user_embedding).to(device)
            self.item_embedding = torch.tensor(self.item_embedding).to(device)
            self.item_nlp_embedding = torch.tensor(self.item_nlp_embedding).to(device)
            self.bundle_nlp_embedding = torch.tensor(self.bundle_nlp_embedding).to(device)
            if conf['query_trm']:
                self.query_embedding = torch.tensor(self.query_embedding).to(device)
            else:
                self.query_embedding = '' 

        self.num_atoms = num_atoms
        input_dim = int(np.prod(state_shape)) #poolsize*2
        action_dim = int(np.prod(action_shape)) * num_atoms
        if concat:
            input_dim += action_dim
        self.use_dueling = dueling_param is not None
        output_dim = action_dim if not self.use_dueling and not concat else 0
        self.model = MLP(
            input_dim, output_dim, hidden_sizes, norm_layer, activation, device,
            linear_layer
        )
        self.nlp_dim = 320 if conf['whitening'] else conf['nlp_dim']
        self.id_dim = 320
        self.pool_trm = conf['pool_trm']
        self.only_id = conf['only_id']
        self.only_nlp = conf['only_nlp']
        self.no_trm = conf['no_trm']
        self.query_trm = conf['query_trm']

        # twotrm
        self.w1 = nn.Parameter(torch.Tensor([0.3]))
        self.encoder_layer = TransformerEncoderLayer(d_model=320, nhead=2)
        self.transformer_encoder = TransformerEncoder(self.encoder_layer, num_layers=1)

        self.nlp_encoder_layer = TransformerEncoderLayer(d_model=self.nlp_dim, nhead=2)
        self.nlp_transformer_encoder = TransformerEncoder(self.nlp_encoder_layer, num_layers=1)

        self.pool_encoder_layer = TransformerEncoderLayer(d_model=320, nhead=2, dless=1e-5)
        self.pool_transformer_encoder = TransformerEncoder(self.pool_encoder_layer, num_layers=1)

        self.pool_nlp_encoder_layer = TransformerEncoderLayer(d_model=self.nlp_dim, nhead=2, dless=1e-5)
        self.pool_nlp_transformer_encoder = TransformerEncoder(self.pool_nlp_encoder_layer, num_layers=1)

        # onetrm
        self.MLP = MLP(
            self.nlp_dim, self.id_dim, hidden_sizes, norm_layer, activation, device,
            linear_layer
        )
        self.only_id_MLP = MLP(
            self.id_dim, self.id_dim, hidden_sizes, norm_layer, activation, device,
            linear_layer
        )
        self.id_nlp_encoder_layer = TransformerEncoderLayer(d_model=320, nhead=2)
        self.id_nlp_transformer_encoder = TransformerEncoder(self.id_nlp_encoder_layer, num_layers=1)
        
        self.pool_id_nlp_encoder_layer = TransformerEncoderLayer(d_model=320, nhead=2, dless=1e-5)
        self.pool_id_nlp_transformer_encoder = TransformerEncoder(self.pool_id_nlp_encoder_layer, num_layers=1)

        self.output_dim = self.model.output_dim
        if self.use_dueling:  # dueling DQN
            q_kwargs, v_kwargs = dueling_param  # type: ignore
            q_output_dim, v_output_dim = 0, 0
            if not concat:
                q_output_dim, v_output_dim = action_dim, num_atoms
            q_kwargs: Dict[str, Any] = {
                **q_kwargs, "input_dim": self.output_dim,
                "output_dim": q_output_dim,
                "device": self.device
            }
            v_kwargs: Dict[str, Any] = {
                **v_kwargs, "input_dim": self.output_dim,
                "output_dim": v_output_dim,
                "device": self.device
            }
            self.Q, self.V = MLP(**q_kwargs), MLP(**v_kwargs)
            self.output_dim = self.Q.output_dim

    def forward(
        self,
        obs: Union[np.ndarray, torch.Tensor, Batch],
        state: Any = None,
        info: Dict[str, Any] = {},
    ) -> Tuple[torch.Tensor, Any]:
        """Mapping: obs -> flatten (inside MLP)-> logits."""
        # logits = self.model(obs)
        user,seq,pool,pos,mask,bundle,state,bid = obs.user,obs.seq,obs.pool,obs.pos,obs.mask,obs.bundle,obs.state, obs.bid
        user = [i-1 for i in user] # user是从零开始的
        pool = [i-1 for i in pool]
        bundle = [i-1 for i in bundle]
        user = torch.tensor(user,dtype=torch.long ).to(self.device)
        seq = torch.tensor(seq.astype('long'),dtype=torch.long ).to(self.device)
        bundle = torch.tensor(bundle,dtype=torch.long ).to(self.device)
        mask = torch.tensor(mask.astype('long'),dtype=torch.long ).to(self.device)
        pool = torch.tensor(pool,dtype=torch.long ).to(self.device)
        bid = list(map(int, bid))
        #import ipdb
        #ipdb.set_trace()
        pid = [list(range(5*i,5*i+5)) for i in bid]
        pid = torch.tensor(pid,dtype=torch.long ).to(self.device)
        bid = torch.tensor(bid,dtype=torch.long ).to(self.device)

        if self.datast=='movielens':
            user_embed = self.user_embedding(user) # (B,E)
            item_embed = self.item_embedding(seq)  # (B,seq_len,E)
            pool_embed = self.item_embedding(pool) # (B,pool_len,E)
        else:
            user_embed = self.user_embedding[user] # (B,1,E)
            item_embed = self.item_embedding[seq]  # (B,seq_len,E)
            pool_embed = self.item_embedding[pool] # (B,pool_len,E)
            bundle_seq_embed = self.item_embedding[bundle] # (B,bundle_len,E)
            pool_nlp_embed = self.item_nlp_embedding[pool] # (B,pool_len,NE)
            item_nlp_embed = self.item_nlp_embedding[seq]  # (B,seq_len,NE)
            bundle_seq_nlp_embed = self.item_nlp_embedding[bundle] # (B,bundle_len,NE)
            bundle_nlp_embed = self.bundle_nlp_embedding[bid] # (B,NE)
            if self.query_trm:
                query_nlp_embed = self.query_embedding[pid] # (B,5,NE)
            bundle_nlp_embed = torch.unsqueeze(bundle_nlp_embed, dim=1) # (B,1,NE)

        if self.onetrm:
            bundle_mask = torch.unsqueeze(bundle==0, dim=2) # (B,bundle_len,1)
            bundle_nlp_mask = torch.cat([torch.tensor([False]).to(self.device).expand(bundle_mask.shape[0],1,1), bundle_mask], dim=1) # (B,bundle_len+1,1)
            user_id_embed = torch.cat([user_embed,bundle_seq_embed],dim=1) # (B,bundle_len+1,E)
            if self.only_id:
                user_id_embed = self.only_id_MLP(user_id_embed.reshape(-1,self.id_dim)).reshape(bundle_mask.shape[0],-1,self.id_dim)
            if self.query_trm:
                bundle_nlp_embed = torch.cat([bundle_nlp_embed,query_nlp_embed],dim=1) # (B,5+1,NE)
                bundle_nlp_mask = torch.cat([torch.tensor([False]).to(self.device).expand(bundle_mask.shape[0],6,1), bundle_mask], dim=1) # (B,bundle_len+6,1)
            bundle_mask = torch.cat([torch.tensor([False]).to(self.device).expand(bundle_mask.shape[0],1,1), bundle_mask], dim=1) # (B,bundle_len+1,1)
            user_nlp_embed = torch.cat([bundle_nlp_embed,bundle_seq_nlp_embed],dim=1) # (B,bundle_len+1,NE)
            user_nlp_embed = self.MLP(user_nlp_embed.reshape(-1,self.nlp_dim)).reshape(bundle_mask.shape[0],-1,self.id_dim) # (B,bundle_len+1,E)
            pool_nlp_embed = self.MLP(pool_nlp_embed.reshape(-1,self.nlp_dim)).reshape(bundle_mask.shape[0],-1,self.id_dim) # (B,pool_len,E)
            
            user_id_nlp_embed = torch.cat([user_id_embed,user_nlp_embed],dim=1) # (B,(bundle_len+1)*2,E)
            trm_mask = torch.cat([bundle_mask.squeeze(dim=2),bundle_nlp_mask.squeeze(dim=2)], dim=1) # (B,(bundle_len+1)*2)
            if not self.no_trm:
                user_id_nlp_embed = self.id_nlp_transformer_encoder(user_id_nlp_embed.transpose(0,1),src_key_padding_mask=trm_mask).transpose(0,1) # (B,(bundle_len+1)*2,E)
            user_id_embed = user_id_nlp_embed[:,:user_id_embed.shape[1],:] # (B,bundle_len+1,E)
            user_nlp_embed = user_id_nlp_embed[:,user_id_embed.shape[1]:,:] # (B,bundle_len+1,E)
            user_id_embed = user_id_embed.masked_fill(bundle_mask, value=0) # (B,bundle_len+1,E) mask掉bundle中空的位置
            user_nlp_embed = user_nlp_embed.masked_fill(bundle_nlp_mask, value=0) # (B,bundle_len+1,E)
            user_id_embed = user_id_embed.sum(dim=1)/(torch.logical_not(bundle_mask).sum(dim=1)) # (B,E)
            user_nlp_embed = user_nlp_embed.sum(dim=1)/(torch.logical_not(bundle_nlp_mask).sum(dim=1)) # (B,E)

            if self.pool_trm:
                pool_id_nlp_embed = torch.cat([pool_embed,pool_nlp_embed],dim=1) # (B,pool_len*2,E)
                pool_id_nlp_embed = self.pool_id_nlp_transformer_encoder(pool_id_nlp_embed.transpose(0,1)).transpose(0,1) # (B,pool_len*2,E)
                pool_embed = pool_id_nlp_embed[:,:pool_embed.shape[1],:] # (B,pool_len,E)
                pool_nlp_embed = pool_id_nlp_embed[:,pool_nlp_embed.shape[1]:,:] # (B,pool_len,NE)
                # pool_embed = self.pool_transformer_encoder(pool_embed.transpose(0,1)).transpose(0,1) # (B,pool_len,E)
                # pool_nlp_embed = self.pool_nlp_transformer_encoder(pool_nlp_embed.transpose(0,1)).transpose(0,1) # (B,pool_len,NE)

            logits1 = torch.cosine_similarity(user_id_embed.unsqueeze(dim=1),pool_embed,dim=-1).squeeze(dim=1) # (B,pool_len)
            logits2 = torch.cosine_similarity(user_nlp_embed.unsqueeze(dim=1),pool_nlp_embed,dim=-1).squeeze(dim=1) # (B,pool_len)
            if self.only_id:
                logits = logits1
            elif self.only_nlp:
                logits = logits2
            else:
                logits = (logits1*self.w1 + logits2) / (1 + self.w1)
        else:
            # import ipdb
            # ipdb.set_trace()
            bundle_mask = torch.unsqueeze(bundle==0, dim=2) # (B,bundle_len,1)
            bundle_mask = torch.cat([torch.tensor([False]).to(self.device).expand(bundle_mask.shape[0],1,1), bundle_mask], dim=1) # (B,bundle_len+1,1)
            user_id_embed = torch.cat([user_embed,bundle_seq_embed],dim=1) # (B,bundle_len+1,E)
            user_nlp_embed = torch.cat([bundle_nlp_embed,bundle_seq_nlp_embed],dim=1) # (B,bundle_len+1,NE)

            user_id_embed = self.transformer_encoder(user_id_embed.transpose(0,1),src_key_padding_mask=bundle_mask.squeeze(dim=2)) # (bundle_len+1,B,E)
            user_nlp_embed = self.nlp_transformer_encoder(user_nlp_embed.transpose(0,1),src_key_padding_mask=bundle_mask.squeeze(dim=2)) # (bundle_len+1,B,NE)
            user_id_embed = user_id_embed.transpose(0,1).masked_fill(bundle_mask, value=0) # (B,bundle_len+1,E) mask掉bundle中空的位置
            user_nlp_embed = user_nlp_embed.transpose(0,1).masked_fill(bundle_mask, value=0) # (B,bundle_len+1,NE)
            user_id_embed = user_id_embed.sum(dim=1)/(torch.logical_not(bundle_mask).sum(dim=1)) # (B,E)
            user_nlp_embed = user_nlp_embed.sum(dim=1)/(torch.logical_not(bundle_mask).sum(dim=1)) # (B,NE)
            
            if self.pool_trm:
                pool_embed = self.pool_transformer_encoder(pool_embed.transpose(0,1)).transpose(0,1) # (B,pool_len,E)
                pool_nlp_embed = self.pool_nlp_transformer_encoder(pool_nlp_embed.transpose(0,1)).transpose(0,1) # (B,pool_len,NE)
        
            # logits1 = torch.bmm(user_id_embed.unsqueeze(dim=1),pool_embed.transpose(1,2)).squeeze(dim=1) # (B,pool_len)
            logits1 = torch.cosine_similarity(user_id_embed.unsqueeze(dim=1),pool_embed,dim=-1).squeeze(dim=1) # (B,pool_len)
            logits2 = torch.cosine_similarity(user_nlp_embed.unsqueeze(dim=1),pool_nlp_embed,dim=-1).squeeze(dim=1) # (B,pool_len)
            logits = (logits1*self.w1 + logits2) / (1 + self.w1)

        logits = logits.masked_fill(mask==0, value=-1e10) # mask掉已选的item
        bsz = logits.shape[0]

        if self.use_dueling:  # Dueling DQN
            q, v = self.Q(logits), self.V(logits)           # QV size
            if self.num_atoms > 1:
                q = q.view(bsz, -1, self.num_atoms)
                v = v.view(bsz, -1, self.num_atoms)
            logits = q - q.mean(dim=1, keepdim=True) + v
        elif self.num_atoms > 1:
            logits = logits.view(bsz, -1, self.num_atoms)
        if self.softmax:
            logits = torch.softmax(logits, dim=-1) #（B,pool_len）
        #得到的logits作为state编码
        return logits, state


class Actor(nn.Module):
    """Simple actor network.

    Will create an actor operated in discrete action space with structure of
    preprocess_net ---> action_shape.

    :param preprocess_net: a self-defined preprocess_net which output a
        flattened hidden state.
    :param action_shape: a sequence of int for the shape of action.
    :param hidden_sizes: a sequence of int for constructing the MLP after
        preprocess_net. Default to empty sequence (where the MLP now contains
        only a single linear layer).
    :param bool softmax_output: whether to apply a softmax layer over the last
        layer's output.
    :param int preprocess_net_output_dim: the output dimension of
        preprocess_net.

    For advanced usage (how to customize the network), please refer to
    :ref:`build_the_network`.

    .. seealso::

        Please refer to :class:`~tianshou.utils.net.common.Net` as an instance
        of how preprocess_net is suggested to be defined.
    """

    def __init__(
        self,
        preprocess_net: nn.Module,
        action_shape: Sequence[int],
        hidden_sizes: Sequence[int] = (),
        softmax_output: bool = True,
        preprocess_net_output_dim: Optional[int] = None,
        device: Union[str, int, torch.device] = "cpu",
    ) -> None:
        super().__init__()
        self.device = device
        self.preprocess = preprocess_net
        self.output_dim = int(np.prod(action_shape))
        input_dim = getattr(preprocess_net, "output_dim", preprocess_net_output_dim)
        self.last = MLP(
            input_dim,  # type: ignore
            self.output_dim,
            hidden_sizes,
            device=self.device
        )
        self.softmax_output = softmax_output

    def forward(
        self,
        obs: Union[np.ndarray, torch.Tensor],
        state: Any = None,
        info: Dict[str, Any] = {},
    ) -> Tuple[torch.Tensor, Any]:
        r"""Mapping: s -> Q(s, \*)."""
        action_mask = obs.mask # (B,pool_len)
        # inf_mask = torch.where(torch.tensor(action_mask).to(self.device)==0,torch.tensor(float("-inf")).to(self.device),torch.tensor(0.).to(self.device))
        logits, hidden = self.preprocess(obs, state)
        logits = self.last(logits)
        logits = logits * torch.tensor(action_mask).to(self.device)
        if self.softmax_output:
            logits = F.softmax(logits, dim=-1)
        return logits, hidden


class Critic(nn.Module):
    """Simple critic network. Will create an actor operated in discrete \
    action space with structure of preprocess_net ---> 1(q value).

    :param preprocess_net: a self-defined preprocess_net which output a
        flattened hidden state.
    :param hidden_sizes: a sequence of int for constructing the MLP after
        preprocess_net. Default to empty sequence (where the MLP now contains
        only a single linear layer).
    :param int last_size: the output dimension of Critic network. Default to 1.
    :param int preprocess_net_output_dim: the output dimension of
        preprocess_net.

    For advanced usage (how to customize the network), please refer to
    :ref:`build_the_network`.

    .. seealso::

        Please refer to :class:`~tianshou.utils.net.common.Net` as an instance
        of how preprocess_net is suggested to be defined.
    """

    def __init__(
        self,
        preprocess_net: nn.Module,
        hidden_sizes: Sequence[int] = (),
        last_size: int = 1,
        preprocess_net_output_dim: Optional[int] = None,
        device: Union[str, int, torch.device] = "cpu",
    ) -> None:
        super().__init__()
        self.device = device
        self.preprocess = preprocess_net
        self.output_dim = last_size
        input_dim = getattr(preprocess_net, "output_dim", preprocess_net_output_dim)
        self.last = MLP(
            input_dim,  # type: ignore
            last_size,
            hidden_sizes,
            device=self.device
        )

    def forward(
        self, obs: Union[np.ndarray, torch.Tensor], **kwargs: Any
    ) -> torch.Tensor:
        """Mapping: s -> V(s)."""
        logits, _ = self.preprocess(obs, state=kwargs.get("state", None))
        return self.last(logits)

