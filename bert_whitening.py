import numpy as np
import torch
import os.path as osp
from iabg.utils import read_npy
from iabg.config import data_dir
conf = dict({'dataset' : 'clothing'})

def compute_kernel_bias(vecs, n_components):
    """计算kernel和bias
    最后的变换：y = (x + bias).dot(kernel)
    """
    # vecs = np.concatenate(vecs, axis=0)
    mu = vecs.mean(axis=0, keepdims=True)
    cov = np.cov(vecs.T)
    u, s, vh = np.linalg.svd(cov)
    W = np.dot(u, np.diag(s**0.5))
    W = np.linalg.inv(W.T)
    W = W[:, :n_components]
    return W, -mu

def transform_and_normalize(vecs, kernel, bias):
    """应用变换，然后标准化
    """
    if not (kernel is None or bias is None):
        vecs = (vecs + bias).dot(kernel)
    return vecs / (vecs**2).sum(axis=1, keepdims=True)**0.5

if __name__ == '__main__':
    new_item_id_nlp_embedding_path = osp.join(data_dir, conf['dataset'],'processed','item_nlp_info_new_ID_sorted_embeddings.npy')
    bundle_nlp_embedding_path = osp.join(data_dir, conf['dataset'],'processed','bundle_intent_nlp.npy')
    new_item_id_nlp_embedding = read_npy(new_item_id_nlp_embedding_path)
    bundle_nlp_embedding = read_npy(bundle_nlp_embedding_path)

    # import ipdb
    # ipdb.set_trace()

    # 计算kernel和bias
    kernel1, bias1 = compute_kernel_bias(new_item_id_nlp_embedding, 320)
    kernel2, bias2 = compute_kernel_bias(bundle_nlp_embedding, 320)

    # 应用变换，然后标准化
    new_item_id_nlp_embedding = transform_and_normalize(new_item_id_nlp_embedding, kernel1, bias1)
    bundle_nlp_embedding = transform_and_normalize(bundle_nlp_embedding, kernel2, bias2)

    # numpy 改dtype=float32
    new_item_id_nlp_embedding = new_item_id_nlp_embedding.astype(np.float32)
    bundle_nlp_embedding = bundle_nlp_embedding.astype(np.float32)

    # 保存
    np.save(osp.join(data_dir, conf['dataset'],'processed','item_nlp_info_new_ID_sorted_embeddings_whitening_320.npy'), new_item_id_nlp_embedding)
    np.save(osp.join(data_dir, conf['dataset'],'processed','bundle_intent_nlp_whitening_320.npy'), bundle_nlp_embedding)