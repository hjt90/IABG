import os.path as osp
from iabg.config import data_dir
from sentence_transformers import SentenceTransformer
from iabg.utils import read_json, read_npy, read_csv
import numpy as np

dataset = 'electronic'
# dataset = 'food'

new_item_id_nlp_path = osp.join(data_dir, dataset,'processed','item_nlp_info_new_ID.json')
bundle_nlp_path = osp.join(data_dir, dataset,'processed','bundle_intent.csv')
new_item_id_nlp_embedding_path = osp.join(data_dir, dataset,'processed','item_nlp_info_new_ID_sorted_embeddings.npy')
bundle_nlp_embedding_path = osp.join(data_dir, dataset,'processed','bundle_intent_nlp.npy')

new_item_id_nlp = read_json(new_item_id_nlp_path)
new_item_id_nlp_sort = sorted(new_item_id_nlp.items(), key=lambda x: int(x[0]))
new_item_id_nlp_sort = [str(i[1]) for i in new_item_id_nlp_sort]
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device='cuda:0')
# import ipdb
# ipdb.set_trace()
embeddings = model.encode(new_item_id_nlp_sort)
np.save(new_item_id_nlp_embedding_path, embeddings)

bundle_nlp = read_csv(bundle_nlp_path ,skip_header=True)
bundle_nlp = [i[1] for i in bundle_nlp]
embeddings = model.encode(bundle_nlp)
np.save(bundle_nlp_embedding_path, embeddings)

