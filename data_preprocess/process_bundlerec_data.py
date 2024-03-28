# process the data from https://github.com/BundleRec/bundle_recommendation/tree/main/dataset
#

import os

os.environ["LOGURU_LEVEL"] = "DEBUG"

from collections import Counter
import pickle
import random

import pandas as pd

import numpy as np
from argparse import ArgumentParser
import random
from pathlib import Path
from tqdm.auto import tqdm
from loguru import logger
from typing import Tuple, Dict, List, Set
import json
from torchtext.vocab import Vocab
import torch as th

# cosine similarity
from sklearn.metrics.pairwise import cosine_similarity

tqdm.pandas()

PATH_DATA = Path("data")
PATH_PROCESSED = "processed"
PATH_PRETRAIN = "pretrain"
PATH_EMBEDDINGS = "embeddings"


def load_id_mappings(
    dataset="clothing",
) -> Tuple[dict, dict, dict, dict, dict, dict, dict, dict]:
    logger.info("loading id mappings on dataset: {}".format(dataset))
    user_idx_map = pd.read_csv(PATH_DATA / "user_idx_mapping.csv")

    user_id_source_id = dict(zip(user_idx_map["user ID"], user_idx_map["source ID"]))
    source_id_user_id = dict(zip(user_idx_map["source ID"], user_idx_map["user ID"]))

    with open(PATH_DATA / dataset / PATH_PRETRAIN / "pretrain_item_ID_map.json") as f:
        pretrain_source_id_item_id = json.load(f)

    with open(PATH_DATA / dataset / PATH_PRETRAIN / "pretrain_user_ID_map.json") as f:
        pretrain_source_id_user_id = json.load(f)

    pretrain_item_id_source_id = {v: k for k, v in pretrain_source_id_item_id.items()}
    pretrain_user_id_source_id = {v: k for k, v in pretrain_source_id_user_id.items()}

    with open(PATH_DATA / dataset / PATH_PROCESSED / "new_item_ID_map.json") as f:
        source_id_new_item_id = json.load(f)

    new_item_id_source_id = {v: k for k, v in source_id_new_item_id.items()}

    return (
        user_id_source_id,
        source_id_user_id,
        new_item_id_source_id,
        source_id_new_item_id,
        pretrain_user_id_source_id,
        pretrain_source_id_user_id,
        pretrain_item_id_source_id,
        pretrain_source_id_item_id,
    )


def load_data(
    dataset: str,
    user_id_source_id: dict,
    pretrain_source_id_user_id: dict,
    new_item_id_source_id: dict,
    pretrain_source_id_item_id: dict,
) -> Tuple[pd.DataFrame, pd.DataFrame, dict, Set[int]]:
    logger.info("loading data on dataset: {}".format(dataset))
    path_processed_data = PATH_DATA / dataset / PATH_PROCESSED
    path_pretrained_data = PATH_DATA / dataset / PATH_PRETRAIN
    user_bundle = pd.read_csv(path_processed_data / "user_bundle.csv")
    bundle_item = pd.read_csv(path_processed_data / "bundle_item_new_ID.csv")

    with open(path_processed_data / "candidate_items_lookup.json") as f:
        candidate_items = json.load(f)

    candidate_items = [
        pretrain_source_id_item_id[new_item_id_source_id[x]] for x in candidate_items
    ]

    with open(path_processed_data / "candidate_items_lookup_pretrain.json", "w") as f:
        json.dump(candidate_items, f)

    candidate_items = set(candidate_items)

    logger.info("getting user sequence from pretrain data")
    user_seq = {}
    with open(path_pretrained_data / "train.txt") as f:
        train = f.readlines()
        for line in tqdm(train):
            seq = line.strip().split(" ")
            user_seq[int(seq[0])] = seq[1:]

    with open(path_pretrained_data / "test.txt") as f:
        test = f.readlines()
        for line in tqdm(test):
            seq = line.strip().split(" ")
            user = int(seq[0])
            if user in user_seq:
                user_seq[user] += seq[1:]
            else:
                logger.warning("unknown user: {}".format(user))
                user_seq[user] = seq[1:]

    logger.info("converting ID to pretrain ID")
    logger.info(
        "filter out users that are not in pretrain data from user_bundle, before: {}".format(
            len(user_bundle)
        )
    )
    user_bundle = user_bundle[
        user_bundle["user ID"].progress_apply(
            lambda x: x in user_id_source_id
            and user_id_source_id[x] in pretrain_source_id_user_id
        )
    ]
    logger.info("after: {}".format(len(user_bundle)))

    user_bundle["user ID"] = user_bundle["user ID"].progress_apply(
        lambda x: pretrain_source_id_user_id[user_id_source_id[x]]
    )
    bundle_item["new item ID"] = bundle_item["new item ID"].progress_apply(
        lambda x: pretrain_source_id_item_id[new_item_id_source_id[x]]
    )
    bundle_item.rename(columns={"new item ID": "item ID"}, inplace=True)

    logger.info("save the bundle_item_pretrain_id.csv")
    bundle_item.to_csv(path_processed_data / "bundle_item_pretrain_ID.csv", index=False)

    return user_bundle, bundle_item, user_seq, candidate_items


def generate_seq(
    args: dict,
    user_bundle: pd.DataFrame,
    bundle_item: pd.DataFrame,
    user_seq: Dict[int, str],
) -> pd.DataFrame:
    logger.info("generating seq on dataset: {}".format(args["dataset"]))
    user_bundle_item = pd.merge(user_bundle, bundle_item, on="bundle ID")
    logger.info("merged user_bundle and bundle_item on: {}".format(args["dataset"]))
    logger.debug(user_bundle_item.head())
    user_bundle_item = (
        user_bundle_item.groupby(["user ID", "bundle ID"])["item ID"]
        .progress_apply(lambda x: list(x))
        .reset_index()
    )

    logger.info("convert bundle content to list on: {}".format(args["dataset"]))
    logger.debug(user_bundle_item.head())
    user_bundle_item["seq"] = user_bundle_item.progress_apply(
        lambda x: user_seq[x["user ID"]] if x["user ID"] in user_seq else [], axis=1
    )
    # user_bundle_item['appended item ID'] = user_bundle_item['new item ID'].progress_apply(lambda x: x + [0] * (100 - len(x)))

    return user_bundle_item


def user_iterator(user_bundle_items: pd.DataFrame):
    for _, row in user_bundle_items.iterrows():
        tokens = [int(row["user ID"])]
        yield tokens


def item_iterator(user_bundle_items: pd.DataFrame):
    for _, row in user_bundle_items.iterrows():
        tokens = row["item ID"] + row["seq"]
        tokens = list(map(int, tokens))
        yield tokens


def build_vocab(iterator, num_lines=None, min_freq=1):
    counter = Counter()
    with tqdm(unit_scale=0, unit="lines", total=num_lines) as t:
        for tokens in iterator:
            counter.update(tokens)
            t.update(1)
    vocab = Vocab(counter, min_freq=min_freq, specials=("<pad>",), specials_first=True)
    return vocab


def generate_vocab(args: dict, user_bundle_item: pd.DataFrame) -> Tuple[Vocab, Vocab]:
    logger.info("generating vocab on dataset: {}".format(args["dataset"]))
    logger.info("generating vocab for user")
    user_vocab = build_vocab(user_iterator(user_bundle_item), min_freq=1)
    logger.info("generating vocab for item")
    item_vocab = build_vocab(item_iterator(user_bundle_item), min_freq=1)
    user_set = set([user_vocab[tok] for tok in user_vocab.freqs])
    item_set = set([item_vocab[tok] for tok in item_vocab.freqs])

    pkl_file = PATH_DATA / args["dataset"] / PATH_PROCESSED / "vocab.user.pkl"
    with open(pkl_file, "wb") as f:
        pickle.dump(user_vocab, f)

    assert len(user_set) + 1 == len(user_vocab)

    pkl_file = PATH_DATA / args["dataset"] / PATH_PROCESSED / "vocab.item.pkl"
    with open(pkl_file, "wb") as f:
        pickle.dump(item_vocab, f)
    assert len(item_set) + 1 == len(item_vocab)

    return user_vocab, item_vocab


def apply_vocab_on_data(
    args: dict, user_bundle_item: pd.DataFrame, user_vocab: Vocab, item_vocab: Vocab
):
    logger.info("applying vocab on dataset: {}".format(args["dataset"]))
    user_bundle_item["user ID"] = user_bundle_item["user ID"].progress_apply(
        lambda x: user_vocab[x]
    )
    user_bundle_item["seq"] = user_bundle_item["seq"].progress_apply(
        lambda x: [item_vocab[int(y)] for y in x]
    )
    user_bundle_item["item ID"] = user_bundle_item["item ID"].progress_apply(
        lambda x: [item_vocab[y] for y in x]
    )

    return user_bundle_item


def generate_test(
    args: dict,
    user_bundle_item: pd.DataFrame,
    user_seq: Dict[int, str],
    candidate_items: List[int],
    user_vocab: Vocab,
    item_vocab: Vocab,
):
    logger.info("generating test on dataset: {}".format(args["dataset"]))

    logger.info(
        "1. generating test for each user (bundle_size >= {})".format(
            args["bundle_size"]
        )
    )
    user_bundle_test = user_bundle_item[
        user_bundle_item["item ID"].progress_apply(
            lambda x: len(x) >= args["bundle_size"]
        )
    ]

    logger.info("2. sample only 1 record for each user")
    user_bundle_test = user_bundle_test.groupby(
        "user ID", group_keys=False
    ).progress_apply(lambda df: df.sample(1))

    logger.info("3. generate neg samples")
    # sample from seq 0-(seq_size - pool_size + bundle_size), sample pool_size - bundle_size items, if seq is not long enough, sample from candidate items

    # iterrows....
    user_bundle_test["pool"] = pd.Series(dtype=object)
    for index, row in tqdm(user_bundle_test.iterrows()):
        seq = row["seq"]
        seq_size = len(seq)
        bundle_content = row["item ID"]
        pool_size = args["pool_size"]
        bundle_size = len(bundle_content)
        if seq_size < pool_size - bundle_size:
            # seq is not enough, sample from candidate items
            samples = random.sample(candidate_items, pool_size - bundle_size - seq_size)
            # apply item_vocab on samples from candidate items
            samples = [item_vocab[y] for y in samples]
            neg_items = seq + samples
        else:
            # seq is enough, randomly slice from seq
            start = random.randint(0, seq_size - pool_size + bundle_size)
            neg_items = seq[start : start + pool_size - bundle_size]

        pool_content = bundle_content + list(map(str, neg_items))
        assert len(pool_content) == args["pool_size"]
        user_bundle_test.at[index, "pool"] = pool_content

    # logger.info('length of neg_items_series: {}, user_bundle_test: {}'.format(len(neg_items_series), len(user_bundle_test)))

    # user_bundle_test['neg'] = user_bundle_test.progress_apply(
    #     lambda x: x['seq'][random.randint(0, len(x['seq']) - args['pool_size'] + args['bundle_size']) : : args['pool_size'] - args['bundle_size'] - 1]
    #         if len(x['item ID']) + len(x['seq']) > args['pool_size']
    #         else
    #             x['seq'] + random.sample(candidate_items - set(x['seq']) - set(x['item ID']), args['pool_size'] - len(x['item ID']) - len(x['seq'])), axis=1)
    return user_bundle_test


def output_to_file(
    args: dict, user_bundle_item: pd.DataFrame, user_bundle_test: pd.DataFrame
):
    logger.info("output to file")
    user_bundle_item["item ID"] = user_bundle_item["item ID"].progress_apply(
        lambda x: "|".join(list(map(str, x)))
    )
    user_bundle_item["seq"] = user_bundle_item["seq"].progress_apply(
        lambda x: "|".join(list(map(str, x)))
    )

    user_bundle_item.rename(
        columns={
            "item ID": "bitems",
            "user ID": "user",
            "item ID": "bitems",
            "bundle ID": "bid",
        },
        inplace=True,
    )
    user_bundle_item.to_csv(
        PATH_DATA / args["dataset"] / PATH_PROCESSED / "user_bundle_item.csv",
        index=False,
    )
    # user, bid, bitems, seq

    user_bundle_test["item ID"] = user_bundle_test["item ID"].progress_apply(
        lambda x: "|".join(list(map(str, x)))
    )
    user_bundle_test["seq"] = user_bundle_test["seq"].progress_apply(
        lambda x: "|".join(list(map(str, x)))
    )
    user_bundle_test["pool"] = user_bundle_test["pool"].progress_apply(
        lambda x: "|".join(list(map(str, x)))
    )

    # bundle content is positive items.
    user_bundle_test.rename(
        columns={"user ID": "user", "item ID": "bitems", "bundle ID": "bid"},
        inplace=True,
    )
    user_bundle_test.to_csv(
        PATH_DATA / args["dataset"] / PATH_PROCESSED / "user_bundle_test.csv",
        index=False,
    )
    # user, bid, bitems, pool, seq


def process_data_for_train_test(args: dict):
    logger.info("*********************************")
    logger.info("* 1. process data for train & test")
    logger.info("*********************************")
    (
        user_id_source_id,
        _,
        new_item_id_source_id,
        _,
        _,
        pretrain_source_id_user_id,
        _,
        pretrain_source_id_item_id,
    ) = load_id_mappings(args["dataset"])

    user_bundle, bundle_item, user_seq, candidate_items = load_data(
        args["dataset"],
        user_id_source_id,
        pretrain_source_id_user_id,
        new_item_id_source_id,
        pretrain_source_id_item_id,
    )
    user_bundle_item = generate_seq(args, user_bundle, bundle_item, user_seq)
    user_vocab, item_vocab = generate_vocab(args, user_bundle_item)
    user_bundle_item = apply_vocab_on_data(
        args, user_bundle_item, user_vocab, item_vocab
    )
    user_bundle_test = generate_test(
        args, user_bundle_item, user_seq, candidate_items, user_vocab, item_vocab
    )
    # user_bundle_test = apply_vocab_on_test_data(args, user_bundle_test, user_vocab, item_vocab)
    output_to_file(args, user_bundle_item, user_bundle_test)


def load_data_for_reward_model(
    args: dict,
) -> Tuple[pd.DataFrame, pd.DataFrame, np.array, List[int]]:
    dataset = args["dataset"]
    # load bundle and intention
    bundle_intent = pd.read_csv(
        PATH_DATA / dataset / PATH_PROCESSED / "bundle_intent.csv"
    )

    # load bundle and item, generated by process_bundlerec_data.py
    bundle_item = pd.read_csv(
        PATH_DATA / dataset / PATH_PROCESSED / "bundle_item_pretrain_ID.csv"
    )

    # item list in each bundle
    bundle_item = bundle_item.groupby("bundle ID")["item ID"].progress_apply(list).reset_index()
    bundle_item = bundle_item[bundle_item["item ID"].progress_apply(
            lambda x: len(x) >= args["bundle_size"]
        )]

    # load item embedding
    item_embedding = (
        th.load(
            PATH_DATA / dataset / PATH_PRETRAIN / PATH_EMBEDDINGS / "item_embd.pt",
            map_location=th.device("cpu"),
        )
        .detach()
        .numpy()
    )

    # load candidate items
    with open(
        PATH_DATA / dataset / PATH_PROCESSED / "candidate_items_lookup_pretrain.json"
    ) as f:
        candidate_items = json.load(f)

    return bundle_intent, bundle_item, item_embedding, candidate_items


def build_data_for_reward_model(
    args: dict,
    bundle_intent: pd.DataFrame,
    bundle_item: pd.DataFrame,
    item_embedding: np.array,
    candidate_items: List[int],
) -> pd.DataFrame:
# item ids are all pretrained id

    logger.info("getting cosine similarity matrix")
    logger.debug("item embeddings shape: {}".format(item_embedding.shape))
    # logger.debug(item_embedding[0])
    sim = cosine_similarity(item_embedding, item_embedding)
    logger.debug("similarity matrix shape: {}".format(sim.shape))
    # write the sim result to file
    np.save(PATH_DATA / args["dataset"] / PATH_PROCESSED / "similarity_matrix.npy", sim)

    # get intent - bundle - item mapping
    logger.info("getting intent - bundle - item mapping")

    bundle_intent_item = bundle_intent.merge(
        bundle_item, left_on="bundle ID", right_on="bundle ID"
    )

    # write the intent - bundle - item mapping to file
    logger.info("writing intent - bundle - item mapping to file")
    bundle_intent_item.to_csv(
        PATH_DATA / args["dataset"] / PATH_PROCESSED / "bundle_intent_item_pretrain_ID.csv",
        index=False,
    )

    # for every bundle, sample candidate with different similarity
    logger.info("sampling candidate items according to the similarity matrix")
    result = []
    # result = pd.DataFrame(columns=["intent", "items", "cand_item", "sim_avg", "score"])
    for _, row in tqdm(bundle_intent_item.iterrows()):
        bundle_id = row["bundle ID"]
        bundle_items = row["item ID"]
        intent = row["intent"]

        for i in range(len(bundle_items)):
            new_bundle_items = bundle_items.copy()
            # remove ith item from bundle
            new_bundle_items.pop(i)

            # 1. sample a item out from bundle items - score 5
            result.append({'intent': intent, 'items': new_bundle_items, 'cand_item': bundle_items[i], 'sim_avg': sim[bundle_items[i]][new_bundle_items].mean(), 'score': 5})

            _sim = sim[new_bundle_items].mean(axis=0).argsort()[::-1]
            _sim = [x for x in _sim if x not in new_bundle_items]
            _sim_topk = _sim[:args['sample_pool_size']]
            _sim_lowk = _sim[-args['sample_pool_size']:]
            # 2. sample a item out from candidate items with high similarity - score 2
            # sample how many? args['sample_size'], mask the items already in bundle

            _score_2_samples = random.sample(set(_sim_topk) - set(bundle_items), args['sample_size'])
            result.extend([{'intent': intent, 'items': new_bundle_items, 'cand_item': x, 'sim_avg': sim[x][new_bundle_items].mean(), 'score': 2} for x in _score_2_samples])

            # 3. sample a item out from candidate items with low similarity - score 1

            _score_1_samples = random.sample(set(_sim_lowk) - set(bundle_items), args['sample_size'])
            result.extend([{'intent': intent, 'items': new_bundle_items, 'cand_item': x, 'sim_avg': sim[x][new_bundle_items].mean(), 'score': 1} for x in _score_1_samples])

        # 4. sample any item out from candidate items bundle already full - score 0

        _score_0_samples = random.sample(set(candidate_items) - set(bundle_items), args['sample_size'])
        result.extend([{'intent': intent, 'items': new_bundle_items, 'cand_item': x, 'sim_avg': sim[x][new_bundle_items].mean(), 'score': 0} for x in _score_0_samples])

    result = pd.DataFrame(result)
    result.to_csv(PATH_DATA / args["dataset"] / PATH_PROCESSED / "reward_model_data_pretrain_ID_non_vocab.csv", index=False)

    return pd.DataFrame(result)

def apply_vocab_for_reward_model(args: dict, augmented_bundle_data: pd.DataFrame):

    pkl_file = PATH_DATA / args["dataset"] / PATH_PROCESSED / "vocab.item.pkl"
    with open(pkl_file, "rb") as f:
        item_vocab = pickle.load(f)

    logger.info("applying vocab for reward model")
    
    augmented_bundle_data['items'] = augmented_bundle_data['items'].progress_apply(lambda x: '|'.join([str(item_vocab[p]) for p in x]))
    augmented_bundle_data['cand_item'] = augmented_bundle_data['cand_item'].progress_apply(lambda x: item_vocab[x])

    augmented_bundle_data.to_csv(PATH_DATA / args["dataset"] / PATH_PROCESSED / "reward_model_data_pretrain_ID_vocab.csv", index=False)
    return augmented_bundle_data


def process_data_for_reward_model(args: dict):
    logger.info("*********************************")
    logger.info("* 2. process data for reward model")
    logger.info("*********************************")

    (
        bundle_intent,
        bundle_item,
        item_embedding,
        candidate_items,
    ) = load_data_for_reward_model(args)
    data = build_data_for_reward_model(
        args, bundle_intent, bundle_item, item_embedding, candidate_items
    )
    apply_vocab_for_reward_model(args, data)



def main():
    p = ArgumentParser()
    p.add_argument("--dataset", type=str, default="clothing")
    p.add_argument("--pool_size", type=int, default=20)
    p.add_argument("--bundle_size", type=int, default=3)
    p.add_argument("--seed", type=int, default=422)
    p.add_argument("--sample_size", type=int, default=5)
    p.add_argument("--sample_pool_size", type=int, default=20)
    args = vars(p.parse_args())
    random.seed(args["seed"])
    np.random.seed(seed=args["seed"])

    process_data_for_train_test(args)
    process_data_for_reward_model(args)


if __name__ == "__main__":
    main()
