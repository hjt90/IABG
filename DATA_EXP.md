# Data explaination
Intention Aware Bundle Generation

```
data
├── clothing
|   ├── pretrain 预训练推荐模型相关文件存放目录
|   │   ├── embeddings 使用推荐模型预训练得到的user和item的ID embedding
│   │   |   ├── item_embed.pt(注意 这里使用的user和item来自原始amazon数据集)
│   │   |   └── user_embed.pt(如果要使用 需要将ID映射到下面user_bundle.csv中的ID user和item对应new_ID)
│   │   ├── data_size.txt 预训练使用的数据量
│   │   ├── pretrain_user_ID_map.json 预训练时的user数字ID与原始数据集中字母序列ID的转换,下同
│   │   ├── pretrain_item_ID_map.json
│   │   └── filter_ui.json 预训练使用的交互数据 被划分成train和test
│   │
|   ├── preprocess
|   │   ├── bundle_intent_embeddings.pkl 使用bert的encoder得到的bundle描述文本的nlp表示
|   │   ├── bundle_intent.csv bundle的ID及其对应的文本描述
|   │   ├── bundle_intent_item_pretrain_ID.csv bundle的ID、对应的文本描述、包含的item的ID，使用的是pretrain时的ID，拼接了intent，是临时文件
|   │   ├── bundle_item_new/original/pretrain_ID.csv bundle和item 的对应关系，其中itemID分别使用的是新的重映射的ID、原始数据中的ID以及在pretrain时的物品ID
|   │   ├── candidate_items_lookup_pretrain.json
|   │   ├── candidate_items_lookup.json 候选物品查询表 是所有bundle中出现过的item集合 两个文件的区别是item的ID 暂时没用
|   │   ├── item_nlp_info_new_ID_embeddings.pkl item的new_IDd对应的使用BERT生成的nlp表征
|   │   ├── item_nlp_info_new_ID.json item的new_ID对应的文本描述
|   │   ├── new_item_ID_map.json item的原始字母ID到new_ID的映射表
|   │   ├── user_bundle_item.csv RL的训练数据从这里读取，逗号隔开的是user_id,bundle_id,bundle中的item_id,用户历史交互序列的item_id
|   │   ├── user_bundle_test.csv RL的测试数据从这里读取
|   │   ├── user_bundle.csv user和bundle的交互记录
|   │   ├── reward_model_data_pretrain_ID_non_vocab.csv bundle的intent，bundle内的物品（使用pretrain模型中的物品ID），候选物品（pretrain），候选物品和bundle内物品的平均相似度、打分；其中5分对应候选物品正是从原本bundle中sample出、2分对应从candidates中sample的和bundle内物品相似度较高的、1分对应candidates中sample出相似度低的、0分代表bundle已满后另sample的物品
│   │   ├── similarity_matrix.npy pretrain的物品相似度矩阵
│   │   └── reward_model_data_pretrain_ID_vocab.csv 使用vocab对上面的csv文件中的物品进行了映射
│   │
│   ├── user_bundle_item.csv 内容和上面preprocess中的一样 为了方便读取复制出来了
│   └── user_bundle_test.csv
|
├── electronic
└── food
```
