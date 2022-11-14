import sys
import os

os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import json
import numpy as np
import pandas as pd
import gc
import tensorflow as tf
from pathlib import Path
import time
import pickle

tf.get_logger().setLevel('ERROR')  # only show error messages

# Transformer Based Models
from src.ssept import SSEPT

# Sampler for sequential prediction
from src.sampler import WarpSampler
from src.util import SASRecDataSet

# from google.cloud import bigquery

print("System version: {}".format(sys.version))
print("Tensorflow version: {}".format(tf.__version__))

# bqclient = bigquery.Client()
#
# query_string = """
# select *
#     from `maximal-furnace-783.sc_cg_isp.sampled_200_2`
# """
#
# df = (
#     bqclient.query(query_string)
#     .result()
#     .to_dataframe(
#         create_bqstorage_client=True
#     )
# )
# df_k = df.iloc[:]
# del df
# gc.collect()
# base_path = "/home/ashish.gupta/ISP/posts/"
#
# like_post_emb_path = base_path+"like/post_embedding_updated.csv"
# like_global_bias_path = base_path+"like/global_bias.txt"
# fav_post_emb_path = base_path+"fav/post_embedding_updated.csv"
# fav_global_bias_path = base_path+"fav/global_bias.txt"
# share_post_emb_path = base_path+"share/post_embedding_updated.csv"
# share_global_bias_path = base_path+"share/global_bias.txt"
# vplay_post_emb_path = base_path+"vplay2/post_embedding_updated.csv"
# vplay_global_bias_path = base_path+"vplay2/global_bias.txt"
# vclick_post_emb_path = base_path+"vclick/post_embedding_updated.csv"
# vclick_global_bias_path = base_path+"vclick/global_bias.txt"
# vskip_post_emb_path = base_path+"vskip/post_embedding_updated.csv"
# vskip_global_bias_path = base_path+"vskip/global_bias.txt"
#
# headers = ['time', 'postId', 'embs', 'bias']
# dtypes = {'time': 'str', 'postId': 'str', 'embs': 'str', 'bias': 'float'}
# parse_dates = ['time']
# like_df = pd.read_csv(like_post_emb_path, header=None, names=headers, converters={"embs": json.loads}, dtype=dtypes, parse_dates=parse_dates)
#
# fav_df = pd.read_csv(fav_post_emb_path, header=None, names=headers, converters={"embs": json.loads}, dtype=dtypes, parse_dates=parse_dates)
#
# share_df = pd.read_csv(share_post_emb_path, header=None, names=headers, converters={"embs": json.loads}, dtype=dtypes, parse_dates=parse_dates)
#
# vplay_df = pd.read_csv(vplay_post_emb_path, header=None, names=headers, converters={"embs": json.loads}, dtype=dtypes, parse_dates=parse_dates)
#
# vclick_df = pd.read_csv(vclick_post_emb_path, header=None, names=headers, converters={"embs": json.loads}, dtype=dtypes, parse_dates=parse_dates)
#
# #vskip_df = pd.read_csv(vskip_post_emb_path, header=None, names=headers, converters={"embs": json.loads}, dtype=dtypes, parse_dates=parse_dates)
#
#
#
# like_df.drop('time', axis=1, inplace=True)
# fav_df.drop('time', axis=1, inplace=True)
# share_df.drop('time', axis=1, inplace=True)
# vplay_df.drop('time', axis=1, inplace=True)
# vclick_df.drop('time', axis=1, inplace=True)
# #vskip_df.drop('time', axis=1, inplace=True)
#
# post_df = pd.concat([like_df, fav_df, share_df, vplay_df, vclick_df])#, vskip_df])
# embs_df = post_df.set_index('postId')['embs'].groupby('postId').apply(list).apply(lambda x:np.mean(x,0)).reset_index()
# bias_df = post_df.set_index('postId')['bias'].groupby('postId').agg(bias= 'mean').reset_index()
#
# post_df2 = pd.merge(embs_df, bias_df, on='postId', how='inner')
# print("Read complete for post embeddings")
#
# base_path = "/home/ashish.gupta/ISP/user/"
#
# like_user_emb_path = base_path+"like/user_embedding_updated.csv"
# fav_user_emb_path = base_path+"fav/user_embedding_updated.csv"
# share_user_emb_path = base_path+"share/user_embedding_updated.csv"
# vplay_user_emb_path = base_path+"vplay2/user_embedding_updated.csv"
# vclick_user_emb_path = base_path+"vclick/user_embedding_updated.csv"
# vskip_user_emb_path = base_path+"vskip/user_embedding_updated.csv"
#
# headers = ['time', 'userId', 'embs', 'bias']
# dtypes = {'time': 'str', 'userId': 'str', 'embs': 'str', 'bias': 'float'}
# parse_dates = ['time']
# user_like_df = pd.read_csv(like_user_emb_path, header=None, names=headers, converters={"embs": json.loads}, dtype=dtypes, parse_dates=parse_dates)
# print("user like completed ")
# user_fav_df = pd.read_csv(fav_user_emb_path, header=None, names=headers, converters={"embs": json.loads}, dtype=dtypes, parse_dates=parse_dates)
# print("user fav completed ")
# user_share_df = pd.read_csv(share_user_emb_path, header=None, names=headers, converters={"embs": json.loads}, dtype=dtypes, parse_dates=parse_dates)
# print("user share completed ")
# user_vplay_df = pd.read_csv(vplay_user_emb_path, header=None, names=headers, converters={"embs": json.loads}, dtype=dtypes, parse_dates=parse_dates)
# print("user vplay completed ")
# user_vclick_df = pd.read_csv(vclick_user_emb_path, header=None, names=headers, converters={"embs": json.loads}, dtype=dtypes, parse_dates=parse_dates)
# print("user vclick completed ")
# #user_vskip_df = pd.read_csv(vskip_user_emb_path, header=None, names=headers, converters={"embs": json.loads}, dtype=dtypes, parse_dates=parse_dates)
#
#
# user_like_df.drop('time', axis=1, inplace=True)
# user_fav_df.drop('time', axis=1, inplace=True)
# user_share_df.drop('time', axis=1, inplace=True)
# user_vplay_df.drop('time', axis=1, inplace=True)
# user_vclick_df.drop('time', axis=1, inplace=True)
# #user_vskip_df.drop('time', axis=1, inplace=True)
#
#
# user_df = pd.concat([user_like_df, user_fav_df, user_share_df, user_vplay_df, user_vclick_df])#, user_vskip_df])
# user_embs_df = user_df.set_index('userId')['embs'].groupby('userId').apply(list).apply(lambda x: np.mean(x, 0)).reset_index()
# user_bias_df = user_df.set_index('userId')['bias'].groupby('userId').agg(bias= 'mean').reset_index()
# user_df2 = pd.merge(user_embs_df, user_bias_df, on='userId', how='inner')
# print("Read complete for user embeddings")
# df2 = pd.merge(user_df2, df_k, on='userId', how='inner')
# df2.rename(columns={'embs': 'user_emb', 'bias': 'user_bias'}, inplace = True)
# post_df2.rename(columns = {'embs': 'post_emb', 'bias': 'post_bias'}, inplace = True)
#
# df3 = pd.merge(df2, post_df2, on='postId', how='inner')
# #df3 = df3.sort_values(by=["userId"])
# df3.drop(columns=["vplay_ts"], inplace=True)
# df3 = df3.fillna('')
# df3.to_csv('isp_set_forSSEPT2.tsv', sep='\t', index=False)

# dfk = pd.read_csv("data/isp_set_forSSEPT2.tsv", sep='\t')
# dfk.head()


num_epochs = 5
batch_size = 2
RANDOM_SEED = 100  # Set None for non-deterministic result

lr = 0.001  # learning rate
maxlen = 50  # maximum sequence length for each user
num_blocks = 2  # number of transformer blocks
hidden_units = 100  # number of units in the attention calculation
num_heads = 1  # number of attention heads
dropout_rate = 0.1  # dropout rate
l2_emb = 0.0  # L2 regularization coefficient
num_neg_test = 100  # number of negative examples per positive example
model_name = 'ssept'  # 'sasrec' or 'ssept'


def data_prep():
    data = SASRecDataSet(filename='data/isp_set_forSSEPT2.tsv', col_sep="\t")
    # create train, validation and test splits
    data.split()

    with open('data.p', 'wb') as fp:
        pickle.dump(data, fp, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    data_prep()
    print("data read complete")
    with open('data.p', 'rb') as fp:
        data = pickle.load(fp)

    model = SSEPT(item_num=data.itemnum,
                  user_num=data.usernum,
                  seq_max_len=maxlen,
                  num_blocks=num_blocks,
                  # embedding_dim=hidden_units,  # optional
                  user_embedding_dim=10,
                  item_embedding_dim=hidden_units,
                  attention_dim=hidden_units,
                  attention_num_heads=num_heads,
                  dropout_rate=dropout_rate,
                  conv_dims=[110, 110],
                  l2_reg=l2_emb,
                  num_neg_test=num_neg_test
                  )

    sampler = WarpSampler(data.user_train, data.user_train_feat, data.item_feat, data.usernum, data.itemnum, batch_size=batch_size,
                          maxlen=maxlen, n_workers=3)

    checkpoint_path = Path("./training_1/cp.ckpt")
    checkpoint_dir = os.path.dirname(checkpoint_path)

    # Create a callback that saves the model's weights
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                     save_weights_only=True,
                                                     verbose=1)
    start = time.time()
    t_test = model.train(data, sampler, num_epochs=num_epochs, batch_size=batch_size, lr=lr, val_epoch=6)

    # Save model weights
    path = 'Weights_folder/Weights'
    # tf.saved_model.save(model, path)
    model.save(path)
    print('Model Saved!')

    # load model
    model = tf.keras.models.load_model(path)#model.load_weights(path)
    print('Model Loaded!')

    end = time.time()
    train_time = end - start
    print('Time cost for training is {0:.2f} mins'.format(end - start))
    res_syn = {"ndcg@10": t_test[0], "Hit@10": t_test[1]}
    print(res_syn)
