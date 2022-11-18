import pickle
import tensorflow as tf
from src.ssept import SSEPT

lr = 0.001  # learning rate
maxlen = 15  # maximum sequence length for each user
num_blocks = 2  # number of transformer blocks
hidden_units = 100  # number of units in the attention calculation
num_heads = 1  # number of attention heads
dropout_rate = 0.1  # dropout rate
l2_emb = 0.0  # L2 regularization coefficient
num_neg_test = 30  # number of negative examples per positive example
model_name = 'ssept'  # 'sasrec' or 'ssept'

data = pickle.load(open('data.p', 'rb'))

model = SSEPT(item_num=data.itemnum,
              user_num=data.usernum,
              seq_max_len=maxlen,
              num_blocks=num_blocks,
              embedding_dim=hidden_units,  # optional
              user_embedding_dim=10,
              item_embedding_dim=hidden_units,
              attention_dim=hidden_units,
              attention_num_heads=num_heads,
              dropout_rate=dropout_rate,
              conv_dims=[110, 110],
              l2_reg=l2_emb,
              num_neg_test=num_neg_test
              )
path = 'Weights_folder/Weights'
model.load_weights(path)
print("ISP model predictions ", model.predict_inter(data, k=10))
