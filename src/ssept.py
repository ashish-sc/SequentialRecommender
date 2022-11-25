import tensorflow as tf
from src.model import SASREC, Encoder, LayerNormalization
import numpy as np


class SSEPT(SASREC):
    """
    SSE-PT Model
    """

    def __init__(self, **kwargs):
        """Model initialization.

        Args:
            item_num (int): Number of items in the dataset.
            seq_max_len (int): Maximum number of items in user history.
            num_blocks (int): Number of Transformer blocks to be used.
            embedding_dim (int): Item embedding dimension.
            attention_dim (int): Transformer attention dimension.
            conv_dims (list): List of the dimensions of the Feedforward layer.
            dropout_rate (float): Dropout rate.
            l2_reg (float): Coefficient of the L2 regularization.
            num_neg_test (int): Number of negative examples used in testing.
            user_num (int): Number of users in the dataset.
            user_embedding_dim (int): User embedding dimension.
            item_embedding_dim (int): Item embedding dimension.
        """
        super().__init__(**kwargs)

        self.user_num = kwargs.get("user_num", None)  # New
        self.conv_dims = kwargs.get("conv_dims", [200, 200])  # modified
        self.user_embedding_dim = kwargs.get(
            "user_embedding_dim", self.embedding_dim
        )  # extra
        self.item_embedding_dim = kwargs.get("item_embedding_dim", self.embedding_dim)
        self.hidden_units = self.item_embedding_dim + self.user_embedding_dim

        # New, user embedding
        self.user_embedding_layer = tf.keras.layers.Embedding(
            input_dim=self.user_num + 1,
            output_dim=self.user_embedding_dim,
            name="user_embeddings",
            mask_zero=True,
            input_length=1,
            embeddings_regularizer=tf.keras.regularizers.L2(self.l2_reg),
        )
        self.positional_embedding_layer = tf.keras.layers.Embedding(
            self.seq_max_len,
            self.user_embedding_dim + self.item_embedding_dim,  # difference
            name="positional_embeddings",
            mask_zero=False,
            embeddings_regularizer=tf.keras.regularizers.L2(self.l2_reg),
        )
        self.dropout_layer = tf.keras.layers.Dropout(self.dropout_rate)
        self.encoder = Encoder(
            self.num_blocks,
            self.seq_max_len,
            self.hidden_units,
            self.hidden_units,
            self.attention_num_heads,
            self.conv_dims,
            self.dropout_rate,
        )
        self.mask_layer = tf.keras.layers.Masking(mask_value=0)
        self.layer_normalization = LayerNormalization(
            self.seq_max_len, self.hidden_units, 1e-08
        )

    def call(self, x, training):
        """Model forward pass.

        Args:
            x (tf.Tensor): Input tensor.
            training (tf.Tensor): Training tensor.

        Returns:
            tf.Tensor, tf.Tensor, tf.Tensor:
            - Logits of the positive examples.
            - Logits of the negative examples.
            - Mask for nonzero targets
        """

        users = x["users"]
        input_seq = x["input_seq"]
        pos = x["positive"]
        neg = x["negative"]
        seq_feat = tf.convert_to_tensor(x["seq_feat"])
        pos_feat = tf.convert_to_tensor(x["pos_feat"])
        neg_feat = tf.convert_to_tensor(x["neg_feat"])
        mask = tf.expand_dims(tf.cast(tf.not_equal(input_seq, 0), tf.float32), -1)
        seq_embeddings, positional_embeddings = self.embedding(input_seq)

        # User Encoding
        # u0_latent = self.user_embedding_layer(users[0])
        # u0_latent = u0_latent * (self.embedding_dim ** 0.5)
        u_latent = self.user_embedding_layer(users)
        u_latent = u_latent * (self.user_embedding_dim ** 0.5)  # (b, 1, h)
        # return users

        # replicate the user embedding for all the items
        u_latent = tf.tile(u_latent, [1, tf.shape(input_seq)[1], 1])  # (b, s, h)

        seq_embeddings = tf.reshape(
            tf.concat([seq_embeddings, u_latent], 2),
            [tf.shape(input_seq)[0], -1, self.hidden_units],
        )
        seq_embeddings += positional_embeddings

        # dropout
        seq_embeddings = self.dropout_layer(seq_embeddings, training=training)

        # masking
        seq_embeddings *= mask

        # --- ATTENTION BLOCKS ---
        seq_attention = seq_embeddings  # (b, s, h1 + h2)

        seq_attention = self.encoder(seq_attention, training, mask)
        seq_attention = self.layer_normalization(seq_attention)  # (b, s, h1+h2)

        # --- PREDICTION LAYER ---
        # user's sequence embedding
        pos = self.mask_layer(pos)
        neg = self.mask_layer(neg)

        user_emb = tf.reshape(
            u_latent,
            [tf.shape(input_seq)[0] * self.seq_max_len, self.user_embedding_dim],
        )
        pos = tf.reshape(pos, [tf.shape(input_seq)[0] * self.seq_max_len])
        neg = tf.reshape(neg, [tf.shape(input_seq)[0] * self.seq_max_len])
        pos_emb = self.item_embedding_layer(pos)
        neg_emb = self.item_embedding_layer(neg)
        # Add user embeddings
        pos_emb = tf.reshape(tf.concat([pos_emb, user_emb], 1), [-1, self.hidden_units])
        neg_emb = tf.reshape(tf.concat([neg_emb, user_emb], 1), [-1, self.hidden_units])

        seq_emb = tf.reshape(
            seq_attention,
            [tf.shape(input_seq)[0] * self.seq_max_len, self.hidden_units],
        )  # (b*s, d)

        seq_emb = tf.cast(seq_emb, dtype=tf.float32)
        pos_emb = tf.cast(pos_emb, dtype=tf.float32)
        neg_emb = tf.cast(neg_emb, dtype=tf.float32)
        h, w = tf.shape(seq_emb)[0], tf.shape(seq_feat)[2]
        seq_feat = tf.reshape(seq_feat, [h, w])
        h, w = tf.shape(pos_emb)[0], tf.shape(pos_feat)[2]
        pos_feat = tf.reshape(pos_feat, [h, w])
        h, w = tf.shape(neg_emb)[0], tf.shape(neg_feat)[2]
        neg_feat = tf.reshape(neg_feat, [h, w])

        seq_emb = tf.concat([seq_emb, seq_feat], 1)
        pos_emb = tf.concat([pos_emb, pos_feat], 1)
        neg_emb = tf.concat([neg_emb, neg_feat], 1)

        pos_logits = tf.reduce_sum(pos_emb * seq_emb, -1)
        neg_logits = tf.reduce_sum(neg_emb * seq_emb, -1)
        pos_logits = tf.expand_dims(pos_logits, axis=-1)  # (bs, 1)
        # pos_prob = tf.keras.layers.Dense(1, activation='sigmoid')(pos_logits)  # (bs, 1)

        neg_logits = tf.expand_dims(neg_logits, axis=-1)  # (bs, 1)
        # neg_prob = tf.keras.layers.Dense(1, activation='sigmoid')(neg_logits)  # (bs, 1)

        # output = tf.concat([pos_logits, neg_logits], axis=0)

        # masking for loss calculation
        istarget = tf.reshape(
            tf.cast(tf.not_equal(pos, 0), dtype=tf.float32),
            [tf.shape(input_seq)[0] * self.seq_max_len],
        )

        return pos_logits, neg_logits, istarget

    def predict(self, inputs):
        """
        Model prediction for candidate (negative) items

        """
        training = False
        user = inputs["user"]
        input_seq = inputs["input_seq"]
        candidate = inputs["candidate"]
        seq_feat = inputs["seq_feat"]
        cand_feat = inputs["cand_feat"]

        mask = tf.expand_dims(tf.cast(tf.not_equal(input_seq, 0), tf.float32), -1)
        seq_embeddings, positional_embeddings = self.embedding(input_seq)  # (1, s, h)

        u0_latent = self.user_embedding_layer(user)
        u0_latent = u0_latent * (self.user_embedding_dim ** 0.5)  # (1, 1, h)
        u0_latent = tf.squeeze(u0_latent, axis=0)  # (1, h)
        test_user_emb = tf.tile(u0_latent, [1 + self.num_neg_test, 1])  # (101, h)

        u_latent = self.user_embedding_layer(user)
        u_latent = u_latent * (self.user_embedding_dim ** 0.5)  # (b, 1, h)
        u_latent = tf.tile(u_latent, [1, tf.shape(input_seq)[1], 1])  # (b, s, h)

        seq_embeddings = tf.reshape(
            tf.concat([seq_embeddings, u_latent], 2),
            [tf.shape(input_seq)[0], -1, self.hidden_units],
        )
        seq_embeddings += positional_embeddings  # (b, s, h1 + h2)

        seq_embeddings *= mask
        seq_attention = seq_embeddings
        seq_attention = self.encoder(seq_attention, training, mask)
        seq_attention = self.layer_normalization(seq_attention)  # (b, s, h1+h2)
        seq_emb = tf.reshape(
            seq_attention,
            [tf.shape(input_seq)[0] * self.seq_max_len, self.hidden_units],
        )  # (b*s1, h1+h2)

        seq_emb = tf.cast(seq_emb, dtype=tf.float32)
        seq_feat = tf.convert_to_tensor(seq_feat)
        cand_feat = tf.convert_to_tensor(cand_feat)
        seq_feat = tf.reshape(seq_feat, [seq_emb.shape[0], seq_feat.shape[1]])
        seq_emb = tf.concat([seq_emb, seq_feat], 1)

        candidate_emb = self.item_embedding_layer(candidate)  # (b, s2, h2)
        candidate_emb = tf.squeeze(candidate_emb, axis=0)  # (s2, h2)
        candidate_emb = tf.reshape(
            tf.concat([candidate_emb, test_user_emb], 1), [-1, self.hidden_units]
        )  # (b*s2, h1+h2)

        cand_feat = tf.reshape(cand_feat, [candidate_emb.shape[0], cand_feat.shape[1]])
        candidate_emb = tf.concat([candidate_emb, cand_feat], 1)

        candidate_emb = tf.transpose(candidate_emb, perm=[1, 0])  # (h1+h2, b*s2)

        test_logits = tf.matmul(seq_emb, candidate_emb)  # (b*s1, b*s2)

        test_logits = tf.reshape(
            test_logits,
            [tf.shape(input_seq)[0], self.seq_max_len, 1 + self.num_neg_test],
        )  # (1, s, 101)
        test_logits = test_logits[:, -1, :]  # (1, 101)
        return test_logits

    def loss_function(self, pos_logits, neg_logits, istarget):
        """Losses are calculated separately for the positive and negative
        items based on the corresponding logits. A mask is included to
        take care of the zero items (added for padding).

        Args:
            pos_logits (tf.Tensor): Logits of the positive examples.
            neg_logits (tf.Tensor): Logits of the negative examples.
            istarget (tf.Tensor): Mask for nonzero targets.

        Returns:
            float: Loss.
        """

        pos_logits = pos_logits[:, 0]
        neg_logits = neg_logits[:, 0]

        # ignore padding items (0)
        # istarget = tf.reshape(
        #     tf.cast(tf.not_equal(self.pos, 0), dtype=tf.float32),
        #     [tf.shape(self.input_seq)[0] * self.seq_max_len],
        # )
        # for logits
        loss = tf.reduce_sum(
            -tf.math.log(tf.math.sigmoid(pos_logits) + 1e-24) * istarget
            - tf.math.log(1 - tf.math.sigmoid(neg_logits) + 1e-24) * istarget
        ) / tf.reduce_sum(istarget)

        # for probabilities
        # loss = tf.reduce_sum(
        #         - tf.math.log(pos_logits + 1e-24) * istarget -
        #         tf.math.log(1 - neg_logits + 1e-24) * istarget
        # ) / tf.reduce_sum(istarget)
        reg_loss = tf.compat.v1.losses.get_regularization_loss()
        # reg_losses = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.REGULARIZATION_LOSSES)
        # loss += sum(reg_losses)
        loss += reg_loss

        return loss

    @tf.function(input_signature=[
        tf.TensorSpec([None], "string", name="recent_posts"),
        tf.TensorSpec([None], "string", name="cand_posts"),
        tf.TensorSpec([32], tf.float32, name="user_emb"),
        tf.TensorSpec([], tf.float32, name="user_bias"),
        tf.TensorSpec([None, 32], tf.float32, name="recent_post_embs"),
        tf.TensorSpec([None], tf.float32, name="recent_post_bias"),
        tf.TensorSpec([None, 32], tf.float32, name="cand_post_embs"),
        tf.TensorSpec([None], tf.float32, name="cand_post_bias"),
    ])
    def inference(self, recent_posts, cand_posts, user_emb, user_bias, recent_post_embs,
                  recent_post_bias, cand_post_embs, cand_post_bias):
        seq = np.zeros([self.seq_max_len], dtype=np.float64)
        seq_len = tf.shape(user_emb)[0] + tf.shape(recent_post_embs)[1] + 2
        seq_feat = np.zeros((self.seq_max_len, 66), dtype=list)

        cand_user_emb = tf.repeat([user_emb], repeats=len(cand_post_embs), axis=0)
        cand_user_bias = tf.repeat(user_bias, repeats=len(cand_post_bias), axis=0)
        cand_user_feat = tf.concat([cand_user_emb, cand_user_bias[:, None]], axis=1)
        cand_feat = tf.concat([cand_post_embs, cand_post_bias[:, None]], axis=1)
        overall_cand_feat = tf.concat([cand_user_feat, cand_feat], axis=1)

        recent_user_emb = tf.repeat([user_emb], repeats=len(recent_post_embs), axis=0)
        recent_user_bias = tf.repeat(user_bias, repeats=len(recent_post_bias), axis=0)
        recent_user_feat = tf.concat([recent_user_emb, recent_user_bias[:, None]], axis=1)
        recent_feat = tf.concat([recent_post_embs, recent_post_bias[:, None]], axis=1)
        overall_recent_feat = tf.concat([recent_user_feat, recent_feat], axis=1)

        ####### Recent post sequence and feature creation #####################
        len_rposts, len_rfeat = len(recent_posts), len(overall_recent_feat)
        # print("len_rpost ", len_rposts.numpy(), " len_rfeat ", len_rfeat.numpy())
        if self.seq_max_len > len_rposts:
            seq[self.seq_max_len - len_rposts:] = recent_posts
        else:
            seq = recent_posts[-self.seq_max_len:]
        if self.seq_max_len > len_rfeat:
            seq_feat[self.seq_max_len - len_rfeat:] = np.array(overall_recent_feat)
        else:
            seq_feat = np.array(overall_recent_feat[-self.seq_max_len:])

        cand_posts_new = np.zeros([self.num_neg_test + 1], dtype=np.float64)
        cand_seq_feat = np.zeros((self.num_neg_test + 1, seq_len), dtype=list)

        len_cposts, len_cfeat = len(cand_posts), len(overall_cand_feat)
        if self.num_neg_test + 1 > len_cposts:
            cand_posts_new[self.num_neg_test + 1 - len_cposts:] = cand_posts
        else:
            cand_posts_new = cand_posts[-self.num_neg_test - 1:]
        if self.num_neg_test + 1 > len_cfeat:
            cand_seq_feat[self.num_neg_test - len_cfeat + 1:] = np.array(overall_cand_feat)
        else:
            cand_seq_feat = np.array(overall_cand_feat[-self.num_neg_test - 1:])

        ####### Candidate post sequence and feature creation #####################
        inputs = {"user": np.expand_dims(np.array([1]), axis=-1), "input_seq": np.array([seq]),
                  "seq_feat": np.asarray(seq_feat).astype(np.float32),
                  "candidate": np.array([cand_posts_new]),
                  "cand_feat": np.asarray(cand_seq_feat).astype(np.float32)}

        # inverse to get descending sort
        predictions = -1.0 * self.predict(inputs)
        predictions = np.array(predictions)
        predictions = predictions[0]

        return predictions
