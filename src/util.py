from collections import defaultdict
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer


class SASRecDataSet:
    """
    A class for creating SASRec specific dataset used during
    train, validation and testing.

    Attributes:
        usernum: integer, total number of users
        itemnum: integer, total number of items
        User: dict, all the users (keys) with items as values
        Items: set of all the items
        user_train: dict, subset of User that are used for training
        user_valid: dict, subset of User that are used for validation
        user_test: dict, subset of User that are used for testing
        col_sep: column separator in the data file
        filename: data filename
    """

    def __init__(self, **kwargs):
        self.usernum = 0
        self.itemnum = 0
        self.User = defaultdict(list)
        self.User_feat = defaultdict(list)
        self.item_feat = defaultdict(list)
        self.Items = set()
        self.user_list = {}
        self.item_list = {}
        self.user_train = {}
        self.user_valid = {}
        self.user_test = {}
        self.user_train_feat = {}
        self.user_valid_feat = {}
        self.user_test_feat = {}
        self.col_sep = kwargs.get("col_sep", " ")
        self.filename = kwargs.get("filename", None)

        if self.filename:
            with open(self.filename, "r") as fr:
                sample = fr.readline()
            ncols = sample.strip().split(self.col_sep)
            if ncols == 3:
                self.with_time = True
            else:
                self.with_time = False

    def split(self, **kwargs):
        self.filename = kwargs.get("filename", self.filename)
        if not self.filename:
            raise ValueError("Filename is required")

        if self.with_time:
            self.data_partition_with_time()
        else:
            self.data_partition()

    def data_partition(self):
        # assume user/item index starting from 1
        df = pd.read_csv(self.filename, sep='\t', nrows=200)
        # df['weekday'] = df['vplay_ts'].dt.dayofweek
        # df["Is Weekend"] = df['vplay_ts'].dt.dayofweek > 4

        df = df.iloc[1:, :]
        df = df.fillna('')
        item_feat_temp = defaultdict(list)
        model = SentenceTransformer('all-MiniLM-L6-v2')
        user_li, item_li = [], []
        for ind, row in df.iterrows():
            u, uemb, ubias, i, iemb, ibias, tagname, referrer = row['userId'], row['user_emb'], row['user_bias'], row['postId'], row[
                'post_emb'], row['post_bias'], row['tagName'], row['referrer']
            #referrerEmb = model.encode(referrer)
            #tag_name_emb = model.encode(tagname)
            u = int(float(u))
            if u not in user_li:
                user_li.append(u)
            i = int(float(i))
            if i not in item_li:
                item_li.append(i)

            self.usernum = max(user_li.index(u)+1, self.usernum)
            self.itemnum = max(item_li.index(i)+1, self.itemnum)
            temp_li = []
            for _ in uemb.strip('][').split():
                temp_li.append(float(_))
            temp_li.append(float(ubias))
            for _ in iemb.strip('][').split():
                temp_li.append(float(_))
            temp_li.append(float(ibias))
            # for _ in tag_name_emb:
            #     temp_li.append(float(_))
            # for _ in referrerEmb:
            #     temp_li.append(_)
            self.User[user_li.index(u)+1].append(item_li.index(i)+1)
            self.User_feat[user_li.index(u)+1].append(temp_li)
            item_feat_temp[item_li.index(i)+1].append(temp_li)
            self.user_list[user_li.index(u)+1] = u
            self.item_list[item_li.index(i) + 1] = i

        for _ in item_feat_temp.keys():
            self.item_feat[_] = np.mean(item_feat_temp[_], axis=0)

        for user in self.User:
            nfeedback = len(self.User[user])
            if nfeedback < 3:
                self.user_train[user] = self.User[user]
                self.user_train_feat[user] = self.User_feat[user]
                self.user_valid[user] = []
                self.user_test[user] = []
            else:
                self.user_train[user] = self.User[user][:-2]
                self.user_train_feat[user] = self.User_feat[user][:-2]
                self.user_valid[user] = []
                self.user_valid[user].append(self.User[user][-2])
                self.user_valid_feat[user] = []
                self.user_valid_feat[user].append(self.User_feat[user][-2])
                self.user_test[user] = []
                self.user_test[user].append(self.User[user][-1])
                self.user_test_feat[user] = []
                self.user_test_feat[user].append(self.User_feat[user][-1])

    def data_partition_with_time(self):
        # assume user/item index starting from 1
        f = open(self.filename, "r")
        for line in f:
            u, i, t = line.rstrip().split(self.col_sep)
            u = int(u)
            i = int(i)
            t = float(t)
            self.usernum = max(u, self.usernum)
            self.itemnum = max(i, self.itemnum)
            self.User[u].append((i, t))
            self.Items.add(i)

        for user in self.User.keys():
            # sort by time
            items = sorted(self.User[user], key=lambda x: x[1])
            # keep only the items
            items = [x[0] for x in items]
            self.User[user] = items
            nfeedback = len(self.User[user])
            if nfeedback < 3:
                self.user_train[user] = self.User[user]
                self.user_valid[user] = []
                self.user_test[user] = []
            else:
                self.user_train[user] = self.User[user][:-2]
                self.user_valid[user] = []
                self.user_valid[user].append(self.User[user][-2])
                self.user_test[user] = []
                self.user_test[user].append(self.User[user][-1])
