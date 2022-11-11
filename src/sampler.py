import numpy as np
from multiprocessing import Process, Queue, set_start_method
from operator import itemgetter


def random_neq(left, right, s):
    t = np.random.randint(left, right)
    while t in s:
        t = np.random.randint(left, right)
    return int(t)


def sample_function(
        user_train, user_feat, item_feat, usernum, itemnum, batch_size, maxlen, result_queue, seed
):
    """Batch sampler that creates a sequence of negative items based on the
    original sequence of items (positive) that the user has interacted with.

    Args:
        user_train (dict): dictionary of training exampled for each user
        usernum (int): number of users
        itemnum (int): number of items
        batch_size (int): batch size
        maxlen (int): maximum input sequence length
        result_queue (multiprocessing.Queue): queue for storing sample results
        seed (int): seed for random generator
    """

    def sample():
        user = np.random.randint(1, usernum + 1)
        while len(user_train[user]) <= 1:
            user = np.random.randint(1, usernum + 1)

        seq = np.zeros([maxlen], dtype=np.float64)
        pos = np.zeros([maxlen], dtype=np.float64)
        neg = np.zeros([maxlen], dtype=np.float64)
        nxt = user_train[user][-1]
        feat = user_feat[user][-1]
        idx = maxlen - 1

        ts = set(user_train[user])
        seqLen = len(user_feat[user][-1])
        seq_feat, pos_feat, neg_feat = np.zeros((maxlen, seqLen), dtype=list), np.zeros((maxlen, seqLen), dtype=list), \
                                       np.zeros((maxlen, seqLen), dtype=list)

        # for i, j in zip(reversed(user_train[user][:-1]), reversed(user_feat[user][:-1])):
        #     seq[idx] = i
        #     pos[idx] = nxt
        #     seq_feat[idx] = j
        #     pos_feat[idx] = feat
        #     # print("feat ",feat)
        #     if nxt != 0:
        #         neg[idx] = random_neq(1, itemnum + 1, ts)
        #         # print("neg[idx] ", neg[idx], " item_feat[neg[idx]] ",item_feat[neg[idx]][0])
        #         neg_feat[idx] = list(item_feat[neg[idx]][0])
        #     nxt, feat = i, j
        #     idx -= 1
        #     if idx == -1:
        #         break

        len_utrain, len_ufeat = len(user_train[user][:-1]), len(user_feat[user][:-1])
        if maxlen > len_utrain:
            seq[maxlen-len_utrain:], pos[maxlen-len_utrain-1:] = user_train[user][:-1], user_train[user]
        else:
            seq, pos = user_train[user][-maxlen-1:-1], user_train[user][-maxlen:]
        neg = [random_neq(1, itemnum + 1, ts) for _ in range(maxlen)]
        if maxlen > len_ufeat:
            seq_feat[maxlen-len_ufeat:], pos_feat[maxlen-len_ufeat-1:] = np.array(user_feat[user][:-1]), np.array(user_feat[user])
        else:
            seq_feat, pos_feat = np.array(user_feat[user][-maxlen-1:-1]), np.array(
                user_feat[user][-maxlen:])
        neg_feat = itemgetter(*neg)(item_feat)

        return user, seq, pos, neg, seq_feat, pos_feat, neg_feat

    np.random.seed(seed)
    while True:
        one_batch = []
        for _ in range(batch_size):
            one_batch.append(sample())

        result_queue.put(zip(*one_batch))


class WarpSampler(object):
    """Sampler object that creates an iterator for feeding batch data while training.

    Attributes:
        User: dict, all the users (keys) with items as values
        usernum: integer, total number of users
        itemnum: integer, total number of items
        batch_size (int): batch size
        maxlen (int): maximum input sequence length
        n_workers (int): number of workers for parallel execution
    """

    def __init__(self, User, User_feat, item_feat, usernum, itemnum, batch_size=64, maxlen=10, n_workers=1):
        self.result_queue = Queue(maxsize=n_workers * 10)
        self.processors = []
        for i in range(n_workers):
            self.processors.append(
                Process(
                    target=sample_function,
                    args=(
                        User,
                        User_feat,
                        item_feat,
                        usernum,
                        itemnum,
                        batch_size,
                        maxlen,
                        self.result_queue,
                        2022,#np.random.randint(2e9),
                    ),
                )
            )
            set_start_method('fork', force=True)
            self.processors[-1].daemon = True
            self.processors[-1].start()

    def next_batch(self):
        return self.result_queue.get()

    def close(self):
        for p in self.processors:
            p.terminate()
            p.join()
