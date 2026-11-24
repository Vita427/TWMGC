import numpy as np
from munkres import Munkres
from sklearn import metrics
from sklearn.metrics.cluster import adjusted_rand_score as ari_score
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score


def cluster_acc(y_true, y_pred):
    """ACC and macro F1 after Hungarian matching of clusters to labels."""
    y_true = y_true - np.min(y_true)
    l1 = list(set(y_true))
    l2 = list(set(y_pred))
    ind = 0
    if len(l1) != len(l2):
        for i in l1:
            if i not in l2:
                y_pred[ind] = i
                ind += 1
    l2 = list(set(y_pred))
    if len(l1) != len(l2):
        raise ValueError("the number of clusters differs from the number of classes")

    cost = np.zeros((len(l1), len(l2)), dtype=int)
    for i, c1 in enumerate(l1):
        mps = [i1 for i1, e1 in enumerate(y_true) if e1 == c1]
        for j, c2 in enumerate(l2):
            cost[i][j] = len([i1 for i1 in mps if y_pred[i1] == c2])

    indexes = Munkres().compute(cost.__neg__().tolist())
    new_predict = np.zeros(len(y_pred))
    for i, c in enumerate(l1):
        c2 = l2[indexes[i][1]]
        ai = [k for k, elm in enumerate(y_pred) if elm == c2]
        new_predict[ai] = c
    return (metrics.accuracy_score(y_true, new_predict),
            metrics.f1_score(y_true, new_predict, average='macro'))


def eva(y_true, y_pred, epoch=0):
    acc, f1 = cluster_acc(y_true, y_pred)
    nmi = nmi_score(y_true, y_pred, average_method='arithmetic')
    ari = ari_score(y_true, y_pred)
    print('epoch {}: acc {:.4f}, nmi {:.4f}, ari {:.4f}, f1 {:.4f}'.format(epoch, acc, nmi, ari, f1))
    return acc, nmi, ari, f1


def data_split(full_list, n_sample):
    """Shuffle the node indices and split them into batches."""
    import random
    random.shuffle(full_list)
    return [full_list[i:i + n_sample] for i in range(0, len(full_list), n_sample)]
