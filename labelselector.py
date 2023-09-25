from tokenize import group
from sklearn.impute import KNNImputer
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE

import random
import numpy as np

def split_trainvalid_clusterbased(sample_n, train_n, features, cluster_n):
    embeds = TSNE(n_components=2, n_iter=2000, learning_rate='auto', init='pca', random_state=0).fit_transform(features)
    clusters = KMeans(n_clusters=cluster_n, random_state=0).fit_predict(embeds)
    sample_seqs = np.asarray(list(range(sample_n)))
    train = list()
    for i in range(cluster_n):
        group_indices = sample_seqs[clusters == i]

        #* sampling
        #* (train_n / cluster_n) samples per cluster
        group_selected = group_indices[random.sample(range(len(group_indices)), int(train_n / cluster_n))]
        train.extend(group_selected)
        #print(i, group_selected)
    
    valid = [x for x in range(sample_n) if (x not in train)]

    return train, valid, embeds, clusters
    
def split_trainvalid_randomsampling(sample_n, train_n):
    random.seed(0)
    train = random.sample(range(sample_n), train_n)
    valid = [x for x in range(sample_n) if (x not in train)]
    return train, valid 
    
def split_trainvalid_fixed(sample_n, train_n):
    train = list(range(train_n))
    valid = list(range(train_n, sample_n))
    return train, valid

def split_trainvalid_randomsamplingfromfixedrange(sample_n, train_n, range_n):
    train = random.sample(range(range_n), train_n)
    valid = [x for x in range(sample_n) if (x not in train)]
    return train, valid 


if __name__ == "__main__":
    #train_idx, valid_idx, embeds, clVusters = split_trainvalid_clusterbased(140, 50, np.random.rand(140, 25), 50)

    train_idx, valid_idx = split_trainvalid_randomsamplingfromfixedrange(100, 10, 50)

    train_idx, valid_idx = split_trainvalid_randomsampling(50, 10)
    print("train:", train_idx)
    print("valid:", valid_idx)
    train_idx, valid_idx = split_trainvalid_randomsampling(50, 10)
    print("train:", train_idx)
    print("valid:", valid_idx)
    


    