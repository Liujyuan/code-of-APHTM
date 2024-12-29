import numpy as np
import matplotlib.pyplot as plt
import os
from gensim.models.coherencemodel import CoherenceModel
import requests
from sklearn.decomposition import PCA
from torch import nn
import manifolds
import torch
def compute_overlap(level1, level2):
    sum_overlap_score = 0.0
    for N in [5, 10, 15]:
        word_idx1 = np.argpartition(level1, -N)[-N:]
        word_idx2 = np.argpartition(level2, -N)[-N:]
        c = 0
        for n in word_idx1:
            if n in word_idx2:
                c += 1
        sum_overlap_score += c / N
    return sum_overlap_score / 3
def compute_hierarchical_affinity_new(topic_dist_1,topic_dist_2, relation):
    child_ha, unchild_ha = [], []
    topic_dist_1 = topic_dist_1 / np.linalg.norm(topic_dist_1, axis=1, keepdims=True)
    topic_dist_2 = topic_dist_2 / np.linalg.norm(topic_dist_2, axis=1, keepdims=True)
    for index in range(32):
        for index_2 in range(128):
            ha = topic_dist_1[index].dot(topic_dist_2[index_2])   
            if (index,index_2) in relation:
                child_ha.append(ha)
            else:
                unchild_ha.append(ha)  
    return np.mean(child_ha), np.mean(unchild_ha)  
def compute_hierarchical_affinity_new_2(topic_dist_1,topic_dist_2, relation):
    child_ha, unchild_ha = [], []
    topic_dist_1 = topic_dist_1 / np.linalg.norm(topic_dist_1, axis=1, keepdims=True)
    topic_dist_2 = topic_dist_2 / np.linalg.norm(topic_dist_2, axis=1, keepdims=True)
    for index in range(8):
        for index_2 in range(32):
            ha = topic_dist_1[index].dot(topic_dist_2[index_2])   
            if (index,index_2) in relation:
                child_ha.append(ha)
            else: 
                unchild_ha.append(ha)  
    return np.mean(child_ha), np.mean(unchild_ha)  
def compute_hierarchical_affinity(topic_dist, relation):
    child_ha, unchild_ha = [], []
    topic_dist = topic_dist / np.linalg.norm(topic_dist, axis=1, keepdims=True)

    for child_index,child in enumerate(relation[0]):
        for parent in relation[1]:
            if parent == child:
                continue
            ha = topic_dist[child] * topic_dist[parent]
            if relation[1][child_index] == parent:
                child_ha.append(ha)
            else:
                unchild_ha.append(ha)           
    return np.mean(child_ha), np.mean(unchild_ha)
def evaluate_topic_diversity(topic_words):
    '''topic_words is in the form of [[w11,w12,...],[w21,w22,...]]'''
    vocab = set(sum(topic_words,[]))
    total = sum(topic_words,[])
    return len(vocab) / len(total)

def build_bert_embedding(embedding_fn, vocab, data_dir):
    print(f"building bert embedding matrix for dit {len(vocab)}")
    tokenize = BertTokenizer.from_pretrained('bert-base-uncased')
    embedding_mat_fn = os.path.join(data_dir, f"bert_emb_{len(vocab)}.npy")

    # if matrix exists
    if os.path.exists(embedding_mat_fn):
        embedding_mat = np.load(embedding_mat_fn)
        return embedding_mat

    # build bert mat
    index = np.array(
        tokenize.encode(list(vocab.token2id.keys()), add_special_tokens=False))
    bert_mat = np.load(embedding_fn)
    bert_emb = bert_mat[index]
    np.save(embedding_mat_fn, bert_emb)
    return bert_emb
# def compute_topic_specialization(topic_word, corpus_topic):
#     print('topic_word',topic_word.shape,corpus_topic.shape)
#     topics_vec = topic_word.sum(axis=0)
#     corpus_topic=corpus_topic.sum(axis=0)
#     print(topics_vec.shape)
#     topics_vec = topics_vec / np.linalg.norm(topics_vec)
#     corpus_topic=corpus_topic/np.linalg.norm(corpus_topic)
#     topics_spec = 1 - topics_vec.dot(corpus_topic.T)
#
#     return topics_spec
def compute_topic_specialization(topic_word, corpus_topic):
  #  print('topic_word',topic_word.shape,corpus_topic.shape)
    if topic_word.shape[0] > 0:
      for i in range(topic_word.shape[0]):
                topic_word[i] = topic_word[i] / np.linalg.norm(topic_word[i])
    corpus_topic=corpus_topic.sum(axis=0)
    topics_vec=topic_word
    corpus_topic=corpus_topic/np.linalg.norm(corpus_topic)
    topics_spec = 1 - topics_vec.dot(corpus_topic.T)
    topics_spec=np.mean(topics_spec)
    return topics_spec
# def compute_topic_specialization(topic_word, corpus_topic):
#     print('topic_word',topic_word.shape,corpus_topic.shape)
#     topics_vec = topic_word
#     if topics_vec.shape[0] > 0:
#         for i in range(topics_vec.shape[0]):
#             topics_vec[i] = topics_vec[i] / np.linalg.norm(topics_vec[i])
#         for i in range(corpus_topic.shape[0]):
#             corpus_topic[i] = corpus_topic[i] / np.linalg.norm(corpus_topic[i])
#         topics_spec = 1 - topics_vec.dot(corpus_topic.T)
#         #print(topics_vec.dot(corpus_topic.T))
#        # print(topics_spec.shape)
#         depth_spec = np.mean(topics_spec)
#         return depth_spec
#     else:
#         return 0
def evaluate_NPMI(test_data, topic_dist, n_list=[5,10,15]):
    TU = 0.0
    for n in n_list:
        TU += compute_coherence(test_data, topic_dist, n)
    TU /= len(n_list)
    return TU

def evaluate_NPMI2(test_data, topic_dist, n_list=[5,10,15]):
    coh_list_n=[]
    for n in n_list:
        coh_list=compute_coherence2(test_data, topic_dist, n)
        coh_list_n= np.concatenate((coh_list_n,coh_list),axis=0)
    return coh_list_n
    
def evaluate_TU(topic_word, n_list=[5,10,15]):
    TU = 0.0
    for n in n_list:
        TU += compute_TU(topic_word, n)
    TU /= len(n_list)
    return TU

def evaluate_TD(topic_word, n_list=[5,10,15]):
    TU = 0.0
    for n in n_list:
        TU += compute_TD(topic_word, n)
    TU /= len(n_list)
    return TU

def compute_TD(topic_word, N):
    topic_size, word_size = np.shape(topic_word)
    if topic_size == 0:
        return 0
    else:
        topic_list = []
        for topic_idx in range(topic_size):
            top_word_idx = np.argpartition(topic_word[topic_idx, :], -N)[-N:]
            topic_list.append(top_word_idx)
        TD = len(np.unique(np.array(topic_list).flatten()))/len(np.array(topic_list).flatten())
        return TD

def compute_TU(topic_word, N):
    topic_size, word_size = np.shape(topic_word)
    if topic_size == 0:
        return 0
    else:
        topic_list = []
        for topic_idx in range(topic_size):
            top_word_idx = np.argpartition(topic_word[topic_idx, :], -N)[-N:]
            topic_list.append(top_word_idx)
        TU = 0
        cnt = [0 for i in range(word_size)]
        for topic in topic_list:
            for word in topic:
                cnt[word] += 1
        for topic in topic_list:
            TU_t = 0
            for word in topic:
                TU_t += 1 / cnt[word]
            TU_t /= N
            TU += TU_t
        TU /= topic_size
        return TU
def build_level(adj_matrix,flag = 0):
    adj_matrix= adj_matrix.T.detach().to("cpu").numpy()

    adj_matrix_new = np.array(adj_matrix.copy())
    for index in range(len(adj_matrix)):
        adj_matrix_new[index][adj_matrix_new[index]==np.max(adj_matrix_new[index])] = -1
        adj_matrix_new[index][adj_matrix_new[index]==np.max(adj_matrix_new[index])] = -1
    adj_matrix_new[adj_matrix_new != -1] = 0
    adj_matrix_new[adj_matrix_new == -1] = 1
    #print('trees',adj_matrix_new.shape,adj_matrix_new)
    trees = adj_matrix_new
    
    relation = np.where(adj_matrix_new == 1)
    print('len(relation[0])',len(relation[0]))
    relation = list(zip(relation[0], relation[1]))
    return trees, relation

def compute_clnpmi(level1, level2, doc_word):

    sum_coherence_score = 0.0
    c = 0

    for N in [5,10,15]:
        word_idx1 = np.argpartition(level1, -N)[-N:]
        word_idx2 = np.argpartition(level2, -N)[-N:]
        
        sum_score = 0.0
        set1 = set(word_idx1)
        set2 = set(word_idx2)
        inter = set1.intersection(set2)
        word_idx1 = list(set1.difference(inter))
        word_idx2 = list(set2.difference(inter))

        for n in range(len(word_idx1)):
            flag_n = doc_word[:, word_idx1[n]] > 0
            p_n = np.sum(flag_n) / len(doc_word)
            for l in range(len(word_idx2)):
                flag_l = doc_word[:, word_idx2[l]] > 0
                p_l = np.sum(flag_l)
                p_nl = np.sum(flag_n * flag_l)
                if p_nl == len(doc_word):
                    sum_score += 1
                elif p_n * p_l * p_nl > 0:
                    p_l = p_l / len(doc_word)
                    p_nl = p_nl / len(doc_word)
                    p_nl += 1e-10
                    sum_score += np.log(p_nl / (p_l * p_n)) / -np.log(p_nl)
                c += 1
        if c > 0:
            sum_score /= c
        else:
            sum_score = 0
        sum_coherence_score += sum_score
    return sum_coherence_score / 3
def build_embedding(embedding_fn, vocab, data_dir):
    print(f"building embedding matrix for dict {len(vocab)} if need...")
    embedding_mat_fn = os.path.join(data_dir, f"embedding_mat_{len(vocab)}.npy")
    
    # if matrix exists
    if os.path.exists(embedding_mat_fn):
        embedding_mat = np.load(embedding_mat_fn)
        return embedding_mat
    
    # build embedding mat
    embedding_index = {}
    with open(embedding_fn, encoding='UTF-8') as fin:
        first_line = True
        l_id = 0
        for line in fin:
            if l_id % 100000 == 0:
                print("loaded %d words embedding..." % l_id)
            if ("glove" not in embedding_fn) and first_line:
                first_line = False
                continue
            line = line.rstrip()
            values = line.split(' ')
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embedding_index[word] = coefs
            l_id += 1
            
    embedding_dim = len(list(embedding_index.values())[0])
    embedding_mat = np.zeros((len(vocab) + 1, embedding_dim))    # -1 is for padding
    for i,word  in vocab.items():
        embedding_vec = embedding_index.get(word)
        if embedding_vec is not None:
            embedding_mat[i] = embedding_vec
    np.save(embedding_mat_fn, embedding_mat)
    return embedding_mat
    
def build_embedding(embedding_fn, vocab, data_dir):
    print(f"building embedding matrix for dict {len(vocab)} if need...")
    embedding_mat_fn = os.path.join(data_dir, f"embedding_mat_{len(vocab)}.npy")
    
    # if matrix exists
    if os.path.exists(embedding_mat_fn):
        embedding_mat = np.load(embedding_mat_fn)
        return embedding_mat
    
    # build embedding mat
    embedding_index = {}
    with open(embedding_fn, encoding='UTF-8') as fin:
        first_line = True
        l_id = 0
        for line in fin:
            if l_id % 100000 == 0:
                print("loaded %d words embedding..." % l_id)
            if ("glove" not in embedding_fn) and first_line:
                first_line = False
                continue
            line = line.rstrip()
            values = line.split(' ')
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embedding_index[word] = coefs
            l_id += 1
            
    embedding_dim = len(list(embedding_index.values())[0])
    embedding_mat = np.zeros((len(vocab) + 1, embedding_dim))    # -1 is for padding
    for i,word  in vocab.items():
        embedding_vec = embedding_index.get(word)
        if embedding_vec is not None:
            embedding_mat[i] = embedding_vec
    np.save(embedding_mat_fn, embedding_mat)
    return embedding_mat

def build_embedding_PCA(embedding_fn, vocab, data_dir):
    print(f"building embedding matrix to PCA for dict {len(vocab)} if need...")
    embedding_PCA_fn = os.path.join(data_dir, f"embedding_mat_PCA_{len(vocab)}.npy")

    if os.path.exists(embedding_PCA_fn):
        embedding_PCA_mat = np.load(embedding_PCA_fn)
        return embedding_PCA_mat
    
    embedding_mat = build_embedding(embedding_fn, vocab, data_dir)
    pca = PCA(n_components=2)
    embedding_PCA_mat = pca.fit_transform(embedding_mat)
    np.save(embedding_PCA_fn, embedding_PCA_mat)
    return embedding_PCA_mat

def compute_coherence(doc_word, topic_word, N):
    # print('computing coherence ...')    
    topic_size, word_size = np.shape(topic_word)
   # print('topic_size',topic_size,word_size)
    doc_size = np.shape(doc_word)[0]
    # find top words'index of each topic
    topic_list = []
    for topic_idx in range(topic_size):
        top_word_idx = np.argpartition(topic_word[topic_idx, :], -N)[-N:]
        topic_list.append(top_word_idx)

    # compute coherence of each topic
    sum_coherence_score = 0.0
    for i in range(topic_size):
        word_array = topic_list[i]
        sum_score = 0.0
        for n in range(N):
            flag_n = doc_word[:, word_array[n]] > 0
            p_n = np.sum(flag_n) / doc_size
            for l in range(n + 1, N):
                flag_l = doc_word[:, word_array[l]] > 0
                p_l = np.sum(flag_l)
                p_nl = np.sum(flag_n * flag_l)
                if p_n * p_l * p_nl > 0:
                    p_l = p_l / doc_size
                    p_nl = p_nl / doc_size
                    sum_score += np.log(p_nl / (p_l * p_n)) / -np.log(p_nl)
        sum_coherence_score += sum_score * (2 / (N * N - N))
    sum_coherence_score = sum_coherence_score / topic_size
    return sum_coherence_score

def compute_coherence2(doc_word, topic_word, N):
    # print('computing coherence ...')
    topic_size, word_size = np.shape(topic_word)
   # print('topic_size',topic_size,word_size)
    doc_size = np.shape(doc_word)[0]
    # find top words'index of each topic
    topic_list = []
    for topic_idx in range(topic_size):
        top_word_idx = np.argpartition(topic_word[topic_idx, :], -N)[-N:]
        topic_list.append(top_word_idx)

    # compute coherence of each topic
    coh_list = []
    for i in range(topic_size):
        word_array = topic_list[i]
        sum_score = 0.0
        for n in range(N):
            flag_n = doc_word[:, word_array[n]] > 0
            p_n = np.sum(flag_n) / doc_size
            for l in range(n + 1, N):
                flag_l = doc_word[:, word_array[l]] > 0
                p_l = np.sum(flag_l)
                p_nl = np.sum(flag_n * flag_l)
                if p_n * p_l * p_nl > 0:
                    p_l = p_l / doc_size
                    p_nl = p_nl / doc_size
                    sum_score += np.log(p_nl / (p_l * p_n)) / -np.log(p_nl)
        coh_list.append( sum_score * (2 / (N * N - N)))
 #   print('coh_list',len(coh_list))
   # sum_coherence_score = coh_list / topic_size
    return coh_list

def evaluate_coherence(topic_words, texts, vocab):
    coherence = {}
    methods = ["c_v", "c_npmi", "c_uci", "u_mass"]
    for method in methods:
        coherence[method] = CoherenceModel(topics=topic_words, texts=texts, dictionary=vocab, coherence=method).get_coherence()
    return coherence
    

def evaluate_topic_diversity(topic_words):
    '''topic_words is in the form of [[w11,w12,...],[w21,w22,...]]'''
    vocab = set(sum(topic_words,[]))
    total = sum(topic_words,[])
    return len(vocab) / len(total)


def print_topic_word(topic_word, vocab, N):
    topic_size, word_size = np.shape(topic_word)

    top_word_idx = np.argsort(topic_word, axis=1)
    top_word_N = top_word_idx[:,-N:]

    for k, top_word_k in enumerate(top_word_N[:,::-1]):
        top_words = [vocab[id] for id in top_word_k]
        print(f'Topic {k}:{top_words}')


def get_palmetto(topic, url):
    res = requests.get(url, {'words' : ' '.join(topic)})
    coh = np.float(res.text)
    return coh



def plot_acc(history):
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Valid'], loc='upper left')
    plt.show()

def plot_loss(history):
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.title("Model Loss")
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Valid'], loc="best")


class GIN(nn.Module):
    def __init__(self, num_layers, in_features, out_features, hidden_features=[],
                 eps=0.0, train_eps=True, residual=True, batchNorm=True,
                 nonlinearity='leaky_relu', negative_slope=0.1):
        super(GIN, self).__init__()

        self.GINLayers = nn.ModuleList()

        if in_features != out_features:
            first_layer_res = False
        else:
            first_layer_res = True
        self.GINLayers.append(GINLayer(MLP(in_features, out_features, hidden_features, batchNorm,
                                           nonlinearity, negative_slope),
                                       eps, train_eps, first_layer_res))
        for i in range(num_layers - 1):
            self.GINLayers.append(GINLayer(MLP(out_features, out_features, hidden_features, batchNorm,
                                               nonlinearity, negative_slope),
                                           eps, train_eps, residual))

        self.reset_parameters()

    def reset_parameters(self):
        for l in self.GINLayers:
            l.reset_parameters()

    def forward(self, input, adj):
        for l in self.GINLayers:
            input = l(input, adj)
        return input

def cosine_similarity(a, b):
    a=a.squeeze()
    b=b.squeeze()
    dot_product = np.dot(a, b.T)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    similarity = dot_product / (norm_a * norm_b)
    # 如果需要转换为“距离”，可以用 1 - similarity
    return similarity
manifold_f = getattr(manifolds, 'PoincareBall')()
def fuc(x,y):
    x=torch.tensor(x)
    y=torch.tensor(y)
    re=manifold_f.dist(
        x, y, torch.tensor(-0.01))
   # cos_sim_1=x@y
    return re
def custom_distance(a, b, metric_func):
    """
    应用自定义距离度量函数计算两个点或向量间的距离。
    """
    if metric_func=='cos':
        dst=1-cosine_similarity(a,b)
    if metric_func=='eul':
        dst=np.sqrt(np.sum((a - b) ** 2))
    if metric_func=='hp':
        dst=fuc(a,b)
    return dst


def compute_a_and_b(X, labels, metric_func):
    """
    计算每个样本的a值和b值。
    """
    unique_labels = np.unique(labels)
    a = np.zeros(len(X))
    b = np.full(len(X), np.inf)

    for label in unique_labels:
        cluster_indices = labels == label
        print('cluster_indices',cluster_indices,len(X),X[0].shape)
        cluster_points = X[cluster_indices]
        other_points = X[~cluster_indices]

        # 计算a(i): 样本到其簇内其他点的平均距离
        intra_distances = np.sum(
            custom_distance(cluster_points[:, np.newaxis], cluster_points[np.newaxis, :], metric_func), axis=-1)
        a[cluster_indices] = np.mean(intra_distances, axis=-1)

        # 计算b(i): 样本到最近簇的平均距离
        for other_label in unique_labels:
            if other_label != label:
                other_cluster_points = X[labels == other_label]
                distances_to_other_cluster = np.min(
                    custom_distance(cluster_points[:, np.newaxis], other_cluster_points[np.newaxis, :], metric_func),
                    axis=-1)
                b[cluster_indices] = np.minimum(b[cluster_indices], np.mean(distances_to_other_cluster))

    return a, b


def custom_silhouette(X, labels, metric_func):
    """
    使用自定义距离度量计算轮廓系数。
    """
    a, b = compute_a_and_b(X, labels, metric_func)
    s = (b - a) / np.maximum(a, b)
    return np.mean(s), s


# 示例自定义距离函数，如曼哈顿距离



# 示例数据
# X = np.random.rand(100, 2)  # 示例数据集
# assignments = np.random.choice([0, 1, 2], size=len(X))  # 示例聚类标签

# 计算轮廓系数
# avg_silhouette, silhouettes = custom_silhouette(X, assignments, 'cos')
# print(f"Average Silhouette Coefficient: {avg_silhouette}")