# import hypertools as hyp
import matplotlib
import matplotlib.pyplot as plt
#import manifolds.poincare
import matplotlib
import copy
import pickle
import time
import scipy.sparse as sp
from customized_linear import CustomizedLinear
from utils import *
from learning_utils import *
from scipy import sparse
import numpy as np
from pyparsing import Word
import torch.optim as optim
import yaml
from numpy.random import normal

# import km
from sklearn import metrics
from torch.autograd import Variable
from torch.nn import init
from tqdm import tqdm
import utils
from reader import TextReader
# from layers import LinkPredictionLoss_cosine
from sklearn.metrics.pairwise import cosine_similarity

Tensor = torch.cuda.FloatTensor
np.random.seed(0)
torch.manual_seed(0)
torch.autograd.set_detect_anomaly(True)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

torch.cuda.empty_cache()

guide_para_0=0.05#
guide_para_1=0.05#
guide_para_2=0.05#
guide_para=[guide_para_0,guide_para_1,guide_para_2]
adj_alpha_0=2#
adj_alpha_1=2#
adj_alpha_2=2#
adj_alpha=[adj_alpha_0,adj_alpha_1,adj_alpha_2]

plot_map_vis=False
plot_map=False
compute_cluster=False

inv_flag = False
use_GG=True  #get_dist 图引导嵌入
use_CL=False
use_SE=False
use_MSE=False

use_MG=False   #M引导主题嵌入 (废

use_GCL=False  #link prediction

print('GMM_ori')
print(inv_flag,use_GG,use_CL,use_SE,use_MSE,use_MG,use_GCL,'guide_para:',guide_para,'adj_alpha:',adj_alpha)
def fuc(x,y):
    cos_sim_2 = cosine_similarity(x.reshape(1, -1), y.reshape(1, -1))
   # cos_sim_1=x@y
    return (1-cos_sim_2)
def kl_loss(z_mean, z_stddev):
    mean_sq = z_mean * z_mean
    stddev_sq = z_stddev * z_stddev
    return 0.5 * torch.mean(mean_sq + stddev_sq - torch.log(stddev_sq) - 1)
class LinkPredictionLoss_cosine(nn.Module):
    def __init__(self):
        super(LinkPredictionLoss_cosine, self).__init__()

    def forward(self, emb, adj):
        '''
        Parameters
        ----------
        emb : Tensor
            An MxE tensor, the embedding of the ith node is stored in emb[i,:].
        adj : Tensor
            An MxM tensor, adjacent matrix of the graph.

        Returns
        -------
        loss : float
            The link prediction loss.
        '''
        emb_norm = emb.norm(dim=1, keepdim=True)
        emb_norm = emb / (emb_norm + 1e-6)
        adj_pred = torch.matmul(emb_norm, emb_norm.t())  #这里不应该是cos吗
        loss = torch.mean(torch.pow(adj - adj_pred, 2))

        return loss
class LossFunctions:
    eps = 1e-8
    def reconstruction_loss(self, real, predicted, dropout_mask=None, rec_type="mse"):
        if rec_type == "mse":
            if dropout_mask is None:
                loss = -torch.sum(torch.log(predicted+1e-8) * (real+1e-8))
               # loss = (real - predicted).pow(2)
            else:
                loss = torch.sum((real - predicted).pow(2) * dropout_mask) / torch.sum(
                    dropout_mask
                )
        elif rec_type == "bce":
            loss = F.binary_cross_entropy(predicted, real, reduction="none").mean()
        else:
            raise Exception
        return loss

    def log_normal(self, x, mu, var):

        if self.eps > 0.0:
            var = var + self.eps
        return -0.5 * torch.mean(
            torch.log(torch.FloatTensor([2.0 * np.pi]).to(device)).sum(0)
            + torch.log(var)
            + torch.pow(x - mu, 2) / var,
            dim=-1,
        )

    def gaussian_loss(
        self, z, z_mu, z_var, z_mu_prior, z_var_prior
    ):
        loss = self.log_normal(z, z_mu, z_var) - self.log_normal(z, z_mu_prior, z_var_prior)
        return loss.sum()
    def entropy(self, logits, targets):
        log_q = F.log_softmax(logits, dim=-1)
        return -torch.sum(torch.sum(targets * log_q, dim=-1))


class GumbelSoftmax(nn.Module):

  def __init__(self, f_dim, c_dim):
    super(GumbelSoftmax, self).__init__()
    self.logits = nn.Linear(f_dim, c_dim)
    self.f_dim = f_dim
    self.c_dim = c_dim

  def sample_gumbel(self, shape, is_cuda=False, eps=1e-20):
    U = torch.rand(shape)
    if is_cuda:
      U = U.to(device)
    return -torch.log(-torch.log(U + eps) + eps)

  def gumbel_softmax_sample(self, logits, temperature):
    y = logits + self.sample_gumbel(logits.size(), logits.is_cuda)
    return F.softmax(y / temperature, dim=-1)

  def gumbel_softmax(self, logits, temperature, hard=False):
    """
    ST-gumple-softmax
    input: [*, n_class]
    return: flatten --> [*, n_class] an one-hot vector
    """
    #categorical_dim = 10
    y = self.gumbel_softmax_sample(logits, temperature)

    if not hard:
        return y

    shape = y.size()
    _, ind = y.max(dim=-1)
    y_hard = torch.zeros_like(y).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)
    # Set gradients w.r.t. y_hard gradients w.r.t. y
    y_hard = (y_hard - y).detach() + y
    return y_hard

  def forward(self, x, temperature=1.0, hard=False):
    logits = self.logits(x).view(-1, self.c_dim)
    prob = F.softmax(logits, dim=-1)
    y = self.gumbel_softmax(logits, temperature, hard)
    return logits, prob, y

class Gaussian(nn.Module):
    def __init__(self, in_dim, z_dim):
        super(Gaussian, self).__init__()
        self.mu = nn.Linear(in_dim, z_dim)
        self.var = nn.Linear(in_dim, z_dim)

    def forward(self, x):
        mu = self.mu(x)
        logvar = self.var(x)
        return mu, logvar

# Encoder
class InferenceNet(nn.Module):
    def __init__(self,topic_num_1,topic_num_2,topic_num_3, x_dim, z_dim, y_dim, n_gene, nonLinear):
        super(InferenceNet, self).__init__()
        self.encoder = nn.Sequential(nn.Linear(topic_num_1,topic_num_2), nn.BatchNorm1d(topic_num_2), nonLinear)
        self.encoder_2 = nn.Sequential(nn.Linear(topic_num_2,topic_num_3), nn.BatchNorm1d(topic_num_3), nonLinear)
        self.inference_qyx3 = torch.nn.ModuleList(
            [
                nn.Linear(topic_num_3, 300),  # 64 1
                nn.BatchNorm1d(300),
                nonLinear,
                GumbelSoftmax(300, y_dim),  # 1 256
            ]
        )
        self.inference_qzyx3 = torch.nn.ModuleList(
            [
                nn.Linear(topic_num_3 + y_dim, 300),
                nn.BatchNorm1d(300),
                nonLinear,
                Gaussian(300, topic_num_3),
            ]
        )

    def reparameterize(self, mu, var):
        std = torch.sqrt(var + 1e-10)
        noise = torch.randn_like(std)
        z = mu + noise * std
        return z

    def qyx3(self, x,temperature,hard):
        num_layers = len(self.inference_qyx3)
        for i, layer in enumerate(self.inference_qyx3):
            if i == num_layers - 1:
                x = layer(x, temperature, hard)
            else:
                x = layer(x)
        return x
    def qzxy3(self, x, y):
        concat = torch.cat((x.squeeze(2), y), dim=1)
        for layer in self.inference_qzyx3:
            concat = layer(concat)
        return concat


    def forward(self, x, adj, adj_2, adj_3, temperature, hard = 0):
        if inv_flag ==True:
            x_1 = torch.matmul(adj.to(torch.float32),x.squeeze(2).T).T
            x_2 = self.encoder(x_1)
            x_2 = torch.matmul(adj_2.to(torch.float32),x_2.T).T
            x_3 = self.encoder_2(x_2)
            x_3 = torch.matmul(adj_3.to(torch.float32),x_3.T).T
        else:
            x_1 = x.squeeze(2)
            x_2 = self.encoder(x_1)
            x_3 = self.encoder_2(x_2)

        logits_3, prob_3, y_3  = self.qyx3(x_3,temperature, hard = 0)
        mu_3, logvar_3 = self.qzxy3(x_3.view(x_3.size(0), -1, 1), y_3)
        var_3 = torch.exp(logvar_3)
        z_3 = self.reparameterize(mu_3, var_3)
        output_3 = {"mean": mu_3, "var": var_3, "gaussian": z_3, "categorical": y_3,'logits': logits_3, 'prob_cat': prob_3}
        return output_3

# Decoder
class GenerativeNet(nn.Module):
    def __init__(self, topic_num_1,topic_num_2,topic_num_3, x_dim=1, z_dim=1, y_dim=256, n_gene=None, nonLinear=None):
        super(GenerativeNet, self).__init__()
        self.n_gene = n_gene
        self.y_mu_1 = nn.Sequential(nn.Linear(y_dim, topic_num_3))
        self.y_var_1 = nn.Sequential(nn.Linear(y_dim, topic_num_3))
        self.decoder = nn.Sequential(CustomizedLinear(torch.ones(topic_num_3,topic_num_2),bias=False), nn.BatchNorm1d(topic_num_2), nonLinear)
        self.decoder_2 = nn.Sequential(CustomizedLinear(torch.ones(topic_num_2,topic_num_1),bias=False), nn.BatchNorm1d(topic_num_1), nonLinear)

        if True:
            print('Constraining decoder to positive weights', flush=True)

            self.decoder[0].reset_params_pos()
            self.decoder[0].weight.data *= self.decoder[0].mask
            self.decoder_2[0].reset_params_pos()
            self.decoder_2[0].weight.data *= self.decoder_2[0].mask

        self.generative_pxz = torch.nn.ModuleList(
            [
                nn.BatchNorm1d(topic_num_3),
                nonLinear,
            ]
        )
        self.generative_pxz_1 = torch.nn.ModuleList(
            [
                nn.BatchNorm1d(topic_num_2),
                nonLinear,
            ]
        )
        self.generative_pxz_2 = torch.nn.ModuleList(
            [
                nn.BatchNorm1d(topic_num_1),
                nonLinear,
            ]
        )

    def pzy1(self, y):
        y_mu = self.y_mu_1(y)
        y_logvar = self.y_var_1(y)
        return y_mu, y_logvar
    def pxz(self, z):
        for layer in self.generative_pxz:
            z = layer(z)
        return z
    def pxz_1(self, z):
        for layer in self.generative_pxz_1:
            z = layer(z)
        return z
    def pxz_2(self, z):
        for layer in self.generative_pxz_2:
            z = layer(z)
        return z

    def forward(
        self,
        z,
        y_3,
        adj_A_t_inv_2,
        adj_A_t_inv_1,
        adj_A_t_3,
    ):
        y_mu_3, y_logvar_3 = self.pzy1(y_3)
        y_var_3 = torch.exp(y_logvar_3)

        if inv_flag ==True:
            z = torch.matmul(adj_A_t_3.to(torch.float32), z.T).T
            out_1 = self.pxz(z)
            z_2 = self.decoder(z)
            z_2 = torch.matmul(adj_A_t_inv_2.to(torch.float32), z_2.T).T
            out_2 = self.pxz_1(z_2)
            z_3 = self.decoder_2(z_2)
            z_3 =  torch.matmul(adj_A_t_inv_1.to(torch.float32), z_3.T).T
            out_3 = self.pxz_2(z_3)
        else:
            out_1 = self.pxz(z)
            z_2 = self.decoder(z)
            out_2 = self.pxz_1(z_2)
            z_3 = self.decoder_2(z_2)
            out_3 = self.pxz_2(z_3)

        output_1 = { "x_rec": out_1}
        output_2 = { "x_rec": out_2}
        output_3 = {"y_mean": y_mu_3, "y_var": y_var_3, "x_rec": out_3}
        return output_1, output_2, output_3

class FuseEmbeddings(nn.Module):
    def __init__(self, user_embed_dim, poi_embed_dim):
        super(FuseEmbeddings, self).__init__()
        embed_dim = user_embed_dim + poi_embed_dim
        self.fuse_embed = nn.Linear(embed_dim, user_embed_dim)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, user_embed, poi_embed):
        x = self.fuse_embed(torch.cat((user_embed, poi_embed), -1))
        x = self.leaky_relu(x)
       # print("???????????????")
        return x
class net(nn.Module):
    def __init__(
        self,
        max_topic_num=64,
        batch_size=512,
        adj_A=None,
        adj_A_2=None,
        adj_A_3=None,
        emb_mat=None,
        topic_num_1=None,
        topic_num_2=None,
        topic_num_3=None,
        vocab_num=None,
        hidden_num=None,
        Adj_Graph=None,
        **kwargs,
    ):
        super(net, self).__init__()
        print("net topic_num_1={}".format(topic_num_1))

        self.dropout = nn.Dropout(0.1)
        self.max_topic_num = max_topic_num
        xavier_init = torch.distributions.Uniform(-0.05,0.05)
        if emb_mat == None:
            self.word_embed = nn.Parameter(torch.rand(hidden_num, vocab_num))
        else:
            print("Using pre-train word embedding")
            self.word_embed = nn.Parameter(emb_mat)
        self.vocab_num=vocab_num
        self.topic_embed = nn.Parameter(xavier_init.sample((topic_num_1, hidden_num)))
        self.topic_embed_1 = nn.Parameter(xavier_init.sample((topic_num_2, hidden_num)))
        self.topic_embed_2 = nn.Parameter(xavier_init.sample((topic_num_3, hidden_num)))
        self.eta = nn.Linear(vocab_num, 3)#n
        self.adj_A = nn.Parameter(
            Variable(torch.from_numpy(adj_A).double(), requires_grad=True, name="adj_A")
        )
        self.adj_A_2 = nn.Parameter(
            Variable(
                torch.from_numpy(adj_A_2).double(), requires_grad=True, name="adj_A_2"
            )
        )
        self.adj_A_3 = nn.Parameter(
            Variable(
                torch.from_numpy(adj_A_3).double(), requires_grad=True, name="adj_A_3"
            )
        )

        self.encoder = nn.Sequential(nn.Linear(vocab_num, max_topic_num), nn.BatchNorm1d(max_topic_num), nn.Tanh())
        x_dim, y_dim, z_dim = 64, 10, 10  # x:  y:   z:
        self.n_gene = n_gene = len(adj_A)  # topic_num_1
        self.batch_size=batch_size
        self.inference = InferenceNet( topic_num_1,topic_num_2,topic_num_3,x_dim, y_dim,z_dim, n_gene, nn.Tanh())
        self.generative = GenerativeNet(topic_num_1,topic_num_2,topic_num_3,x_dim, y_dim,z_dim, n_gene, nn.Tanh())
        self.fuse = FuseEmbeddings(hidden_num, hidden_num)
        self.losses = LossFunctions()
        #新增加的组件
        self.Adj_Graph = Adj_Graph

        self.topk_max = 1
        self.topk_min = 1



        for m in self.modules():
            if (
                type(m) == nn.Linear
                or type(m) == nn.Conv2d
                or type(m) == nn.ConvTranspose2d
            ):
                torch.nn.init.xavier_normal_(m.weight)
                if m.bias.data is not None:
                    init.constant_(m.bias, 0)

        self.cos = torch.nn.CosineSimilarity(dim=-1)
        hidden_num_gcn = hidden_num
        if emb_mat == None:
            self.word_embed_graph = nn.Parameter(torch.rand(vocab_num, hidden_num_gcn))
        else:
            print("Using pre-train word embedding for graph")
            self.word_embed_graph = nn.Parameter(emb_mat.T)

        self.LinkPredictionLoss_cosine = LinkPredictionLoss_cosine()
    def to_np(self,x):
        return x.cpu().detach().numpy()

    def build_tree(self, dependency_matrix_0_1, dependency_matrix_1_2):
        [level0, level1] = dependency_matrix_0_1.shape  # 32*32
        level2 = dependency_matrix_1_2.shape[1]  # 64*32
        parents_1 = np.argmax(self.to_np(dependency_matrix_0_1), axis=0)
        parents_2 = np.argmax(self.to_np(dependency_matrix_1_2), axis=0)

    def get_sharpened(self,preds):
           targets = preds ** 2 / preds.sum(dim=0)
           targets = (targets.t() / targets.sum(dim=1)).t()

           return  targets


    def get_topic_dist(self, level=2,use_gg=True):

        if use_gg == True:
            word_embed_guide = self.word_embed @ self.Adj_Graph[2]
            word_embed_return_2 = self.word_embed + guide_para_2 * word_embed_guide  # +0.001*word_embed_guide_2
            word_embed_guide = self.word_embed @ self.Adj_Graph[1]
            word_embed_return_1 = self.word_embed + guide_para_1 * word_embed_guide
            word_embed_guide = self.word_embed @ self.Adj_Graph[0]
            word_embed_return_0 = self.word_embed + guide_para_0 * word_embed_guide
        else:
            word_embed_return_2, word_embed_return_1, word_embed_return_0 = self.word_embed, self.word_embed, self.word_embed

        if level == 2:
            return torch.softmax(self.topic_embed_2 @ word_embed_return_2, dim=1)
        elif level == 1:
            return torch.softmax(self.topic_embed_1 @ word_embed_return_1, dim=1)
        elif level == 0:
            return torch.softmax(self.topic_embed @ word_embed_return_0, dim=1)

    def encode(self, x):
        p1 = self.encoder(x)
        return p1

    def decode(self, x_ori, out_1, out_2, out_3):
        out_3 = torch.softmax(out_3, dim=1)
        out_1 = torch.softmax(out_1, dim=1)
        out_2 = torch.softmax(out_2, dim=1)
        self.topic_embed.data = F.normalize(self.topic_embed.data, dim=-1)#8,300
        self.topic_embed_1.data = F.normalize(self.topic_embed_1.data, dim=-1)
        self.topic_embed_2.data = F.normalize(self.topic_embed_2.data, dim=-1)

        beta = torch.softmax(self.topic_embed @ self.word_embed, dim=1)
        beta_2 = torch.softmax(self.topic_embed_1 @ self.word_embed, dim=1)
        beta_3 = torch.softmax(self.topic_embed_2 @ self.word_embed, dim=1)

        p1 = out_3 @ beta # #
        p2 = out_2 @ beta_2
        p3 = out_1 @ beta_3
        p_fin = (p1.T+p2.T+p3.T)/3.0
        return p_fin.T
    def normalize_adj(self, adj: sp.csr_matrix) -> sp.coo_matrix:
        adj = sp.coo_matrix(adj)
        adj_ = adj
        rowsum = np.array(adj_.sum(0))
        rowsum_power = []
        for i in rowsum:
            for j in i:
                if j !=0 :
                    j_power = np.power(j, -0.5)
                    rowsum_power.append(j_power)
                else:
                    j_power = 0
                    rowsum_power.append(j_power)
        rowsum_power = np.array(rowsum_power)
        degree_mat_inv_sqrt = sp.diags(rowsum_power)
        adj_norm = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
        return adj_norm

    def _one_minus_A_t(self, adj):

        adj_normalized = abs(adj)
        adj_normalized = Tensor(np.eye(adj_normalized.shape[0])).to(device) - (adj_normalized.transpose(0, 1)).to(device)
        return adj_normalized

    def endecode(self,x,x_ori,temperature,adj_A_t,adj_A_t_2,adj_A_t_3,adj_A_t_inv,adj_A_t_inv_2,adj_A_t_inv_3):
        x = self.encode(x)
        x = x.view(x.size(0), -1, 1)
        out_inf_1 = self.inference(  #
            x, adj_A_t, adj_A_t_2, adj_A_t_3, temperature, x_ori.view(x.size(0), -1, 1)
        )
        z_3, y_3 = out_inf_1["gaussian"], out_inf_1["categorical"]
        output_1, output_2, output_3 = self.generative(  # here
            z_3,
            y_3,
            adj_A_t_inv_2,
            adj_A_t_inv,
            adj_A_t_inv_3,
        )  # here

        dec_1 = output_1["x_rec"]  # there is still a,not x~
        dec_2 = output_2["x_rec"]
        dec_3 = output_3["x_rec"]
        return dec_1,dec_2,dec_3
    def get_theta(self):
        beta = torch.softmax(self.topic_embed @ self.word_embed, dim=-1)
        beta_2 = torch.softmax(self.topic_embed_1 @ self.word_embed, dim=-1)
        beta_3 = torch.softmax(self.topic_embed_2 @ self.word_embed, dim=-1)
        theta = beta.T
        theta_2 = beta_2.T
        theta_3 = beta_3.T
        return theta,theta_2,theta_3

    def cons(self,x_ori,p,dec_1,dec_2,dec_3,temperature,
             adj_A_t, adj_A_t_2, adj_A_t_3, adj_A_t_inv, adj_A_t_inv_2, adj_A_t_inv_3):
        theta, theta_2, theta_3=self.get_theta()
        target3 = self.get_sharpened(theta_3).detach()
        target2 = self.get_sharpened(theta_2).detach()
        target1 = self.get_sharpened(theta).detach()


        max_ids_target3 = torch.topk(target3.T, k=1, dim=1).indices
        min_ids_target3 = torch.topk(target3.T, k=1, largest=False, dim=1).indices

        max_ids_target3=max_ids_target3.expand(-1,512)
        max_ids_target3=max_ids_target3.T
        min_ids_target3 = min_ids_target3.expand(-1, 512)
        min_ids_target3 = min_ids_target3.T
        tmp_x_P = copy.copy(x_ori)
        tmp_x_N = copy.copy(x_ori)
       #返回p层的复原最大值最小值
        max_ids = torch.topk(p, k=self.topk_max, dim=1).indices
        min_ids = torch.topk(p, k=self.topk_min, largest=False, dim=1).indices

        max_ids_10 = torch.topk(p, k=1, dim=1).indices

        min_ids_rdm=torch.zeros_like(min_ids)

        for idx,ep in enumerate(p):

           zero_ids=(ep == 0).nonzero()

           id = torch.randperm(zero_ids.nelement())
           zero_ids = zero_ids.view(-1)[id].view(zero_ids.size())
           row = zero_ids[0, 0]
           min_ids_rdm[idx]=row

        max_ids_rdm = torch.zeros_like(max_ids)
        for idx, ep in enumerate(max_ids_10):
            id = torch.randperm(ep.nelement())
            ep = ep.view(-1)[id].view(ep.size())
            row = ep[0]
            max_ids_rdm[idx] = row



        positive_X = tmp_x_P.scatter_(1, min_ids_rdm, 1)
        negative_X = tmp_x_N.scatter_(1, max_ids, 0)

        positive_X = positive_X.scatter_(1, min_ids_target3, 1)
        negative_X = negative_X.scatter_(1, max_ids_target3, 0)

        #positive_X = tmp_x_P.scatter_(1, min_ids_target3, 1)
        #negative_X = tmp_x_N.scatter_(1, max_ids_target3, 0)

        # p层的N/P扰动，引起1,2,3变化
        dec_1N, dec_2N, dec_3N = self.endecode(negative_X, x_ori, temperature, adj_A_t, adj_A_t_2, adj_A_t_3,
                                                     adj_A_t_inv, adj_A_t_inv_2, adj_A_t_inv_3)
        dec_1P, dec_2P, dec_3P = self.endecode(positive_X, x_ori, temperature, adj_A_t, adj_A_t_2, adj_A_t_3,
                                                     adj_A_t_inv, adj_A_t_inv_2, adj_A_t_inv_3)

        positive_component_1 = self.cos(dec_1P, dec_1).mean()
        negative_component_1 = self.cos(dec_1N, dec_1).mean()
        positive_component_2 = self.cos(dec_2P, dec_2).mean()
        negative_component_2 = self.cos(dec_2N, dec_2).mean()
        positive_component_3 = self.cos(dec_3P, dec_3).mean()
        negative_component_3 = self.cos(dec_3N, dec_3).mean()

        return positive_component_1,negative_component_1,positive_component_2,negative_component_2,positive_component_3,negative_component_3

    def Intra_path(self):
        word_embed_guide = self.word_embed @ self.Adj_Graph[2]
        word_embed_gg = self.word_embed + guide_para_2 * word_embed_guide
        Link_loss = self.LinkPredictionLoss_cosine(word_embed_gg.t(), self.Adj_Graph[2])
        return Link_loss,word_embed_gg

    def Inter_path(self,word_embed_gg):
        theta, theta_2, theta_3 = self.get_theta()
        targets_3 = self.get_sharpened(theta_3).detach()
        targets_2 = self.get_sharpened(theta_2).detach()
        targets_1 = self.get_sharpened(theta).detach()
        max_ids_target3 = torch.topk(targets_3.T, k=self.topk_max, dim=1).indices.squeeze()  # 128,1
        max_ids_target2 = torch.topk(targets_2.T, k=self.topk_max, dim=1).indices.squeeze()
        max_ids_target1 = torch.topk(targets_1.T, k=self.topk_max, dim=1).indices.squeeze()
        A3 = word_embed_gg.T[max_ids_target3]  # 128,h
        A2 = word_embed_gg.T[max_ids_target2]
        A1 = word_embed_gg.T[max_ids_target1]

        self.topic_embed_2_AT = self.fuse(A3, self.topic_embed_2)
        self.topic_embed_1_AT = self.fuse(A2, self.topic_embed_1)
        self.topic_embed_AT = self.fuse(A1, self.topic_embed)
        M2_t = torch.softmax(self.topic_embed_AT @ self.topic_embed_1_AT.T, dim=-1)  # 32,8
        M1_t = torch.softmax(self.topic_embed_1_AT @ self.topic_embed_2_AT.T, dim=-1)  # 128,32

        M1 = self.generative.decoder[0].weight
        M1 = torch.softmax(M1, dim=-1)
        M2 = self.generative.decoder_2[0].weight
        M2 = torch.softmax(M2, dim=-1)
        M1_loss = F.kl_div(M1.log(), M1_t, reduction='batchmean')
        M2_loss = F.kl_div(M2.log(), M2_t, reduction='batchmean')

        Inter_loss = M1_loss + M2_loss

        return Inter_loss

    def Cluster_path(self,):
        theta, theta_2, theta_3 = self.get_theta()
        targets_3 = self.get_sharpened(theta_3).detach()
        targets_2 = self.get_sharpened(theta_2).detach()
        targets_1 = self.get_sharpened(theta).detach()

        clus_loss_3 = F.kl_div(theta_3.log(), targets_3, reduction='batchmean')
        clus_loss_2 = F.kl_div(theta_2.log(), targets_2, reduction='batchmean')
        clus_loss_1 = F.kl_div(theta.log(), targets_1, reduction='batchmean')
        Cluster_loss = clus_loss_3
        return Cluster_loss




    def forward(self, x, dropout_mask=None, temperature=1.0,temperature_cls=1.0, hard=0):


#原本ELBO+GMM
        x_ori = copy.copy(x)
        x = self.encode(x)
        x = x.view(x.size(0), -1, 1)

        mask = Variable(
            torch.from_numpy(np.ones(self.n_gene) - np.eye(self.n_gene)).float(),
            requires_grad=False,
        ).to(device)
        self.adj_A=self.adj_A.to(device)
        adj_A_t = self._one_minus_A_t(self.adj_A * mask)
        adj_A_t_inv = torch.inverse(adj_A_t)

        mask_1 = Variable(
            torch.from_numpy(np.ones(32) - np.eye(32)).float(), requires_grad=False
        ).to(device)
        adj_A_t_2 = self._one_minus_A_t(self.adj_A_2 * mask_1)#*self.adj_AW1
        adj_A_t_inv_2 = torch.inverse(adj_A_t_2)

        mask_2 = Variable(
            torch.from_numpy(np.ones(128) - np.eye(128)).float(), requires_grad=False
        ).to(device)
        adj_A_t_3 = self._one_minus_A_t(self.adj_A_3 * mask_2)#*self.adj_AW2
        adj_A_t_inv_3 = torch.inverse(adj_A_t_3)

        out_inf_1 = self.inference(  #
            x, adj_A_t, adj_A_t_2, adj_A_t_3, temperature, x_ori.view(x.size(0), -1, 1)
        )
        z_3, y_3 = out_inf_1["gaussian"], out_inf_1["categorical"]
        output_1, output_2, output_3 = self.generative(  # here
            z_3,
            y_3,
            adj_A_t_inv_2,
            adj_A_t_inv,
            adj_A_t_inv_3,
        )  # here

        dec_1 = output_1["x_rec"]# there is still a,not x~
        dec_2 = output_2["x_rec"]
        dec_3 = output_3["x_rec"]
        dec_res = self.decode(x_ori,dec_1,dec_2,dec_3)

        loss_rec_1 = self.losses.reconstruction_loss(
            x_ori, dec_res, dropout_mask, "mse"
        )
        loss_gauss_3 = (
            self.losses.gaussian_loss(
                z_3,
                out_inf_1["mean"],
                out_inf_1["var"],
                output_3["y_mean"],
                output_3["y_var"],
            )
            * 1)
        loss_cat_3 = (-self.losses.entropy(out_inf_1['logits'], out_inf_1['prob_cat']) - np.log(0.1))


        # 对主题内部词关系的层内约束 A->w
        Intra_loss, word_embed_gg = self.Intra_path()

        # 对主题和关键词的层间约束 A+t->t
        Inter_loss = self.Inter_path(word_embed_gg)

        # 后验聚类约束
        Cluster_loss = self.Cluster_path()


        # MASK A的对比学习
        pc1_1, nc_1_1, pc_1_2, nc_1_2, pc_1_3, nc_1_3 = self.cons(x_ori, x_ori, dec_1, dec_2, dec_3,
                                                                  temperature, adj_A_t,
                                                                  adj_A_t_2, adj_A_t_3,
                                                                  adj_A_t_inv,
                                                                  adj_A_t_inv_2,adj_A_t_inv_3)

        contrastive_loss_1_1 = -torch.log(
            torch.exp(pc1_1) / (torch.exp(pc1_1) + 0.5 * torch.exp(nc_1_1)))
        contrastive_loss_1_2 = -torch.log(
            torch.exp(pc_1_2) / (torch.exp(pc_1_2) + 0.5 * torch.exp(nc_1_2)))
        contrastive_loss_1_3 = -torch.log(
            torch.exp(pc_1_3) / (torch.exp(pc_1_3) + 0.5 * torch.exp(nc_1_3)))
        CF_loss = contrastive_loss_1_2 + contrastive_loss_1_3 + contrastive_loss_1_1


        ELBO_GMM = (
            loss_rec_1
            + loss_gauss_3
            + loss_cat_3
        )

        loss=ELBO_GMM+10*(self.batch_size*Cluster_loss*temperature_cls+self.batch_size*Intra_loss+self.batch_size*CF_loss*2)


        return loss

class AMM_no_dag(object):
    def __init__(
        self,
        reader=None,
        max_topic_num=64,
        model_path=None,
        emb_mat=None,
        topic_num_1=None,
        topic_num_2=None,
        topic_num_3=None,
        epochs=None,
        batch_size=None,
        learning_rate=None,
        **kwargs,
    ):
        # prepare dataset
        if reader == None:
            raise Exception(" [!] Expected data reader")

        self.reader = reader
        self.model_path = model_path
        self.n_classes = self.reader.get_n_classes()  # document class
        self.topic_num_1 = topic_num_1
        self.topic_num_2 = topic_num_2
        self.topic_num_3 = topic_num_3

        self.adj = self.initalize_A(topic_num_1)
        self.adj_2 = self.initalize_A(topic_num_2)  # topic_num_2
        self.adj_3 = self.initalize_A(topic_num_3)  # topic_num_3
        print("AMM_no_dag init model.")

        if emb_mat is None:
            self.Net = net(
                max_topic_num,
                batch_size,
                adj_A=self.adj,
                adj_A_2=self.adj_2,
                adj_A_3=self.adj_3,
                topic_num_1=self.topic_num_1,
                topic_num_2=self.topic_num_2,
                topic_num_3=self.topic_num_3,
                **kwargs,
            ).to(device)
        else:
            emb_mat = torch.from_numpy(emb_mat.astype(np.float32)).to(device)
            self.Net = net(
                max_topic_num,
                batch_size,
                adj_A=self.adj,
                adj_A_2=self.adj_2,
                adj_A_3=self.adj_3,
                topic_num_1=self.topic_num_1,
                topic_num_2=self.topic_num_2,
                topic_num_3=self.topic_num_3,
                emb_mat=emb_mat.T,
                **kwargs,
            ).to(device)

        print(self.Net)

        self.topic_num = max_topic_num
        self.pi_ave = 0
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = learning_rate
        # optimizer uses ADAM

    def initalize_A(self, topic_nums=16):
        A = np.ones([topic_nums, topic_nums]) / (topic_nums - 1) + (
            np.random.rand(topic_nums * topic_nums) * 0.0002
        ).reshape([topic_nums, topic_nums])
        for i in range(topic_nums):
            A[i, i] = 0
        return A

    def to_np(self, x):
        return x.data.cpu().numpy()

    def save_model(self):
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        torch.save(self.Net.state_dict(), f"{self.model_path}/model.pkl")
        with open(f"{self.model_path}/topic_num.txt", "w") as f:
            f.write(str(self.topic_num))
        np.save(f"{self.model_path}/pi_ave.npy", self.pi_ave)
        print(f"Models save to  {self.model_path}/model.pkl")

    def load_model(self, model_filename="model.pkl"):
        model_path = os.path.join(self.model_path, model_filename)

        self.Net.load_state_dict(torch.load(model_path))
        # self.Net = torch.load(model_path)
        with open(f"{self.model_path}/topic_num.txt", "r") as f:
            self.topic_num = int(f.read())
        self.pi_ave = np.load(f"{self.model_path}/pi_ave.npy")
        print("AMM_no_dag model loaded from {}.".format(model_path))


    def get_word_topic(self, data):
        word_topic = self.Net.infer(torch.from_numpy(data).to(device))
        word_topic = self.to_np(word_topic)
        return word_topic

    def get_topic_dist(self, level=2,use_gg=True):
        # topic_dist = self.Net.get_topic_dist()[self.topics]
        topic_dist = self.Net.get_topic_dist(level,use_gg)
        return topic_dist

    def get_topic_word(self, level=2, top_k=15, use_gg=True,vocab=None):
        topic_dist = self.get_topic_dist(level,use_gg)
        vals, indices = torch.topk(topic_dist, top_k, dim=1)
        indices = self.to_np(indices).tolist()
        topic_words = [
            [self.reader.vocab[idx] for idx in indices[i]]
            for i in range(topic_dist.shape[0])
        ]
        return topic_words
    def get_norm(self,x):
        return x*(np.linalg.norm(x))
    def get_topic_parents(self, mat):
        return 0

    def evaluate(self):
        # 重定向回文件
        _, _, texts = self.reader.get_sequence("all")

        for level in range(3):
            topic_word = self.get_topic_word(
                top_k=10, level=level, vocab=self.reader.vocab,use_gg=use_GG
            )
            # 打印top N的主题词
            for k, top_word_k in enumerate(topic_word):
                print(f"Topic {k}:{top_word_k}")

    def save_topics_by_level(self,topics, level):
        # 确定存储目录
        save_dir = "topics_by_level"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # 为当前level创建一个特定的文件
        filename = f"topics_level_{level}.txt"
        filepath = os.path.join(save_dir, filename)

        # 打开文件并写入数据
        with open(filepath, 'w', encoding='utf-8') as f:
            for index, topic in enumerate(topics):
                f.write(f"Topic {index}: {topic}\n")

    def save_top_word(self):

        # 假设你已经有了一个get_topic_word方法来获取主题词
        # 并且你已经循环遍历了所有的level
        for level in range(3):  # 替换为你的实际level数量
            topic_words = self.get_topic_word(top_k=10, level=level, vocab=self.reader.vocab, use_gg=use_GG)
            self.save_topics_by_level(topic_words, level)

    # NPMI
    def sampling(self, flag,data):
        # 计算coherence
        test_data, test_label, _ = self.reader.get_matrix(data, mode="count")
        topic_dist_2 = self.to_np(self.get_topic_dist(level=2,use_gg=use_GG))  # 最低层主题的 coherence
        topic_dist_1 = self.to_np(self.get_topic_dist(level=1,use_gg=use_GG))
        topic_dist_0 = self.to_np(self.get_topic_dist(level=0,use_gg=use_GG))
    #    topic_dist_res = self.to_np(self.get_topic_dist(level=3))
        topic_dist = np.concatenate(
            (np.concatenate((topic_dist_2, topic_dist_1), axis=0), topic_dist_0), axis=0
        )
      # train_coherence_res = utils.evaluate_NPMI(test_data, topic_dist_res)
        train_coherence_2 = utils.evaluate_NPMI(test_data, topic_dist_2)
        train_coherence_1 = utils.evaluate_NPMI(test_data, topic_dist_1)
        train_coherence_0 = utils.evaluate_NPMI(test_data, topic_dist_0)
        train_coherence = utils.evaluate_NPMI(test_data, topic_dist)
        TU2 = utils.evaluate_TU(topic_dist_2)
        TU1 = utils.evaluate_TU(topic_dist_1)
        TU0 = utils.evaluate_TU(topic_dist_0)
        TU = utils.evaluate_TU(topic_dist)
        coh_inc=(train_coherence - best_coh) / best_coh
        TU_inc=(TU - best_TU) / best_TU
        best_TQ=best_TU*best_coh
        TQ=TU*train_coherence
        TQ_inc=(TQ-best_TQ)/best_TQ
        score = 0.5* coh_inc +0.5* TU_inc
        if flag == 1:
            print("TU level 2: " + str(TU2))
            print("TU level 1: " + str(TU1))
            print("TU level 0: " + str(TU0))
            print("TU: " + str(TU))
            print("Topic coherence  level 2: ", train_coherence_2)
            print("Topic coherence  level 1: ", train_coherence_1)
            print("Topic coherence  level 0: ", train_coherence_0)
        print('TU_inc:',TU_inc,'coh_inc:',coh_inc,'TQ_inc:',TQ_inc)
        print("TU: " + str(TU))
        print("Topic coherence:", train_coherence)
        print("TQ:"+str(TQ))
        print("score",score)
      #  if  score> self.best_coherence:
        if score > self.best_score and TU_inc>0 and coh_inc>0 and coh_inc>TU_inc:
            self.best_score = score

            print("New best_score found!!  is",self.best_score)
            self.save_model()

        pass

    def get_batches(self, batch_size=300, rand=True):
        n, d = self.train_data.shape

        batchs = n // batch_size
        while True:
            idxs = np.arange(self.train_data.shape[0])

            if rand:
                np.random.shuffle(idxs)

            for count in range(batchs):
                wordcount = []
                beg = count * batch_size
                end = (count + 1) * batch_size

                idx = idxs[beg:end]
                data = self.train_data[idx].toarray()
                data = torch.from_numpy(data).to(device)
                yield data

    def show_M(self):
        trees_1, relation_1 = utils.build_level(
            self.Net.generative.decoder[0].weight, flag=1)
        trees_2, relation_2 = utils.build_level(
            self.Net.generative.decoder_2[0].weight)
        print('trees_1',trees_1 )
        print('relation_1',relation_1)
        print('trees_2', trees_2)
        print('relation_2', relation_2)
        with open(self.model_path+'/relation_1.pkl', 'wb') as file:
            pickle.dump(relation_1, file)
        with open(self.model_path+'/relation_2.pkl', 'wb') as file:
            pickle.dump(relation_2, file)




    def train(self, epochs=320, batch_size=256, data_type="train+valid"):
        self.t_begin = time.time()
        batch_size = self.batch_size
        (
            self.train_data,
            self.train_label,
            self.train_text,
        ) = self.reader.get_sparse_matrix(data_type, mode="count")   #可以从reader内修改mode，改为tfidf

        (
            self.train_data2,
            self.train_label2,
            self.train_text2,
        ) = self.reader.get_sparse_matrix(data_type, mode="tfidf")  # 可以从reader内修改mode，改为tfidf

        self.train_generator = self.get_batches(batch_size)
        data_size = self.train_data.shape[0]
        n_batchs = data_size // batch_size
        print(batch_size)
        self.best_coherence = -1
        self.best_score= -1
        optimizer = optim.RMSprop(self.Net.parameters(), lr=self.lr)
        optimizer2 = optim.RMSprop(
            [self.Net.adj_A, self.Net.adj_A_2, self.Net.adj_A_3], lr=self.lr * 0.2
        )
        clipper = WeightClipper(frequency=1)
        for epoch in tqdm(range(self.epochs)):

            self.Net.train()
            epoch_word_all = 0
            doc_count = 0

            if epoch % (3) < 1:  #
                self.Net.adj_A.requires_grad = False
                self.Net.adj_A_2.requires_grad = False
                self.Net.adj_A_3.requires_grad = False

            else:
                self.Net.adj_A.requires_grad = True
                self.Net.adj_A_2.requires_grad = True
                self.Net.adj_A_3.requires_grad = True

            for i in range(n_batchs):
                optimizer.zero_grad()
                optimizer2.zero_grad()
                temperature = max(0.95 ** epoch, 0.5)
                temperature_cls=max(0.999 ** epoch, 0.01)
                ori_docs = next(self.train_generator)

                doc_count += ori_docs.shape[0]
                count_batch = []
                for idx in range(ori_docs.shape[0]):
                    count_batch.append(np.sum(self.to_np(ori_docs[idx])))

                epoch_word_all += np.sum(count_batch)
                count_batch = np.add(count_batch, 1e-12)
                loss = self.Net(
                    ori_docs, temperature = temperature,temperature_cls=temperature_cls
                )
                sparse_loss = (
                    1 * torch.mean(torch.abs(self.Net.adj_A))
                    + 1 * torch.mean(torch.abs(self.Net.adj_A_2))
                    + 1 * torch.mean(torch.abs(self.Net.adj_A_3))
                )
                if inv_flag:
                    loss = loss + sparse_loss
                else:
                    loss = loss

                loss.backward()

                if epoch % (3) < 1:
                    optimizer.step()
                else:
                    optimizer2.step()
                if True:
                    self.Net.generative.decoder[0].apply(clipper)
                    self.Net.generative.decoder_2[0].apply(clipper)

            self.Net.eval()
            # if epoch == self.epochs-1:
            #       self.save_model()
            if (epoch + 1) % 10 == 0:
                if (epoch + 1) % 50 == 0:
                    self.sampling(flag = 1,data='test')
                else:
                    self.sampling(flag = 0,data='test')

        self.t_end = time.time()
        print("Time of training-{}".format((self.t_end - self.t_begin)))

    def detach_np(self, x):
        return x.cpu().detach().numpy()


    def test(self):
        self.load_model()
        self.Net.eval()
        self.best_coherence = 999
        self.best_score=999
        self.evaluate()
        self.sampling(flag = 1,data='test')
        #self.show_M()
        #self.save_top_word()




def main(mode='Train',
         dataset="20news",
         max_topic_num=300,
         emb_type="glove",
         **kwargs):

    data_path = f"./data/{dataset}"
    reader = TextReader(data_path)
    print(emb_type)
    if emb_type == "bert":
        bert_emb_path = f"./emb/bert.npy"
        embedding_mat = utils.build_bert_embedding(bert_emb_path, reader.vocab,
                                                   data_path)
    elif emb_type == "glove":
        emb_path = f"./emb/glove.6B.300d.txt"
        embedding_mat = utils.build_embedding(emb_path, reader.vocab,
                                              data_path)[:-1]
    else:
        embedding_mat = None
    name=''
    model_path = f'./model/{dataset}_{max_topic_num}_{reader.vocab_size}'+name
    Adj_Graph_0 = sym_conditional_prob(reader, data_type="train+valid",adj_alpha=adj_alpha_0)
    model = AMM_no_dag(reader, max_topic_num, model_path, embedding_mat,Adj_Graph=[Adj_Graph_0,Adj_Graph_0,Adj_Graph_0],**kwargs)



    if mode == 'Train':
        model.train()
    elif mode == 'Test':
        model.test()
    else:
        print(f'Unknowned mode {mode}!')

def sym_conditional_prob(reader,data_type,adj_alpha):
    (
        train_data2,
        train_label2,
        train_text2,
    ) = reader.get_sparse_matrix(data_type, mode="tfidf")  # 可以从reader内修改mode，改为tfidf
    ALL_data2 = train_data2.toarray()
    ALL_data2 = torch.from_numpy(ALL_data2).to(device)
    y=ALL_data2

    zero=torch.zeros_like(y)
    y = torch.where(y <adj_alpha, zero, y)
    one=torch.ones_like(y)
    y = torch.where(y >=adj_alpha, one, y)  #对于邻接矩阵的构造
    A = torch.matmul(y.t(), y)   #邻接矩阵
    D = torch.sum(y.t(), dim=1, keepdim=True)
    A_hat = A+ torch.eye(A.size(0), dtype=A.dtype,device=device)
    d=torch.pow(D+1, -1 / 2)
    d = torch.diag_embed(d.squeeze())
    lap_matrix=d@A_hat@d
    return lap_matrix


if __name__ == '__main__':
    config = yaml.load(open('config.yaml'), yaml.FullLoader)
    if config['para']['dataset']=="nips":
        best_coh=0.147
        best_TU=0.719
    elif config['para']['dataset']=="20news":
        best_coh = 0.307
        best_TU = 0.811
    elif config['para']['dataset']=="wiki":
        best_coh = 0.255
        best_TU = 0.797
    main(mode="Train", **config["para"])
    #main(mode="Test", **config["para"])
