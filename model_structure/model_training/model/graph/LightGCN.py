from model_structure.model_training.base.graph_recommender import GraphRecommender
from model_structure.model_training.util.sampler import next_batch_pairwise
from model_structure.model_training.base.torch_interface import TorchGraphInterface
from model_structure.model_training.util.loss_torch import bpr_loss, l2_reg_loss
from model_structure.model_training.paras import args
# paper: LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation. SIGIR'20

import torch

torch.manual_seed(1024)

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle


def CL_loss(id_emb_user, id_emb_item, hard_negative_proj, temperature=0.2):
    # 对比学习损失
    positive_similarity = F.cosine_similarity(id_emb_user, id_emb_item)
    negative_similarity = F.cosine_similarity(id_emb_user, hard_negative_proj)

    # 损失函数计算
    loss = - torch.log(torch.exp(positive_similarity / temperature) /
                       (torch.exp(positive_similarity / temperature) + torch.exp(negative_similarity / temperature)))
    return loss.mean()


class LightGCN(GraphRecommender):
    def __init__(self, conf, training_set, test_set):
        super(LightGCN, self).__init__(conf, training_set, test_set)
        args = self.config['LightGCN']
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_layers = int(args['n_layer'])
        self.bpr_loss = bpr_loss
        self.model = LGCN_Encoder(self.data, self.emb_size, self.n_layers)

        # 定义一个神经网络适配器，将嵌入降维到 512
        self.adapter = nn.Sequential(
            nn.Linear(args.input_size, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, args.output_sie),
            nn.ReLU()
        ).to(self.device)

        # 读取和预处理硬负例嵌入
        with open('hard_negative_dict_Yelp_llama3_lora.pkl', 'rb') as f:
            self.hard_negative = pickle.load(f)

    def train(self):
        model = self.model.to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lRate)

        for epoch in range(self.maxEpoch):
            for n, batch in enumerate(next_batch_pairwise(self.data, self.batch_size)):
                user_idx, pos_idx, neg_idx, target_index = batch
                rec_user_emb, rec_item_emb = model()

                # 收集硬负例嵌入并应用适配器
                hard_negative_emb_np = np.empty((len(user_idx), args.inpu_size), dtype=np.float32)

                for i, (user, pos) in enumerate(zip(user_idx, pos_idx)):
                    try:
                        emb = self.hard_negative[user][pos]
                        if len(emb) != args.input_size:
                            if len(emb[0]) == 1:
                                emb = emb[0]
                            else:
                                emb = np.random.randn(args.input_size).astype(np.float32)
                    except (KeyError, IndexError, ValueError):
                        emb = np.random.randn(args.input_size).astype(np.float32)

                    hard_negative_emb_np[i] = emb[0]

                hard_negative_emb = torch.from_numpy(hard_negative_emb_np).to(self.device)
                hard_negative_emb = self.adapter(hard_negative_emb)

                # 动态调整 alpha 值
                alpha = 0.0001  # Toys 0.01

                # 获取正负样本的嵌入
                neg_item_emb = rec_item_emb[neg_idx]
                pos_item_emb = rec_item_emb[pos_idx]
                user_emb = rec_user_emb[user_idx]

                mixed_neg_item_emb = alpha * hard_negative_emb + (1 - alpha) * neg_item_emb

                # 计算对比损失
                pos_sim = torch.sum(user_emb * pos_item_emb, dim=1)
                neg_sim = torch.sum(user_emb * mixed_neg_item_emb, dim=1)

                self.margi = 0.3
                batch_loss = torch.mean(torch.relu(neg_sim - pos_sim + self.margi))
                cl_loss = CL_loss(user_emb, pos_item_emb, hard_negative_emb)
                l2_loss = l2_reg_loss(self.reg, model.embedding_dict['user_emb'][user_idx],
                                      model.embedding_dict['item_emb'][pos_idx], mixed_neg_item_emb) / self.batch_size

                total_loss = batch_loss + l2_loss + 0.001 * cl_loss

                # 优化步骤
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                if n % 100 == 0 and n > 0:
                    print(
                        f'training: epoch {epoch + 1}, batch {n}, batch_loss: {total_loss.item()}, alpha: {alpha:.5f}')

            with torch.no_grad():
                self.user_emb, self.item_emb = self.model()
                self.fast_evaluation(epoch)

        self.user_emb, self.item_emb = self.best_user_emb, self.best_item_emb

    def save(self):
        with torch.no_grad():
            self.best_user_emb, self.best_item_emb = self.model.forward()

    def predict(self, u):
        u = self.data.get_user_id(u)
        score = torch.matmul(self.user_emb[u], self.item_emb.transpose(0, 1))
        return score.cpu().numpy()


class LGCN_Encoder(nn.Module):
    def __init__(self, data, emb_size, n_layers):
        super(LGCN_Encoder, self).__init__()
        device = torch.device("cuda")
        self.data = data
        self.latent_size = emb_size
        self.layers = n_layers
        self.norm_adj = data.norm_adj
        self.embedding_dict = self._init_model()
        self.sparse_norm_adj = TorchGraphInterface.convert_sparse_mat_to_tensor(self.norm_adj).to(device)

    def _init_model(self):
        initializer = nn.init.xavier_uniform_
        embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(torch.empty(self.data.user_num, self.latent_size))),
            'item_emb': nn.Parameter(initializer(torch.empty(self.data.item_num, self.latent_size))),
        })
        return embedding_dict

    def forward(self):
        ego_embeddings = torch.cat([self.embedding_dict['user_emb'], self.embedding_dict['item_emb']], 0)
        all_embeddings = [ego_embeddings]
        for k in range(self.layers):
            ego_embeddings = torch.sparse.mm(self.sparse_norm_adj, ego_embeddings)
            all_embeddings += [ego_embeddings]
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = torch.mean(all_embeddings, dim=1)
        user_all_embeddings = all_embeddings[:self.data.user_num]
        item_all_embeddings = all_embeddings[self.data.user_num:]
        return user_all_embeddings, item_all_embeddings
