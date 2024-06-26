import torch
import math

import numpy as np
import torch
from torch import nn
from torch.nn import Parameter
from torch.nn import functional as F

class SessionModel(torch.nn.Module):

    def __init__(self, config, item_embeddings):
        super().__init__()
        self.config = config

        # The last item is the padding item for short sessions]
        
        self.embedding = item_embeddings

    

# Adopted from https://github.com/RUCAIBox/RecBole/blob/master/recbole/model/sequential_recommender/srgnn.py
class GNN(nn.Module):
    """Graph neural networks are well-suited for session-based recommendation,
    because it can automatically extract features of session graphs with considerations of rich node connections.
    """

    def __init__(self, embedding_size, step=1):
        super(GNN, self).__init__()
        self.step = step
        self.embedding_size = embedding_size
        self.input_size = embedding_size * 2
        self.gate_size = embedding_size * 3
        self.w_ih = Parameter(torch.Tensor(self.gate_size, self.input_size))
        self.w_hh = Parameter(torch.Tensor(self.gate_size, self.embedding_size))
        self.b_ih = Parameter(torch.Tensor(self.gate_size))
        self.b_hh = Parameter(torch.Tensor(self.gate_size))
        self.b_iah = Parameter(torch.Tensor(self.embedding_size))
        self.b_ioh = Parameter(torch.Tensor(self.embedding_size))

        self.linear_edge_in = nn.Linear(
            self.embedding_size, self.embedding_size, bias=True
        )
        self.linear_edge_out = nn.Linear(
            self.embedding_size, self.embedding_size, bias=True
        )

    def GNNCell(self, A, hidden):
        r"""Obtain latent vectors of nodes via graph neural networks.

        Args:
            A(torch.FloatTensor):The connection matrix,shape of [batch_size, max_session_len, 2 * max_session_len]

            hidden(torch.FloatTensor):The item node embedding matrix, shape of
                [batch_size, max_session_len, embedding_size]

        Returns:
            torch.FloatTensor: Latent vectors of nodes,shape of [batch_size, max_session_len, embedding_size]

        """

        input_in = (
            torch.matmul(A[:, :, : A.size(1)], self.linear_edge_in(hidden)) + self.b_iah
        )
        input_out = (
            torch.matmul(
                A[:, :, A.size(1) : 2 * A.size(1)], self.linear_edge_out(hidden)
            )
            + self.b_ioh
        )
        # [batch_size, max_session_len, embedding_size * 2]
        inputs = torch.cat([input_in, input_out], 2)

        # gi.size equals to gh.size, shape of [batch_size, max_session_len, embedding_size * 3]
        gi = F.linear(inputs, self.w_ih, self.b_ih)
        gh = F.linear(hidden, self.w_hh, self.b_hh)
        # (batch_size, max_session_len, embedding_size)
        i_r, i_i, i_n = gi.chunk(3, 2)
        h_r, h_i, h_n = gh.chunk(3, 2)
        reset_gate = torch.sigmoid(i_r + h_r)
        input_gate = torch.sigmoid(i_i + h_i)
        new_gate = torch.tanh(i_n + reset_gate * h_n)
        hy = (1 - input_gate) * hidden + input_gate * new_gate
        return hy

    def forward(self, A, hidden):
        for i in range(self.step):
            hidden = self.GNNCell(A, hidden)
        return hidden


class SRGNNRec(SessionModel):
    r"""SRGNN regards the conversation history as a directed graph.
    In addition to considering the connection between the item and the adjacent item,
    it also considers the connection with other interactive items.

    Such as: A example of a session sequence(eg:item1, item2, item3, item2, item4) and the connection matrix A

    Outgoing edges:
        === ===== ===== ===== =====
         \    1     2     3     4
        === ===== ===== ===== =====
         1    0     1     0     0
         2    0     0    1/2   1/2
         3    0     1     0     0
         4    0     0     0     0
        === ===== ===== ===== =====

    Incoming edges:
        === ===== ===== ===== =====
         \    1     2     3     4
        === ===== ===== ===== =====
         1    0     0     0     0
         2   1/2    0    1/2    0
         3    0     1     0     0
         4    0     1     0     0
        === ===== ===== ===== =====
    """

    def __init__(self, config, item_embedding):
        super(SRGNNRec, self).__init__(config, item_embedding)

        # load parameters info
        self.embedding_size = config["embedding_size"]
        self.step = config["step"]
    

        # define layers and loss
        self.gnn = GNN(self.embedding_size, self.step)
        self.linear_one = nn.Linear(self.embedding_size, self.embedding_size, bias=True)
        self.linear_two = nn.Linear(self.embedding_size, self.embedding_size, bias=True)
        self.linear_three = nn.Linear(self.embedding_size, 1, bias=False)
        self.linear_transform = nn.Linear(
            self.embedding_size * 2, self.embedding_size, bias=True
        )
        
        # parameters initialization
        self._reset_parameters()

    def _reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.embedding_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def _get_slice(self, item_seq):

        # Mask matrix, shape of [batch_size, max_session_len]
        mask = item_seq.lt(self.config['node_count'])
        items, n_node, A, alias_inputs = [], [], [], []
        max_n_node = item_seq.size(1)
        item_seq = item_seq.cpu().numpy()
        for u_input in item_seq:
            node = np.unique(u_input)
            items.append(node.tolist() + (max_n_node - len(node)) * [0])
            u_A = np.zeros((max_n_node, max_n_node))

            for i in np.arange(len(u_input) - 1):
                if u_input[i + 1] == 0:
                    break

                u = np.where(node == u_input[i])[0][0]
                v = np.where(node == u_input[i + 1])[0][0]
                u_A[u][v] = 1

            u_sum_in = np.sum(u_A, 0)
            u_sum_in[np.where(u_sum_in == 0)] = 1
            u_A_in = np.divide(u_A, u_sum_in)
            u_sum_out = np.sum(u_A, 1)
            u_sum_out[np.where(u_sum_out == 0)] = 1
            u_A_out = np.divide(u_A.transpose(), u_sum_out)
            u_A = np.concatenate([u_A_in, u_A_out]).transpose()
            A.append(u_A)

            alias_inputs.append([np.where(node == i)[0][0] for i in u_input])
            
        # The relative coordinates of the item node, shape of [batch_size, max_session_len]
        alias_inputs = torch.LongTensor(alias_inputs).to(mask.device)
        # The connecting matrix, shape of [batch_size, max_session_len, 2 * max_session_len]
        A = torch.FloatTensor(np.array(A)).to(mask.device)
        # The unique item nodes, shape of [batch_size, max_session_len]
        items = torch.LongTensor(items).to(mask.device)

        return alias_inputs, A, items, mask

    def gather_indexes(self, output, gather_index):
        """Gathers the vectors at the specific positions over a minibatch"""
        gather_index = gather_index.view(-1, 1, 1).expand(-1, -1, output.shape[-1])
        output_tensor = output.gather(dim=1, index=gather_index)
        return output_tensor.squeeze(1)

    def forward(self, item_seq, item_seq_len):

        alias_inputs, A, items, mask = self._get_slice(item_seq)

    
        hidden = self.embedding(items)
     
        hidden = self.gnn(A, hidden)

        alias_inputs = alias_inputs.view(-1, alias_inputs.size(1), 1).expand(
            -1, -1, self.embedding_size
        )
        seq_hidden = torch.gather(hidden, dim=1, index=alias_inputs)

        # fetch the last hidden state of last timestamp
        ht = self.gather_indexes(seq_hidden, item_seq_len - 1)
        q1 = self.linear_one(ht).view(ht.size(0), 1, ht.size(1))
        q2 = self.linear_two(seq_hidden)

        alpha = self.linear_three(torch.sigmoid(q1 + q2))
        a = torch.sum(alpha * seq_hidden * mask.view(mask.size(0), -1, 1).float(), 1)
        seq_output = self.linear_transform(torch.cat([a, ht], dim=1))
        return seq_output


class AttentionMixerRec(SessionModel):
    # replication of according to paper Efficiently Leveraging Multi-level User Intent for Session-based 
    # Recommendation via Atten-Mixer Network  https://arxiv.org/pdf/2206.12781.pdf

    def __init__(self, config, item_embedding, L=5, H=4):
        super().__init__(config, item_embedding)
        self.config = config
        self.embedding = item_embedding

        self.L = L

        self.linear_layers = nn.ModuleList([nn.Linear(config['embedding_size'], config['embedding_size'], bias=False) for _ in range(L)])

        self.WQs = nn.ModuleList([nn.Linear(config['embedding_size'], config['embedding_size'], bias=False) for _ in range(H)])
        self.WKs = nn.ModuleList([nn.Linear(config['embedding_size'], config['embedding_size'], bias=False) for _ in range(H)])

        self.WQ = nn.Linear(config['embedding_size'], config['embedding_size'], bias=False)
        self.WK = nn.Linear(config['embedding_size'], config['embedding_size'], bias=False)
        
    def forward(self, item_seq, item_seq_len):
       
        batch_size, max_session_len = item_seq.shape
        

        masks = []
        M = torch.zeros(batch_size, max_session_len).to(item_seq.device)

        for i in range(self.L):
            positions = torch.maximum(item_seq_len - 1 -i, torch.zeros_like(item_seq_len - 1 -i)).long()
            # print("positions:", positions)
            M = (M + F.one_hot(positions, max_session_len).float()) > 0
            masks.append(M.clone())
           


        # [Batch_size, L, max_session_len]
        concated_masks = torch.stack(masks, dim=1)
        # print("concated_masks:", concated_masks)

        # [Batch_size,  max_session_len, embedding_size]
        item_embeddings = self.embedding(item_seq)

        # [Batch_size, max_session_len, embedding_size]
        # print("item_embeddings:", item_embeddings.shape)
        concated_item_embeddings = item_embeddings

        # [Batch_size, L, embedding_size]
        # print("concated_masks", concated_masks.shape)
        attention_query_embeddings = torch.bmm(concated_masks.float(), concated_item_embeddings)
        # print("attention_query_embeddings:", attention_query_embeddings)


        # [Batch_size, L, embedding_size]
        attention_query_embeddings_list = []
        for i in range(self.L):
            attention_query_embeddings_list.append(self.linear_layers[i](attention_query_embeddings[:, i, :]))

        attention_query_embeddings = torch.stack(attention_query_embeddings_list, dim=1)
        # print("attention_query_embeddings:", attention_query_embeddings.shape)

        # ([Batch_size, L, embedding_size] * H)
        Qs = [WQ(attention_query_embeddings) for WQ in self.WQs]

        # [Batch_size, H, L, embedding_size] 
        Qs = torch.stack(Qs, dim=1)
        batch_size, H, L, embedding_size = Qs.shape
        Qs = Qs.reshape(batch_size * H, L, embedding_size)
        
        # [Batch_size, max_session_len, embedding_size * H]
        Ks = [WK(concated_item_embeddings) for WK in self.WKs]

        # [Batch_size, H, max_session_len, embedding_size]
        Ks = torch.stack(Ks, dim=1)
        batch_size, H, max_session_len, embedding_size = Ks.shape
        Ks = Ks.reshape(batch_size * H, max_session_len, embedding_size)


        # [Batch_size * H, L, max_session_len]
        attention_matrix = F.softmax(torch.bmm(Qs, Ks.transpose(1, 2)) / math.sqrt(self.config['embedding_size']), dim=2)
        # print("attention_matrix:", attention_matrix.shape)

        pooled_attention_matrix = torch.norm(attention_matrix, p=4, dim=1)

        # print("pooled_attention_matrix:", pooled_attention_matrix.shape)

        # [Batch_size, H, max_session_len]
        pooled_attention_matrix = pooled_attention_matrix.reshape(batch_size, H, max_session_len)

        # [Batch_size, max_session_len]
        pooled_attention_matrix = torch.mean(pooled_attention_matrix, dim=1)

        # [Batch_size, embedding_size]
        attention_result = (pooled_attention_matrix.unsqueeze(-1) * concated_item_embeddings).sum(dim=1)

        return attention_result


class GRURec(SessionModel):

    def __init__(self, config, item_embedding):
        super().__init__(config, item_embedding)
        self.config = config
        
        self.gru = nn.GRU(input_size=config['embedding_size'], hidden_size=config['embedding_size'], num_layers=3, batch_first=True, bidirectional=True)

    def forward(self, item_seq, item_seq_len):
        item_embeddings = self.embedding(item_seq)
        output, hidden = self.gru(item_embeddings)
        return hidden[0]



class TransRec(SessionModel):
    def __init__(self, config, item_embeddings):
        super().__init__(config, item_embeddings)
        self.config = config

        self.transformer = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=config['embedding_size'], nhead=4), num_layers=2)
    
    def forward(self, item_seq, item_seq_len):

        item_embeddings = self.embedding(item_seq)
        output = self.transformer(item_embeddings)

        # print("output:", output.shape)
        return output[:, 0, :]

if __name__ == "__main__":

    config = {'node_count': 100, 'embedding_size': 32, 'step': 3}


    item_embedding = torch.nn.Embedding(config['node_count'] + 1, config['embedding_size'], padding_idx=config['node_count'])
        
    item_embedding.cuda()
    model = SRGNNRec(config, item_embedding)
    model.cuda()
    session = torch.LongTensor([[1, 2, 3],
                                [1, 2, 100],
                                [1, 2, 1]]).cuda()
    
    session_length = torch.LongTensor([2, 1, 3]).cuda()
    session_embedding = model(session, session_length)

    print("SRGNNRec output:", session_embedding.shape)


    model = AttentionMixerRec(config, item_embedding)
    model.cuda()
    session = torch.LongTensor([[1, 2, 100],
                                [1, 100, 100],
                                [1, 2, 1]]).cuda()
    
    session_length = torch.LongTensor([2, 1, 3]).cuda()
    session_embedding = model(session, session_length)

    print("AttentionMixerRec output:", session_embedding.shape)

    model = GRURec(config, item_embedding)
    model.cuda()

    session = torch.LongTensor([[1, 2, 100],
                                [1, 100, 100],
                                [1, 2, 1]]).cuda()
    
    session_length = torch.LongTensor([2, 1, 3]).cuda()
    session_embedding = model(session, session_length)

    print("GRURec output:", session_embedding.shape)

   
    model = TransRec(config, item_embedding)
    model.cuda()

    session = torch.LongTensor([[1, 2, 100],
                                [1, 100, 100],
                                [1, 2, 1]]).cuda()
    
    session_length = torch.LongTensor([2, 1, 3]).cuda()
    session_embedding = model(session, session_length)

    print("TransRec output:", session_embedding.shape)










