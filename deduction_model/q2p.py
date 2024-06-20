import json

import torch
from torch import nn
from torch.utils.data import DataLoader

from dataloader import TestDataset, ValidDataset, TrainDataset, SingledirectionalOneShotIterator
from model import IterativeModel, LabelSmoothingLoss

import math
from session_model import SRGNNRec, AttentionMixerRec, GRURec, TransRec
import pickle
import numpy as np

class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class SelfAttention(nn.Module):

    def __init__(self, hidden_size):
        super(SelfAttention, self).__init__()

        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)

        self.hidden_size = hidden_size

    def forward(self, particles):
        # [batch_size, num_particles, embedding_size]
        K = self.query(particles)
        V = self.query(particles)
        Q = self.query(particles)

        # [batch_size, num_particles, num_particles]
        attention_scores = torch.matmul(Q, K.permute(0, 2, 1))
        attention_scores = attention_scores / math.sqrt(self.hidden_size)

        # [batch_size, num_particles, num_particles]
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # attention_probs = self.dropout(attention_probs)

        # [batch_size, num_particles, embedding_size]
        attention_output = torch.matmul(attention_probs, V)

        return attention_output


class FFN(nn.Module):
    """
    Actually without the FFN layer, there is no non-linearity involved. That is may be why the model cannot fit
    the training queries so well
    """

    def __init__(self, hidden_size, dropout):
        super(FFN, self).__init__()
        self.linear1 = nn.Linear(hidden_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)

        self.activation = nn.GELU()
        self.dropout = dropout

    def forward(self, particles):
        return self.linear2(self.dropout(self.activation(self.linear1(self.dropout(particles)))))


class ParticleCrusher(nn.Module):

    def __init__(self, embedding_size, num_particles):
        super(ParticleCrusher, self).__init__()

        # self.noise_layer = nn.Linear(embedding_size, embedding_size)
        self.num_particles = num_particles

        self.off_sets = nn.Parameter(torch.zeros([1, num_particles, embedding_size]), requires_grad=True)
        # self.layer_norm = LayerNorm(embedding_size)

    def forward(self, batch_of_embeddings):
        # shape of batch_of_embeddings: [batch_size, embedding_size]
        # the return is a tuple ([batch_size, embedding_size, num_particles], [batch_size, num_particles])
        # The first return is the batch of particles for each entity, the second is the weights of the particles
        # Use gaussian kernel to do this

        batch_size, embedding_size = batch_of_embeddings.shape

        # [batch_size, num_particles, embedding_size]
        expanded_batch_of_embeddings = batch_of_embeddings.reshape(batch_size, -1, embedding_size) + self.off_sets

        return expanded_batch_of_embeddings


class Q2P(IterativeModel):
    def __init__(self, num_entities, num_relations, embedding_size, num_particles=3, label_smoothing=0.1,
                 dropout_rate=0.1, session_encoder="SRGNN", pretrained_weights=None):
        super(Q2P, self).__init__(num_entities, num_relations, embedding_size, session_encoder)

        embedding_size = embedding_size // num_particles

        if pretrained_weights is not None:

            print(len(pretrained_weights), num_entities)
            assert len(pretrained_weights) == num_entities
            assert len(pretrained_weights[0]) == embedding_size

            zero_array = np.random.randn(1, embedding_size)
            
            pretrained_weights = np.concatenate((zero_array, pretrained_weights), axis=0)
            pretrained_weights = torch.tensor(pretrained_weights, dtype=torch.float32)
            self.entity_embedding = nn.Embedding.from_pretrained(pretrained_weights, freeze=False)
        else:
            self.entity_embedding = nn.Embedding(num_entities + 1, embedding_size)

        self.relation_embedding = nn.Embedding(num_relations, embedding_size)
        self.num_particles = num_particles

        self.dropout = nn.Dropout(dropout_rate)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()

        embedding_weights = self.entity_embedding.weight
        self.decoder = nn.Linear(embedding_size,
                                 num_entities,
                                 bias=False)
        self.decoder.weight = embedding_weights

        self.label_smoothing_loss = LabelSmoothingLoss(smoothing=label_smoothing)

        # Crusher
        self.to_particles = ParticleCrusher(embedding_size, num_particles)

        # Projection weights
        self.projection_layer_norm_1 = LayerNorm(embedding_size)
        self.projection_layer_norm_2 = LayerNorm(embedding_size)

        self.projection_self_attn = SelfAttention(embedding_size)

        self.projection_Wz = nn.Linear(embedding_size, embedding_size)
        self.projection_Uz = nn.Linear(embedding_size, embedding_size)

        self.projection_Wr = nn.Linear(embedding_size, embedding_size)
        self.projection_Ur = nn.Linear(embedding_size, embedding_size)

        self.projection_Wh = nn.Linear(embedding_size, embedding_size)
        self.projection_Uh = nn.Linear(embedding_size, embedding_size)

        # Higher Order Projection weights
        self.high_projection_attn = SelfAttention(embedding_size)
        self.high_projection_ffn = FFN(embedding_size, self.dropout)

        self.high_projection_layer_norm_1 = LayerNorm(embedding_size)
        self.high_projection_layer_norm_2 = LayerNorm(embedding_size)

        # Intersection weights
        self.intersection_attn = SelfAttention(embedding_size)
        self.intersection_ffn = FFN(embedding_size, self.dropout)

        self.intersection_layer_norm_1 = LayerNorm(embedding_size)
        self.intersection_layer_norm_2 = LayerNorm(embedding_size)

        self.intersection_layer_norm_3 = LayerNorm(embedding_size)
        self.intersection_layer_norm_4 = LayerNorm(embedding_size)

        # Complement weights
        self.complement_attn = SelfAttention(embedding_size)
        self.complement_ffn = FFN(embedding_size, self.dropout)

        self.complement_layer_norm_1 = LayerNorm(embedding_size)
        self.complement_layer_norm_2 = LayerNorm(embedding_size)

        self.complement_layer_norm_3 = LayerNorm(embedding_size)
        self.complement_layer_norm_4 = LayerNorm(embedding_size)



        # Session Encoders 
        assert session_encoder in ["SRGNN", "AttnMixer", "GRURec", "TransRec"]

        session_encoder_config = {'node_count': num_entities, 'embedding_size': embedding_size, 'step': 3}

        if session_encoder == "SRGNN":
            self.session_encoder = SRGNNRec(session_encoder_config, self.entity_embedding)
        
        elif session_encoder == "AttnMixer":
            self.session_encoder = AttentionMixerRec(session_encoder_config, self.entity_embedding)
        
        elif session_encoder == "GRURec":
            self.session_encoder = GRURec(session_encoder_config, self.entity_embedding)
        
        elif session_encoder == "TransRec":
            self.session_encoder = TransRec(session_encoder_config, self.entity_embedding)
        
       
    def scoring(self, query_encoding):
        """

        :param query_encoding: [batch_size, num_particles, embedding_size]
        :return: [batch_size, num_entities]
        """
        query_scores = self.decoder(query_encoding)

        # [batch_size, num_entities]
        prediction_scores, _ = query_scores.max(dim=1)

        return prediction_scores

    def selected_scoring(self, query_encoding, selected_entities):
        """

        :param query_encoding: [num_particles, embedding_size]
        :param selected_entities: [num_entities]
        :return: [ num_particles]
        """
        
        # [batch_size, num_entities, embedding_size]
        entity_embeddings = self.entity_embedding(selected_entities)

        # [batch_size, num_particles, num_entities]
        query_scores = torch.matmul(query_encoding, entity_embeddings.transpose(-2, -1))

        prediction_scores = query_scores.max(dim=1)[0]

        return prediction_scores

    def loss_fnt(self, query_encoding, labels):
        # The size of the query_encoding is [batch_size, num_particles, embedding_size]
        # and the labels are [batch_size]

        # [batch_size, num_entities]
        query_scores = self.scoring(query_encoding)

        # [batch_size]
        labels = torch.tensor(labels)
        labels = labels.to(self.entity_embedding.weight.device)

        _loss = self.label_smoothing_loss(query_scores, labels)

        return _loss

    def projection(self, relation_ids, sub_query_encoding):
        """
        The relational projection of GQE. To fairly evaluate results, we use the same size of relation and use
        TransE embedding structure.

        :param relation_ids: [batch_size]
        :param sub_query_encoding: [batch_size, num_particles, embedding_size]
        :return: [batch_size, num_particles, embedding_size]
        """

        Wz = self.projection_Wz
        Uz = self.projection_Uz

        Wr = self.projection_Wr
        Ur = self.projection_Ur

        Wh = self.projection_Wh
        Uh = self.projection_Uh

        # [batch_size, embedding_size]
        relation_ids = torch.tensor(relation_ids)
        relation_ids = relation_ids.to(self.relation_embedding.weight.device)
        relation_embeddings = self.relation_embedding(relation_ids)

        #  [batch_size, 1, embedding_size]
        relation_transition = torch.unsqueeze(relation_embeddings, 1)

        #  [batch_size, num_particles, embedding_size]
        projected_particles = sub_query_encoding

        z = self.sigmoid(Wz(self.dropout(relation_transition)) + Uz(self.dropout(projected_particles)))
        r = self.sigmoid(Wr(self.dropout(relation_transition)) + Ur(self.dropout(projected_particles)))

        h_hat = self.tanh(Wh(self.dropout(relation_transition)) + Uh(self.dropout(projected_particles * r)))

        h = (1 - z) * sub_query_encoding + z * h_hat

        projected_particles = h
        projected_particles = self.projection_layer_norm_1(projected_particles)

        projected_particles = self.projection_self_attn(self.dropout(projected_particles))
        projected_particles = self.projection_layer_norm_2(projected_particles)

        return projected_particles

    def higher_projection(self, relation_ids, sub_query_encoding):
        particles = self.high_projection_attn(sub_query_encoding)
        particles = self.high_projection_layer_norm_1(particles)

        particles = self.high_projection_ffn(particles) + particles
        particles = self.high_projection_layer_norm_2(particles)

        return self.projection(relation_ids, particles)

    def intersection(self, sub_query_encoding_list):
        """

        :param sub_query_encoding_list: a list of the sub-query embeddings of size [batch_size, num_particles, embedding_size]
        :return:  [batch_size, num_particles, embedding_size]
        """
        


        # [batch_size, number_sub_queries * num_particles, embedding_size]

        all_subquery_encodings = torch.cat(sub_query_encoding_list, dim=1)

        num_sets = len(sub_query_encoding_list)

        """
                :param particles_sets: [batch_size, num_sets, num_particles, embedding_size]
                :param weights_sets: [batch_size, num_sets, num_particles]
                :return: [batch_size, num_particles, embedding_size]
                """

        batch_size, num_particles, embedding_size = all_subquery_encodings.shape

        # [batch_size, num_sets * num_particles, embedding_size]
        flatten_particles = all_subquery_encodings.view(batch_size, -1, embedding_size)

        # [batch_size, num_sets * num_particles, embedding_size]
        flatten_particles = self.intersection_attn(self.dropout(flatten_particles))
        flatten_particles = self.intersection_layer_norm_1(flatten_particles)

        flatten_particles = self.intersection_ffn(flatten_particles) + flatten_particles
        flatten_particles = self.intersection_layer_norm_2(flatten_particles)

        flatten_particles = self.intersection_attn(self.dropout(flatten_particles))
        flatten_particles = self.intersection_layer_norm_3(flatten_particles)

        flatten_particles = self.intersection_ffn(flatten_particles) + flatten_particles
        flatten_particles = self.intersection_layer_norm_4(flatten_particles)

        # particles = flatten_particles
        particles = flatten_particles[:, num_sets * torch.arange(self.num_particles)]

        return particles

    def negation(self, sub_query_encoding_list):
        # [batch_size, num_particles, embedding_size]
        new_particles = sub_query_encoding_list

        new_particles = self.complement_attn(self.dropout(new_particles))
        new_particles = self.complement_layer_norm_1(new_particles)
        new_particles = self.complement_ffn(new_particles) + new_particles
        new_particles = self.complement_layer_norm_2(new_particles)

        new_particles = self.complement_attn(self.dropout(new_particles))
        new_particles = self.complement_layer_norm_3(new_particles)
        new_particles = self.complement_ffn(new_particles) + new_particles
        new_particles = self.complement_layer_norm_4(new_particles)

        return new_particles

    def union(self, sub_query_encoding_list):
        # [batch_size, number_sub_queries * num_particles, embedding_size]
        all_subquery_encodings = torch.cat(sub_query_encoding_list, dim=1)

        return all_subquery_encodings

    def forward(self, batched_structured_query, label=None):
        assert batched_structured_query[0] in ["p", "e", "i", "u", "n", "s"]

        if batched_structured_query[0] == "p":

            sub_query_result = self.forward(batched_structured_query[2])
            if batched_structured_query[2][0] == 'e':
                this_query_result = self.projection(batched_structured_query[1], sub_query_result)

            else:
                this_query_result = self.higher_projection(batched_structured_query[1], sub_query_result)

        elif batched_structured_query[0] == "i":
            sub_query_result_list = []
            for _i in range(1, len(batched_structured_query)):
                sub_query_result = self.forward(batched_structured_query[_i])
                sub_query_result_list.append(sub_query_result)

            this_query_result = self.intersection(sub_query_result_list)

        elif batched_structured_query[0] == "u":
            sub_query_result_list = []
            for _i in range(1, len(batched_structured_query)):
                sub_query_result = self.forward(batched_structured_query[_i])
                sub_query_result_list.append(sub_query_result)

            this_query_result = self.union(sub_query_result_list)

        elif batched_structured_query[0] == "n":
            sub_query_result = self.forward(batched_structured_query[1])
            this_query_result = self.negation(sub_query_result)

        elif batched_structured_query[0] == "e":

            entity_ids = torch.tensor(batched_structured_query[1])
            entity_ids = entity_ids.to(self.entity_embedding.weight.device)
            this_query_result = self.to_particles(self.entity_embedding(entity_ids))
        
        elif batched_structured_query[0] == "s":
            entity_ids = torch.tensor(batched_structured_query[1])
            entity_ids = entity_ids.to(self.entity_embedding.weight.device)
            sequence_length = (entity_ids != self.num_entities).sum(dim=1)
    
            this_query_result = self.session_encoder(entity_ids, sequence_length)

            this_query_result = self.to_particles(this_query_result)

        else:
            this_query_result = None

        if label is None:
            return this_query_result

        else:
            return self.loss_fnt(this_query_result, label)


if __name__ == "__main__":

    train_data_path = "../sampled_hyper_train_merged/amazon_train_queries_0.json"
    valid_data_path = "../sampled_hyper_valid/amazon_valid_queries_0.json"
    test_data_path = "../sampled_hyper_test/amazon_test_queries_0.json"

    with open(train_data_path, "r") as fin:
        train_data_dict = json.load(fin)

    with open(valid_data_path, "r") as fin:
        valid_data_dict = json.load(fin)

    with open(test_data_path, "r") as fin:
        test_data_dict = json.load(fin)

    data_dir = "../hyper_graph_data_en"

    asin_ids = json.load(open(data_dir + "/asin_ids.json", "r"))
    value_ids = json.load(open(data_dir + "/value_ids.json", "r"))
    session_dict = json.load(open(data_dir + "/session_dict.json", "r"))

    id2node = json.load(open(data_dir + "/id2node.json", "r"))
    node2id = json.load(open(data_dir + "/node2id.json", "r"))
    id2relation = json.load(open(data_dir + "/id2relation.json", "r"))
    relation2id = json.load(open(data_dir + "/relation2id.json", "r"))
    

    nentity = len(id2node)
    nrelation = len(id2relation)
    max_length = max([len(_) for _ in session_dict.values()])


    print("nentity: ", nentity)
    print("nrelation: ", nrelation)
    print("max_length: ", max_length)

    batch_size = 512 // 4
    embedding_size = 384

    print("batch_size: ", batch_size)
    print("embedding_size: ", embedding_size)

    # pretrained_weights = pickle.load(open(data_dir + "/nodeid_to_semantic_embeddings.pkl", "rb"))
    pretrained_weights = None


    for session_encoder in [ "SRGNN", "GRURec", "AttnMixer", "TransRec"]:

    # for session_encoder in [ "TransRec"]:

        q2p_model = Q2P(num_entities=nentity, num_relations=nrelation, embedding_size=embedding_size, 
                        session_encoder=session_encoder, 
                        pretrained_weights=pretrained_weights)
        
        if torch.cuda.is_available():
            q2p_model = q2p_model.cuda()

        
        train_iterators = {}
        for query_type, query_answer_dict in train_data_dict.items():
            print("====================================")
            print(query_type)

            new_iterator = SingledirectionalOneShotIterator(DataLoader(
                TrainDataset(nentity, nrelation, query_answer_dict, max_length),
                batch_size=batch_size,
                shuffle=True,
                collate_fn=TrainDataset.collate_fn
            ))
            train_iterators[query_type] = new_iterator

        for key, iterator in train_iterators.items():
            print("read ", key)
            batched_query, unified_ids, positive_sample = next(iterator)
            print(batched_query)
            print(unified_ids)
            print(positive_sample)

            query_embedding = q2p_model(batched_query)
            print(query_embedding.shape)
            loss = q2p_model(batched_query, positive_sample)
            print(loss)

        validation_loaders = {}
        for query_type, query_answer_dict in valid_data_dict.items():
            print("====================================")

            print(query_type)

            new_iterator = DataLoader(
                ValidDataset(nentity, nrelation, query_answer_dict, max_length),
                batch_size=batch_size,
                shuffle=True,
                collate_fn=ValidDataset.collate_fn
            )
            validation_loaders[query_type] = new_iterator

        for key, loader in validation_loaders.items():
            print("read ", key)
            for batched_query, unified_ids, train_answers, valid_answers in loader:
                print(batched_query)
                print(unified_ids)
                print([len(_) for _ in train_answers])
                print([len(_) for _ in valid_answers])

                query_embedding = q2p_model(batched_query)
                result_logs = q2p_model.evaluate_entailment(query_embedding, train_answers)
                print(result_logs)

                result_logs = q2p_model.evaluate_generalization(query_embedding, train_answers, valid_answers)
                print(result_logs)

                break

        test_loaders = {}
        for query_type, query_answer_dict in test_data_dict.items():
            print("====================================")
            print(query_type)

            new_loader = DataLoader(
                TestDataset(nentity, nrelation, query_answer_dict, max_length),
                batch_size=batch_size,
                shuffle=True,
                collate_fn=TestDataset.collate_fn
            )

            test_loaders[query_type] = new_loader

        for key, loader in test_loaders.items():
            print("read ", key)
            for batched_query, unified_ids, train_answers, valid_answers, test_answers in loader:
                print(batched_query)
                print(unified_ids)

                query_embedding = q2p_model(batched_query)
                result_logs = q2p_model.evaluate_entailment(query_embedding, train_answers)
                print(result_logs)

                result_logs = q2p_model.evaluate_generalization(query_embedding, valid_answers, test_answers)
                print(result_logs)

                print(train_answers[0])
                print([len(_) for _ in train_answers])
                print([len(_) for _ in valid_answers])
                print([len(_) for _ in test_answers])

                break
    print("done")

    