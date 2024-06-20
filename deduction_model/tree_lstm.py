import json

import torch
from torch import nn
from torch.utils.data import DataLoader

import dataloader
import model

from model import LabelSmoothingLoss

from session_model import SRGNNRec, AttentionMixerRec, GRURec, TransRec
import numpy as np
import pickle


# from .dataloader import TestDataset, ValidDataset, TrainDataset, SingledirectionalOneShotIterator
# from .model import IterativeModel


class TreeLSTM(model.IterativeModel):
    def __init__(self, num_entities, num_relations, embedding_size, label_smoothing=0.1, use_old_loss=False, session_encoder="SRGNN", pretrained_weights=None):
        super(TreeLSTM, self).__init__(num_entities, num_relations, embedding_size, use_old_loss)

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

        self.operation_embedding = nn.Embedding(3, embedding_size)

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

        self.W_i = nn.Linear(embedding_size, embedding_size)
        self.U_i = nn.Linear(embedding_size, embedding_size)

        self.W_f = nn.Linear(embedding_size, embedding_size)
        self.U_f = nn.Linear(embedding_size, embedding_size)

        self.W_o = nn.Linear(embedding_size, embedding_size)
        self.U_o = nn.Linear(embedding_size, embedding_size)

        self.W_u = nn.Linear(embedding_size, embedding_size)
        self.U_u = nn.Linear(embedding_size, embedding_size)

        self.decoder = nn.Linear(embedding_size, num_entities)
        self.label_smoothing_loss = LabelSmoothingLoss(smoothing=label_smoothing)

        embedding_weights = self.entity_embedding.weight

        self.decoder = nn.Linear(embedding_size,
                                 num_entities,
                                 bias=False)
        self.decoder.weight = embedding_weights


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


    def selected_scoring(self, query_encoding, selected_entities):
        """

        :param query_encoding: [num_particles, embedding_size]
        :param selected_entities: [num_entities]
        :return: [ num_particles]
        """
        
        # [batch_size, num_entities, embedding_size]
        entity_embeddings = self.entity_embedding(selected_entities)

        # [batch_size, num_particles, num_entities]
        if len(query_encoding.shape) == 2:
            query_scores = torch.matmul(query_encoding[0, :], entity_embeddings.transpose(-2, -1))
        
        else:
            query_scores = torch.matmul(query_encoding[:, 0, :], entity_embeddings.transpose(-2, -1))

        return query_scores



    def scoring(self, query_encoding):
        """

        :param query_encoding: [batch_size, embedding_size]
        :return: [batch_size, num_entities]
        """

        # print("query_encoding", query_encoding.shape)
        query_scores = self.decoder(query_encoding[:, 0, :])
        return query_scores


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
        :param sub_query_encoding: [batch_size, embedding_size]
        :return: [batch_size, embedding_size]
        """

        # [batch_size, embedding_size]
        if len(sub_query_encoding.shape) == 2:
            prev_h = sub_query_encoding
            prev_c = torch.zeros_like(prev_h)

        else:
            prev_h = sub_query_encoding[:, 0, :]
            prev_c = sub_query_encoding[:, 1, :]



        relation_ids = torch.tensor(relation_ids)
        relation_ids = relation_ids.to(self.relation_embedding.weight.device)
        x = self.relation_embedding(relation_ids)
        
        i = self.sigmoid(self.W_i(x) + self.U_i(prev_h))
        f = self.sigmoid(self.W_f(x) + self.U_f(prev_h))
        o = self.sigmoid(self.W_o(x) + self.U_o(prev_h))
        u = self.tanh(self.W_u(x) + self.U_u(prev_h))

        next_c = f * prev_c + i * u
        next_h = o * self.tanh(next_c)



        return torch.stack((next_h, next_c), dim=1)

    def higher_projection(self, relation_ids, sub_query_encoding):
        return self.projection(relation_ids, sub_query_encoding)


    def intersection(self, sub_query_encoding_list):
        """

        :param sub_query_encoding_list: a list of the sub-query embeddings of size [batch_size, embedding_size]
        :return:  [batch_size, embedding_size]
        """

        # [batch_size, number_sub_queries, 2, embedding_size]
        all_subquery_encodings = torch.stack(sub_query_encoding_list, dim=1)

        # [batch_size, embedding_size]
        prev_h = all_subquery_encodings[:, :, 0, :].sum(dim=1)

        # [batch_size, number_sub_queries, embedding_size]
        c_k = all_subquery_encodings[:, :, 1, :]

        x = self.operation_embedding(
            torch.zeros(all_subquery_encodings.shape[0]).long().to(self.operation_embedding.weight.device)
        )

        i = self.sigmoid(self.W_i(x) + self.U_i(prev_h))

        # [batch_size, number_sub_queries, embedding_size]
        f_k = self.sigmoid( self.W_f(all_subquery_encodings[:, :, 0, :]) + self.U_f(prev_h).unsqueeze(1))

        o = self.sigmoid(self.W_o(x) + self.U_o(prev_h))

        u = self.tanh(self.W_u(x) + self.U_u(prev_h))


        next_c = torch.sum(f_k * c_k, dim=1) + i * u

        next_h = o * self.tanh(next_c)


        return torch.stack((next_h, next_c), dim=1)

    def negation(self, sub_query_encoding):
        # [batch_size, 2, embedding_size]

        prev_h = sub_query_encoding[:, 0, :]
        prev_c = sub_query_encoding[:, 1, :]

        operation_ids = torch.tensor(2).to(self.operation_embedding.weight.device).unsqueeze(0)
        x = self.operation_embedding(operation_ids)

        
        i = self.sigmoid(self.W_i(x) + self.U_i(prev_h))
        f = self.sigmoid(self.W_f(x) + self.U_f(prev_h))
        o = self.sigmoid(self.W_o(x) + self.U_o(prev_h))
        u = self.tanh(self.W_u(x) + self.U_u(prev_h))

        next_c = f * prev_c + i * u
        next_h = o * self.tanh(next_c)

        return torch.stack((next_h, next_c), dim=1)

    def union(self, sub_query_encoding_list):
        """

        :param sub_query_encoding_list: a list of the sub-query embeddings of size [batch_size, embedding_size]
        :return:  [batch_size, embedding_size]
        """

        # [batch_size, number_sub_queries, 2, embedding_size]
        all_subquery_encodings = torch.stack(sub_query_encoding_list, dim=1)

        # [batch_size, embedding_size]
        prev_h = all_subquery_encodings[:, :, 0, :].sum(dim=1)

        # [batch_size, number_sub_queries, embedding_size]
        c_k = all_subquery_encodings[:, :, 1, :]

        x = self.operation_embedding(
            torch.ones(all_subquery_encodings.shape[0]).long().to(self.operation_embedding.weight.device)
        )

        i = self.sigmoid(self.W_i(x) + self.U_i(prev_h))

        # [batch_size, number_sub_queries, embedding_size]
        f_k = self.sigmoid( self.W_f(all_subquery_encodings[:, :, 0, :]) + self.U_f(prev_h).unsqueeze(1))

        o = self.sigmoid(self.W_o(x) + self.U_o(prev_h))

        u = self.tanh(self.W_u(x) + self.U_u(prev_h))


        next_c = torch.sum(f_k * c_k, dim=1) + i * u

        next_h = o * self.tanh(next_c)


        return torch.stack((next_h, next_c), dim=1)



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

    
    # pretrained_weights = pickle.load(open(data_dir + "/nodeid_to_semantic_embeddings.pkl", "rb"))

    nentity = len(id2node)
    nrelation = len(id2relation)
    max_length = max([len(_) for _ in session_dict.values()])

    print("nentity: ", nentity)
    print("nrelation: ", nrelation)
    print("max_length: ", max_length)

    batch_size = 128
    embedding_size = 384

    print("batch_size: ", batch_size)
    print("embedding_size: ", embedding_size)


    model = TreeLSTM(num_entities=nentity, num_relations=nrelation, embedding_size=embedding_size)
    if torch.cuda.is_available():
        model = model.cuda()

    batch_size = batch_size
    train_iterators = {}
    for query_type, query_answer_dict in train_data_dict.items():

        print("====================================")
        print(query_type)

        new_iterator = dataloader.SingledirectionalOneShotIterator(DataLoader(
            dataloader.TrainDataset(nentity, nrelation, query_answer_dict, max_length),
            batch_size=batch_size,
            shuffle=True,
            collate_fn=dataloader.TrainDataset.collate_fn
        ))
        train_iterators[query_type] = new_iterator

    for key, iterator in train_iterators.items():
        print("read ", key)
        batched_query, unified_ids, positive_sample = next(iterator)
        print(batched_query)
        print(unified_ids)
        print(positive_sample)

        query_embedding = model(batched_query)
        print(query_embedding.shape)
        loss = model(batched_query, positive_sample)
        print(loss)

    validation_loaders = {}
    for query_type, query_answer_dict in valid_data_dict.items():

    

        print("====================================")

        print(query_type)

        new_iterator = DataLoader(
            dataloader.ValidDataset(nentity, nrelation, query_answer_dict, max_length),
            batch_size=batch_size,
            shuffle=True,
            collate_fn=dataloader.ValidDataset.collate_fn
        )
        validation_loaders[query_type] = new_iterator

    for key, loader in validation_loaders.items():
        print("read ", key)
        for batched_query, unified_ids, train_answers, valid_answers in loader:
            print(batched_query)
            print(unified_ids)
            print([len(_) for _ in train_answers])
            print([len(_) for _ in valid_answers])

            query_embedding = model(batched_query)
            result_logs = model.evaluate_entailment(query_embedding, train_answers)
            print(result_logs)

            result_logs = model.evaluate_generalization(query_embedding, train_answers, valid_answers)
            print(result_logs)

            break

    test_loaders = {}
    for query_type, query_answer_dict in test_data_dict.items():

    

        print("====================================")
        print(query_type)

        new_loader = DataLoader(
            dataloader.TestDataset(nentity, nrelation, query_answer_dict, max_length),
            batch_size=batch_size,
            shuffle=True,
            collate_fn=dataloader.TestDataset.collate_fn
        )
        test_loaders[query_type] = new_loader

    for key, loader in test_loaders.items():
        print("read ", key)
        for batched_query, unified_ids, train_answers, valid_answers, test_answers in loader:
            print(batched_query)
            print(unified_ids)

            query_embedding = model(batched_query)
            result_logs = model.evaluate_entailment(query_embedding, train_answers)
            print(result_logs)

            result_logs = model.evaluate_generalization(query_embedding, valid_answers, test_answers)
            print(result_logs)

            print(train_answers[0])
            print([len(_) for _ in train_answers])
            print([len(_) for _ in valid_answers])
            print([len(_) for _ in test_answers])

            break
