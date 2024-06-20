import json

import torch
from torch import nn
from torch.utils.data import DataLoader

import dataloader
import model

from model import LabelSmoothingLoss

import numpy as np
import pickle

class DualHeterogeneousTransformer(nn.Module):

    def __init__(self, embedding_size):
        super(DualHeterogeneousTransformer, self).__init__()

        self.embedding_size = embedding_size
        
       
        self.mask_embedding = nn.Embedding(1, embedding_size)

        hidden_size = embedding_size

        # Define linear layers for query, key, and value projections
        self.query_e = nn.Linear(hidden_size, hidden_size)
        self.key_e = nn.Linear(hidden_size, hidden_size)
        self.value_e = nn.Linear(hidden_size, hidden_size)
    
        self.query_r = nn.Linear(hidden_size, hidden_size)
        self.key_r = nn.Linear(hidden_size, hidden_size)
        self.value_r = nn.Linear(hidden_size, hidden_size)

        self.positional_encoding_e = nn.Embedding(100, hidden_size)
        self.positional_encoding_r = nn.Embedding(100, hidden_size)

        



    def forward(self, query_entity_encoding, relation_encoding):

        if len(query_entity_encoding.shape) == 2:
            query_entity_encoding = query_entity_encoding.unsqueeze(1)
        

        batch_size, encoding_length, embedding_size = query_entity_encoding.shape

        if encoding_length > 99:
            query_entity_encoding = query_entity_encoding[:, -99:, :]
            encoding_length = 99

        index = torch.zeros(batch_size, 1).long().to(query_entity_encoding.device)

        query_entity_encoding = torch.concat([query_entity_encoding, self.mask_embedding(index)], dim=1)

        index = torch.arange(encoding_length + 1).to(query_entity_encoding.device)
        query_entity_encoding = query_entity_encoding + self.positional_encoding_e(index)

        # [batch_size, encoding_length, embedding_size]
        relation_encoding = relation_encoding.unsqueeze(1).repeat(1, encoding_length, 1)
        index = torch.arange(encoding_length).to(query_entity_encoding.device)
        relation_encoding = relation_encoding + self.positional_encoding_r(index)

        Qe = self.query_e(query_entity_encoding)
        Ke = self.key_e(query_entity_encoding)
        Ve = self.value_e(query_entity_encoding)

        Qr = self.query_r(relation_encoding)
        Kr = self.key_r(relation_encoding)
        Vr = self.value_r(relation_encoding)

        # [batch_size, encoding_length, encoding_length]

        Aee = torch.matmul(Qe, Ke.transpose(1, 2)) / torch.sqrt(torch.tensor(self.embedding_size).to(query_entity_encoding.device))
        Aer = torch.matmul(Qe, Kr.transpose(1, 2)) / torch.sqrt(torch.tensor(self.embedding_size).to(query_entity_encoding.device))

        Are = torch.matmul(Qr, Ke.transpose(1, 2)) / torch.sqrt(torch.tensor(self.embedding_size).to(query_entity_encoding.device))
        Arr = torch.matmul(Qr, Kr.transpose(1, 2)) / torch.sqrt(torch.tensor(self.embedding_size).to(query_entity_encoding.device))

        Ae = torch.concat([Aee, Aer], dim=2)
        Ar = torch.concat([Are, Arr], dim=2)

        A = torch.concat([Ae, Ar], dim=1)

        A = torch.softmax(A, dim=2)

        output = torch.matmul(A, torch.concat([Ve, Vr], dim=1))

        

        return output[:,0]

        
     
    
class NQE(model.IterativeModel):

    def __init__(self, num_entities, num_relations, embedding_size, label_smoothing=0.1, use_old_loss=False,negative_size=128, pretrained_weights=None):
        super(NQE, self).__init__(num_entities, num_relations, embedding_size, use_old_loss)

        if pretrained_weights is not None:

            print(len(pretrained_weights), num_entities)
            assert len(pretrained_weights) == num_entities
            assert len(pretrained_weights[0]) == embedding_size

            zero_array = np.random.randn(1, embedding_size)
            
            pretrained_weights = np.concatenate((zero_array, pretrained_weights), axis=0)
            pretrained_weights = torch.tensor(pretrained_weights, dtype=torch.float32)
            self.entity_embedding = nn.Embedding.from_pretrained(pretrained_weights)
        else:
            self.entity_embedding = nn.Embedding(num_entities + 1, embedding_size)

        self.relation_embedding = nn.Embedding(num_relations, embedding_size)

        self.relu = nn.ReLU()
      
        embedding_weights = self.entity_embedding.weight
        self.decoder = nn.Linear(embedding_size,
                                 num_entities,
                                 bias=False)
        self.decoder.weight = embedding_weights

        self.encoder = DualHeterogeneousTransformer(embedding_size)
        # There is only two roles in the session graph, so we adjust the number of roles to 2, for relations and 
        # entities respectively. 

        self.MLP = nn.Sequential(
            nn.Linear(embedding_size, embedding_size),
            nn.ReLU(),
            nn.Linear(embedding_size, embedding_size),
            nn.LayerNorm(embedding_size),
            nn.Sigmoid()
        )

        self.label_smoothing_loss = LabelSmoothingLoss(smoothing=label_smoothing)


    def scoring(self, query_encoding):
        """

        :param query_encoding: [batch_size, embedding_size]
        :return: [batch_size, num_entities]
        """

        query_scores = self.decoder(query_encoding)
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

        :param relation_ids: [batch_size]
        :param sub_query_encoding: [batch_size, embedding_size]
        :return: [batch_size, embedding_size]
        """

        # [batch_size, embedding_size]
        relation_ids = torch.tensor(relation_ids)
        relation_ids = relation_ids.to(self.relation_embedding.weight.device)
        relation_embeddings = self.relation_embedding(relation_ids)

        attention_output = self.encoder(sub_query_encoding, relation_embeddings)
        query_encoding = self.MLP(attention_output) 

        return query_encoding

    def higher_projection(self, relation_ids, sub_query_encoding):
        return self.projection(relation_ids, sub_query_encoding)


    def intersection(self, sub_query_encoding_list):
        """

        :param sub_query_encoding_list: a list of the sub-query embeddings of size [batch_size, embedding_size]
        :return:  [batch_size, embedding_size]
        """

        if len(sub_query_encoding_list) == 2:
            return sub_query_encoding_list[0] * sub_query_encoding_list[1]

        elif len(sub_query_encoding_list) == 3:
            return sub_query_encoding_list[0] * sub_query_encoding_list[1] * sub_query_encoding_list[2]
        
        else:
            raise NotImplementedError
        
    def union(self, sub_query_encoding_list):

        if len(sub_query_encoding_list) == 2:
            return (sub_query_encoding_list[0] + sub_query_encoding_list[1]) - (sub_query_encoding_list[0] * sub_query_encoding_list[1])

        elif len(sub_query_encoding_list) == 3:
            term_1 = sub_query_encoding_list[0] + sub_query_encoding_list[1] + sub_query_encoding_list[2]
            term_2 = sub_query_encoding_list[0] * sub_query_encoding_list[1] + sub_query_encoding_list[1] * sub_query_encoding_list[2] + sub_query_encoding_list[0] * sub_query_encoding_list[2]
            term_3 = sub_query_encoding_list[0] * sub_query_encoding_list[1] * sub_query_encoding_list[2]
            return term_1 - term_2 + term_3
        
        else:
            raise NotImplementedError

    def negation(self, sub_query_encoding):
        return 1 - sub_query_encoding

    def forward(self, batched_structured_query, label=None):
        assert batched_structured_query[0] in ["p", "e", "i", "u", "n", "s"]

        if batched_structured_query[0] == "p":

            sub_query_result = self.forward(batched_structured_query[2])

            if batched_structured_query[2][0] == 's':
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

        elif batched_structured_query[0] == "e" or batched_structured_query[0] == "s":

            entity_ids = torch.tensor(batched_structured_query[1])
            entity_ids = entity_ids.to(self.entity_embedding.weight.device)
            this_query_result = self.entity_embedding(entity_ids)
        
        else:
            this_query_result = None

        if label is None:
            return this_query_result

        else:
            return self.loss_fnt(this_query_result, label)



if __name__ == "__main__":

    train_data_path = "../sampled_hyper_train_dressipi_merged/dressipi_train_queries.json"
    valid_data_path = "../sampled_hyper_valid_dressipi/dressipi_valid_queries_0.json"
    test_data_path = "../sampled_hyper_test_dressipi/dressipi_test_queries_0.json"

    with open(train_data_path, "r") as fin:
        train_data_dict = json.load(fin)

    with open(valid_data_path, "r") as fin:
        valid_data_dict = json.load(fin)

    with open(test_data_path, "r") as fin:
        test_data_dict = json.load(fin)

    data_dir = "../hyper_graph_dressipi"

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

    batch_size = 15
    embedding_size = 384

    print("batch_size: ", batch_size)
    print("embedding_size: ", embedding_size)

    


    gqe_model = NQE(num_entities=nentity, num_relations=nrelation, embedding_size=embedding_size,use_old_loss=True)
    # if torch.cuda.is_available():
    #     gqe_model = gqe_model.cuda()

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

        query_embedding = gqe_model(batched_query)
        print(query_embedding.shape)
        loss = gqe_model(batched_query, positive_sample)
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

            query_embedding = gqe_model(batched_query)
            result_logs = gqe_model.evaluate_entailment(query_embedding, train_answers)
            print(result_logs)

            result_logs = gqe_model.evaluate_generalization(query_embedding, train_answers, valid_answers)
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

            query_embedding = gqe_model(batched_query)
            result_logs = gqe_model.evaluate_entailment(query_embedding, train_answers)
            print(result_logs)

            result_logs = gqe_model.evaluate_generalization(query_embedding, valid_answers, test_answers)
            print(result_logs)

            print(train_answers[0])
            print([len(_) for _ in train_answers])
            print([len(_) for _ in valid_answers])
            print([len(_) for _ in test_answers])

            break
