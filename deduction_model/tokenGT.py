import json

import torch
from torch import nn
from torch.utils.data import DataLoader

from dataloader import TrainDatasetTokenGT, ValidDatasetTokenGT, TestDatasetTokenGT, SingledirectionalOneShotIterator, std_offset, \
    special_token_dict
from model import SequentialModel, LabelSmoothingLoss
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel, BertConfig
import numpy as np

class TokenGTModel(SequentialModel):
    def __init__(self, num_entities, num_relations, embedding_size, label_smoothing=0.1, num_layers=2, no_logic = False, no_order = False):
        super(TokenGTModel, self).__init__(num_entities, num_relations, embedding_size)

        config = BertConfig(hidden_size=embedding_size, vocab_size=num_entities + num_relations + std_offset + 128,
                            num_attention_heads=8, num_hidden_layers=num_layers, position_embedding_type='relative_key')#for study on improving compositional generalizability
        self.transformer_encoder = BertModel(config)

        self.max_length = 64
      
        self.unified_embeddings = nn.Embedding(num_entities + num_relations + std_offset + 128, embedding_size)


        self.node_type_embeddings = nn.Embedding(10, embedding_size)


        self.node_identitier_embeddings = nn.Embedding(self.max_length, embedding_size)
        # initialize the node identifier embeddings with orthogonal normal vectors
        nn.init.orthogonal_(self.node_identitier_embeddings.weight)
        # freeze the node identifier embeddings
        self.node_identitier_embeddings.weight.requires_grad = False
        
        
        self.type_embeddings = nn.Embedding(10, embedding_size)
        

        self.transformer_encoder.embeddings.word_embeddings = self.unified_embeddings

        embedding_weights = self.transformer_encoder.embeddings.word_embeddings.weight


        self.decoder = nn.Linear(self.transformer_encoder.config.hidden_size,
                                 self.transformer_encoder.config.vocab_size, bias=False)
        
        self.decoder.weight = embedding_weights

        self.label_smoothing_loss = LabelSmoothingLoss(smoothing=label_smoothing)

        self.graph_token_embeddings = nn.Embedding(1, embedding_size * 4)
        self.trainable_convertion = nn.Linear(embedding_size * 4, embedding_size, bias=False)

        self.no_logic = no_logic
        self.no_order = no_order

    def encode(self, input_feature, attention_mask):

        encoded = self.transformer_encoder(inputs_embeds=input_feature, attention_mask=attention_mask).pooler_output

        return encoded

    def scoring(self, query_encoding):
        # [batch_size, num_entities]
        query_scores = self.decoder(query_encoding)[:, std_offset:std_offset + self.num_entities + 1]
        return query_scores
    
    def selected_scoring(self, query_encoding, selected_entities):
        """

        :param query_encoding: [num_particles, embedding_size]
        :param selected_entities: [num_entities]
        :return: [ num_particles]
        """
        
        # [batch_size, num_entities, embedding_size]
        entity_embeddings = self.unified_embeddings(selected_entities+std_offset)

        # [batch_size, num_entities]
        query_scores = torch.matmul(query_encoding, entity_embeddings.transpose(-2, -1))

        return query_scores

    def loss_fnt(self, query_encoding, labels):
        # [batch_size, num_entities]
        query_scores = self.scoring(query_encoding)

        # [batch_size]
        labels = torch.tensor(labels)
        labels = labels.to(query_encoding.device)

        _loss = self.label_smoothing_loss(query_scores, labels)

        return _loss

    def forward(self, unified_feature_ids, node_identifiers, type_identifiers, attention_mask, label=None):
        

        unified_feature_ids = torch.tensor(unified_feature_ids).to(self.unified_embeddings.weight.device)
        node_identifiers = torch.tensor(node_identifiers).to(self.unified_embeddings.weight.device)
        type_identifiers = torch.tensor(type_identifiers).to(self.unified_embeddings.weight.device)

        attention_mask = torch.tensor(attention_mask).to(self.unified_embeddings.weight.device)
        if self.no_logic:
            attention_mask = (type_identifiers == 0)

        max_relation_id = self.num_entities + std_offset + self.num_relations + 128

        # For required ablation study...
        if self.no_order:
            max_relation_id = self.num_entities + std_offset + self.num_relations

        unified_feature_ids[unified_feature_ids >= max_relation_id] = max_relation_id - 1



        # [batch_size, max_length, embedding_size]
        unified_token_feature = self.unified_embeddings(unified_feature_ids).unsqueeze(2)

        # [batch_size, max_length, embedding_size]
        node_type_feature = self.node_type_embeddings(type_identifiers).unsqueeze(2)

        # [batch_size, max_length, 2, embedding_size]
        node_identifier_feature = self.node_identitier_embeddings(node_identifiers)
        
        
        stacked_feature = torch.cat([unified_token_feature, node_type_feature, node_identifier_feature], dim=2)
        stacked_feature = stacked_feature.reshape(-1, self.max_length, self.embedding_size * 4)

        # append the graph token
        graph_token_feature = self.graph_token_embeddings(torch.zeros(stacked_feature.shape[0], dtype=torch.long).to(stacked_feature.device)).unsqueeze(1)
        stacked_feature = torch.cat([graph_token_feature, stacked_feature], dim=1)

        # [batch_size, max_length+1, embedding_size]
        stacked_feature = self.trainable_convertion(stacked_feature)

        # [batch_size, max_length+1]
        attention_mask = torch.cat([torch.ones(attention_mask.shape[0], 1).to(attention_mask.device), attention_mask], dim=1)

        # [batch_size, embedding_size]

        # print(stacked_feature.shape)
        query_encoding = self.encode(stacked_feature, attention_mask=attention_mask)
        

        if label is not None:
            return self.loss_fnt(query_encoding, label)

        return query_encoding
    


    
if __name__ == "__main__":

    train_data_path = "../sampled_hyper_train_diginetica_merged_graph_dgl/diginetica_train_queries_dgl.json"
    valid_data_path = "../sampled_hyper_valid_dininetica_graph_dgl/diginetica_valid_queries_10_dgl.json"
    test_data_path = "../sampled_hyper_test_diginetica_graph_dgl/diginetica_test_queries_9_dgl.json"

    with open(train_data_path, "r") as fin:
        train_data_dict = json.load(fin)

    with open(valid_data_path, "r") as fin:
        valid_data_dict = json.load(fin)

    with open(test_data_path, "r") as fin:
        test_data_dict = json.load(fin)

    data_dir = "../hyper_graph_diginetica"

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

    batch_size = 512 
    embedding_size = 384

    print("batch_size: ", batch_size)
    print("embedding_size: ", embedding_size)

   
    model = TokenGTModel(num_entities=nentity, num_relations=nrelation, embedding_size=embedding_size, no_order=True)
    if torch.cuda.is_available():
        model = model.cuda()

    batch_size = batch_size
    train_iterators = {}
    for query_type, query_answer_dict in train_data_dict.items():
        print("====================================")
        print(query_type)

        new_iterator = SingledirectionalOneShotIterator(DataLoader(
            TrainDatasetTokenGT(nentity, nrelation, query_answer_dict),
            batch_size=batch_size,
            shuffle=True,
            collate_fn=TrainDatasetTokenGT.collate_fn
        ))
        train_iterators[query_type] = new_iterator

    for key, iterator in train_iterators.items():
        print("read ", key)
        unified_feature_ids, node_identifiers, type_identifiers, attention_mask, positive_sample = next(iterator)
        print(unified_feature_ids)
        print(node_identifiers)
        print(positive_sample)

        query_embedding = model(unified_feature_ids, node_identifiers, type_identifiers, attention_mask)
        print(query_embedding.shape)
        loss = model(unified_feature_ids, node_identifiers, type_identifiers, attention_mask, positive_sample)
        print(loss)

    validation_loaders = {}
    for query_type, query_answer_dict in valid_data_dict.items():
        print("====================================")

        print(query_type)

        new_iterator = DataLoader(
            ValidDatasetTokenGT(nentity, nrelation, query_answer_dict),
            batch_size=batch_size,
            shuffle=True,
            collate_fn=ValidDatasetTokenGT.collate_fn
        )
        validation_loaders[query_type] = new_iterator

    for key, loader in validation_loaders.items():
        print("read ", key)
        for unified_feature_ids, node_identifiers, type_identifiers, attention_mask, train_answer_list, valid_answer_list  in loader:
            print(unified_feature_ids)
            print(node_identifiers)
            print([len(_) for _ in train_answer_list])
            print([len(_) for _ in valid_answer_list])

            query_embedding = model(unified_feature_ids, node_identifiers, type_identifiers, attention_mask)
            result_logs = model.evaluate_entailment(query_embedding, train_answer_list)
            print(result_logs)

            result_logs = model.evaluate_generalization(query_embedding, train_answer_list, valid_answer_list)
            print(result_logs)

            break

    test_loaders = {}
    for query_type, query_answer_dict in test_data_dict.items():
        print("====================================")
        print(query_type)

        new_loader = DataLoader(
            TestDatasetTokenGT(nentity, nrelation, query_answer_dict),
            batch_size=batch_size,
            shuffle=True,
            collate_fn=TestDatasetTokenGT.collate_fn
        )
        test_loaders[query_type] = new_loader

    for key, loader in test_loaders.items():
        print("read ", key)
        for unified_feature_ids, node_identifiers, type_identifiers, attention_mask, train_answer_list, valid_answer_list, test_answer_list in loader:
            print(unified_feature_ids)
            print(node_identifiers)

            query_embedding = model(unified_feature_ids, node_identifiers, type_identifiers, attention_mask)
            result_logs = model.evaluate_entailment(query_embedding, train_answer_list)
            print(result_logs)

            result_logs = model.evaluate_generalization(query_embedding, valid_answer_list, test_answer_list)
            print(result_logs)

            print(train_answer_list[0])
            print([len(_) for _ in train_answer_list])
            print([len(_) for _ in valid_answer_list])
            print([len(_) for _ in test_answer_list])

            break
