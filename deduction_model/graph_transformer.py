import json

import torch
from torch import nn
from torch.utils.data import DataLoader

from dataloader import TestDatasetPE, ValidDatasetPE, TrainDatasetPE, SingledirectionalOneShotIterator, std_offset, \
    special_token_dict
from model import SequentialModel, LabelSmoothingLoss, GeneralModel
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel, BertConfig
import numpy as np

class GTModel(GeneralModel):
    def __init__(self, num_entities, num_relations, embedding_size, label_smoothing=0.1, num_layers=2, pretrained_weights=None, positonal_encoding_size = 16):
        super(GTModel, self).__init__(num_entities, num_relations, embedding_size)

        config = BertConfig(hidden_size=embedding_size, vocab_size=num_entities + num_relations + std_offset,
                            num_attention_heads=8, num_hidden_layers=num_layers, position_embedding_type='relative_key')#for study on improving compositional generalizability
        self.transformer_encoder = BertModel(config)


        if pretrained_weights is not None:

            print(len(pretrained_weights), num_entities)
            assert len(pretrained_weights) == num_entities
            assert len(pretrained_weights[0]) == embedding_size

    
            # zero_array_1 =  np.array([[0.0] * embedding_size] * std_offset)
            zero_array_1 = np.random.randn(std_offset, embedding_size)

            # zero_array_2 = np.array([[0.0] * embedding_size] * num_relations)
            zero_array_2 = np.random.randn(num_relations, embedding_size)

            pretrained_weights = np.concatenate([zero_array_1, pretrained_weights, zero_array_2], axis=0)   

    
            pretrained_weights = torch.tensor(pretrained_weights, dtype=torch.float32)
            self.unified_embeddings = nn.Embedding.from_pretrained(pretrained_weights, freeze=False)


        else:
            # self.entity_embedding = nn.Embedding(num_entities + 1, embedding_size)

            self.unified_embeddings = nn.Embedding(num_entities + num_relations + std_offset, embedding_size)


        self.transformer_encoder.embeddings.word_embeddings = self.unified_embeddings

        embedding_weights = self.transformer_encoder.embeddings.word_embeddings.weight
        self.decoder = nn.Linear(self.transformer_encoder.config.hidden_size,
                                 self.transformer_encoder.config.vocab_size, bias=False)
        self.decoder.weight = embedding_weights



        self.positonal_encoding_size = positonal_encoding_size
        self.pe_transformation = nn.Linear(self.positonal_encoding_size, embedding_size, bias=False)

        self.label_smoothing_loss = LabelSmoothingLoss(smoothing=label_smoothing)

    def encode(self, batched_structured_query, attention_mask, positional_encoding):

        token_embeddings = self.transformer_encoder.embeddings.word_embeddings(batched_structured_query)
        converted_positional_encoding = self.pe_transformation(positional_encoding)


        encoded = self.transformer_encoder(inputs_embeds = token_embeddings + converted_positional_encoding, 
                                           attention_mask=attention_mask,
                                           ).pooler_output

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

        # [batch_size, num_particles, num_entities]
        query_scores = torch.matmul(query_encoding, entity_embeddings.transpose(-2, -1))

        return query_scores
    
    def forward(self, batched_structured_query, attention_mask, positional_encoding, label=None):
        
    
        batched_structured_query = torch.tensor(batched_structured_query)
        if torch.cuda.is_available():
            batched_structured_query = batched_structured_query.cuda()
        
        attention_mask = torch.tensor(attention_mask)
        if torch.cuda.is_available():
            attention_mask = attention_mask.cuda()
        
        positional_encoding = torch.tensor(positional_encoding)
        if torch.cuda.is_available():
            positional_encoding = positional_encoding.cuda()

        representations = self.encode(batched_structured_query, attention_mask, positional_encoding)

        if label is not None:
            return self.loss_fnt(representations, label)

        else:
            return representations
    

    def loss_fnt(self, query_encoding, labels):
        # [batch_size, num_entities]
        query_scores = self.scoring(query_encoding)

        # [batch_size]
        labels = torch.tensor(labels)
        labels = labels.to(query_encoding.device)

        _loss = self.label_smoothing_loss(query_scores, labels)

        return _loss


if __name__ == "__main__":

    train_data_path = "../sampled_hyper_train_merged_old_graph_pe/amazon_train_queries_9_pe.json"
    valid_data_path = "../sampled_hyper_valid_graph_pe/amazon_valid_queries_25_pe.json"
    test_data_path = "../sampled_hyper_test_graph_pe/amazon_test_queries_34_pe.json"

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

    batch_size = 512
    embedding_size = 384

    print("batch_size: ", batch_size)
    print("embedding_size: ", embedding_size)


    model = GTModel(num_entities=nentity, num_relations=nrelation, embedding_size=embedding_size)
    if torch.cuda.is_available():
        model = model.cuda()

    batch_size = batch_size
    train_iterators = {}
    for query_type, query_answer_dict in train_data_dict.items():
        print("====================================")
        print(query_type)

        new_iterator = SingledirectionalOneShotIterator(DataLoader(
            TrainDatasetPE(nentity, nrelation, query_answer_dict),
            batch_size=batch_size,
            shuffle=True,
            collate_fn=TrainDatasetPE.collate_fn
        ))
        train_iterators[query_type] = new_iterator

    for key, iterator in train_iterators.items():
        print("read ", key)
        unified_graph_id_list, graph_RWPE_list, graph_LapPE_list, positive_sample, attention_mask = next(iterator)
        print(unified_graph_id_list)
        
        print(positive_sample)

        query_embedding = model(unified_graph_id_list, attention_mask, graph_RWPE_list)
        print(query_embedding.shape)
        loss = model(unified_graph_id_list, attention_mask, graph_RWPE_list, positive_sample)
        print(loss)


        query_embedding = model(unified_graph_id_list, attention_mask, graph_LapPE_list)
        print(query_embedding.shape)
        loss = model(unified_graph_id_list, attention_mask, graph_LapPE_list, positive_sample)
        print(loss)

    validation_loaders = {}
    for query_type, query_answer_dict in valid_data_dict.items():
        print("====================================")

        print(query_type)

        new_iterator = DataLoader(
            ValidDatasetPE(nentity, nrelation, query_answer_dict),
            batch_size=batch_size,
            shuffle=True,
            collate_fn=ValidDatasetPE.collate_fn
        )
        validation_loaders[query_type] = new_iterator

    for key, loader in validation_loaders.items():
        print("read ", key)
        for unified_graph_id_list, graph_RWPE_list, graph_LapPE_list, train_answers, valid_answers, attention_mask  in loader:
      
            print(unified_graph_id_list)
            print([len(_) for _ in train_answers])
            print([len(_) for _ in valid_answers])

            query_embedding = model(unified_graph_id_list, attention_mask, graph_RWPE_list)
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
            TestDatasetPE(nentity, nrelation, query_answer_dict),
            batch_size=batch_size,
            shuffle=True,
            collate_fn=TestDatasetPE.collate_fn
        )
        test_loaders[query_type] = new_loader

    for key, loader in test_loaders.items():
        print("read ", key)
        for unified_graph_id_list, graph_RWPE_list, graph_LapPE_list,  train_answers, valid_answers, test_answers, attention_mask in loader:
          
            print(unified_graph_id_list)

            query_embedding = model(unified_graph_id_list, attention_mask, graph_LapPE_list)
            result_logs = model.evaluate_entailment(query_embedding, train_answers)
            print(result_logs)

            result_logs = model.evaluate_generalization(query_embedding, valid_answers, test_answers)
            print(result_logs)

            print(train_answers[0])
            print([len(_) for _ in train_answers])
            print([len(_) for _ in valid_answers])
            print([len(_) for _ in test_answers])

            break
