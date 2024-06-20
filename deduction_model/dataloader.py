#!/usr/bin/python3

import json

import numpy as np
from torch.utils.data import Dataset, DataLoader
import os

os.environ["DGLBACKEND"] = "pytorch"
import dgl
import torch

# TODO: Add unified tokens, expressing both relations, entities, and logical operations
#
special_token_dict = {
    "(": 0,
    ")": 1,
    "p": 2,
    "i": 3,
    "u": 4,
    "n": 5,
    "s": 6,
    "[PAD]": 99
}
std = special_token_dict
std_offset = 100




def abstraction(instantiated_query, nentity, nrelation, max_session_length):

    query = instantiated_query[1:-1]
    parenthesis_count = 0

    sub_queries = []
    jj = 0

    def r(relation_id):
        return relation_id + nentity + std_offset

    def e(entity_id):
        return entity_id + std_offset

    for ii, character in enumerate(query):
        # Skip the comma inside a parenthesis
        if character == "(":
            parenthesis_count += 1

        elif character == ")":
            parenthesis_count -= 1

        if parenthesis_count > 0:
            continue

        if character == ",":
            sub_queries.append(query[jj: ii])
            jj = ii + 1

    sub_queries.append(query[jj: len(query)])

    if sub_queries[0] == "p":

        sub_ids_list, sub_query_type, sub_unified_ids = abstraction(sub_queries[2], nentity, nrelation, max_session_length)
        relation_id = int(sub_queries[1][1:-1])
        ids_list =  [relation_id] + sub_ids_list
        this_query_type = "(p," + sub_query_type + ")"
        this_unified_ids = [std["("], std["p"] , r(relation_id)] + sub_unified_ids + [std[")"]]
        return ids_list, this_query_type, this_unified_ids

    elif sub_queries[0] == "e":

        entity_id = int(sub_queries[1][1:-1])

        ids_list = [entity_id]
        this_query_type = "(e)"
        this_unified_ids = [std["("], e(entity_id), std[")"]]
        return ids_list, this_query_type, this_unified_ids

    elif sub_queries[0] == "s":
        # session_id = int(sub_queries[1][1:-1])
        session_item_ids = [int(item) for item in sub_queries[2][1:-1].split(",")]
        extended_session_item_ids = session_item_ids 

        this_query_type = "(s)"
        while len(extended_session_item_ids) < max_session_length:
            extended_session_item_ids.append(nentity)
        ids_list = session_item_ids

        this_unified_ids = [std["("], std["s"]] + [e(item) for item in session_item_ids] + [
            std["PAD"] for _ in range(len(extended_session_item_ids) - len(session_item_ids))] + [std[")"]]

        return ids_list, this_query_type, this_unified_ids


    elif sub_queries[0] == "i":

        ids_list = []
        this_query_type = "(i"
        this_unified_ids = [std["("], std["i"]]

        for i in range(1, len(sub_queries)):
            sub_ids_list, sub_query_type, sub_unified_ids = abstraction(sub_queries[i], nentity, nrelation, max_session_length)
            ids_list.extend(sub_ids_list)
            this_query_type = this_query_type +"," + sub_query_type

            this_unified_ids = this_unified_ids + sub_unified_ids

        this_query_type = this_query_type + ")"
        this_unified_ids = this_unified_ids + [std[")"]]

        return ids_list, this_query_type, this_unified_ids


    elif sub_queries[0] == "u":
        ids_list = []
        this_query_type = "(u"
        this_unified_ids = [std["("], std["u"]]

        for i in range(1, len(sub_queries)):
            sub_ids_list, sub_query_type, sub_unified_ids = abstraction(sub_queries[i], nentity, nrelation, max_session_length)
            ids_list.extend(sub_ids_list)
            this_query_type = this_query_type + "," + sub_query_type

            this_unified_ids = this_unified_ids + sub_unified_ids

        this_query_type = this_query_type + ")"
        this_unified_ids = this_unified_ids + [std[")"]]

        return ids_list, this_query_type, this_unified_ids

    elif sub_queries[0] == "n":
        sub_ids_list, sub_query_type, sub_unified_ids = abstraction(sub_queries[1], nentity, nrelation, max_session_length)
        return sub_ids_list, "(n," + sub_query_type + ")", [std["("], std["n"]] + sub_unified_ids + [std[")"]]
    
    

    else:
        print("Invalid Pattern")
        exit()



class Instantiation(object):


    def __init__(self, value_matrix, max_session_length):
        self.value_matrix = np.array(value_matrix)
        self.max_session_length = max_session_length


    def instantiate(self, query_pattern):

        query = query_pattern[1:-1]
        parenthesis_count = 0

        sub_queries = []
        jj = 0

        for ii, character in enumerate(query):
            # Skip the comma inside a parenthesis
            if character == "(":
                parenthesis_count += 1

            elif character == ")":
                parenthesis_count -= 1

            if parenthesis_count > 0:
                continue

            if character == ",":
                sub_queries.append(query[jj: ii])
                jj = ii + 1

        sub_queries.append(query[jj: len(query)])

        if sub_queries[0] == "p":


            relation_ids = self.value_matrix[:,0]
            self.value_matrix = self.value_matrix[:,1:]
            sub_batched_query = self.instantiate(sub_queries[1])

            return ("p", relation_ids, sub_batched_query )

        elif sub_queries[0] == "e":
            entity_ids = self.value_matrix[:,0]
            self.value_matrix = self.value_matrix[:, 1:]

            return  ("e",  entity_ids)
        
        elif sub_queries[0] == "s":
            session_item_ids = self.value_matrix[:,0: self.max_session_length]
            self.value_matrix = self.value_matrix[:,self.max_session_length:]

            return ("s", session_item_ids)

        elif sub_queries[0] == "i":

            return_list = ["i"]
            for i in range(1, len(sub_queries)):
                sub_batched_query = self.instantiate(sub_queries[i])
                return_list.append(sub_batched_query)

            return tuple(return_list)


        elif sub_queries[0] == "u":
            return_list = ["u"]
            for i in range(1, len(sub_queries)):
                sub_batched_query = self.instantiate(sub_queries[i])
                return_list.append(sub_batched_query)

            return tuple(return_list)

        elif sub_queries[0] == "n":
            sub_batched_query = self.instantiate(sub_queries[1])

            return ("n", sub_batched_query)

        else:
            print("Invalid Pattern")
            exit()



class TestDataset(Dataset):
    def __init__(self, nentity, nrelation, query_answers_dict, max_session_length):
        self.len = len(query_answers_dict)
        self.query_answers_dict = query_answers_dict
        self.nentity = nentity
        self.nrelation = nrelation

        self.max_session_length = max_session_length

        self.id_list = []
        self.train_answer_list = []
        self.valid_answer_list = []
        self.test_answer_list = []

        self.unified_id_list = []

        self.query_type = None

        for query, answer_list in query_answers_dict.items():
            this_id_list, this_query_type, unified_ids = abstraction(query, nentity, nrelation, max_session_length)
            self.train_answer_list.append([int(ans) for ans in answer_list["train_answers"]])
            self.valid_answer_list.append([int(ans) for ans in answer_list["valid_answers"]])
            self.test_answer_list.append([int(ans) for ans in answer_list["test_answers"]])

            self.unified_id_list.append(unified_ids)

            self.id_list.append(this_id_list)

            if self.query_type is None:
                self.query_type = this_query_type
            else:
                assert self.query_type == this_query_type


    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        ids_in_query = self.id_list[idx]
        train_answer_list = self.train_answer_list[idx]
        valid_answer_list = self.valid_answer_list[idx]
        test_answer_list = self.test_answer_list[idx]
        unified_id_list  = self.unified_id_list[idx]

        return ids_in_query, unified_id_list, train_answer_list, valid_answer_list, test_answer_list, self.query_type, self.max_session_length

    @staticmethod
    def collate_fn(data):
        train_answers = [_[2] for _ in data]
        valid_answers = [_[3] for _ in data]
        test_answers = [_[4] for _ in data]

        ids_in_query_matrix = [_[0] for _ in data]

        query_type = [_[5] for _ in data]
        max_session_length = [_[6] for _ in data]

        unified_ids = [_[1] for _ in data]

        batched_query = Instantiation(ids_in_query_matrix, max_session_length[0]).instantiate(query_type[0])

        return batched_query, unified_ids, train_answers, valid_answers, test_answers


class ValidDataset(Dataset):
    def __init__(self, nentity, nrelation, query_answers_dict, max_session_length):
        self.len = len(query_answers_dict)
        self.query_answers_dict = query_answers_dict
        self.nentity = nentity
        self.nrelation = nrelation
        self.max_session_length = max_session_length

        self.id_list = []
        self.train_answer_list = []
        self.valid_answer_list = []
        self.unified_id_list = []

        self.query_type = None

        for query, answer_list in query_answers_dict.items():
            this_id_list, this_query_type, unified_ids = abstraction(query, nentity, nrelation, max_session_length)
            self.train_answer_list.append([int(ans) for ans in answer_list["train_answers"]])
            self.valid_answer_list.append([int(ans) for ans in answer_list["valid_answers"]])

            self.unified_id_list.append(unified_ids)

            self.id_list.append(this_id_list)
            if self.query_type is None:
                self.query_type = this_query_type
            else:
                assert self.query_type == this_query_type


    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        ids_in_query = self.id_list[idx]
        train_answer_list = self.train_answer_list[idx]
        valid_answer_list = self.valid_answer_list[idx]
        unified_id_list = self.unified_id_list[idx]

        return ids_in_query, unified_id_list, train_answer_list, valid_answer_list, self.query_type, self.max_session_length

    @staticmethod
    def collate_fn(data):
        train_answers = [_[2] for _ in data]
        valid_answers = [_[3] for _ in data]


        ids_in_query_matrix = [_[0] for _ in data]
        query_type = [_[4] for _ in data]
        max_session_length = [_[5] for _ in data]

        unified_ids = [_[1] for _ in data]

        batched_query = Instantiation(ids_in_query_matrix, max_session_length[0]).instantiate(query_type[0])

        return batched_query, unified_ids, train_answers, valid_answers


class TrainDataset(Dataset):
    def __init__(self, nentity, nrelation, query_answers_dict, max_session_length):
        self.len = len(query_answers_dict)
        self.query_answers_dict = query_answers_dict
        self.nentity = nentity
        self.nrelation = nrelation
        self.max_session_length = max_session_length

        self.id_list = []
        self.answer_list = []
        self.query_type = None

        self.unified_id_list = []

        for query, answer_list in query_answers_dict.items():
            this_id_list, this_query_type, unified_ids = abstraction(query, nentity, nrelation, max_session_length)
            self.answer_list.append([int(ans )for ans in answer_list["train_answers"]])
            self.id_list.append(this_id_list)

            self.unified_id_list.append(unified_ids)

            if self.query_type is None:
                self.query_type = this_query_type
            else:
                assert self.query_type == this_query_type


    def __len__(self):
        return self.len

    def __getitem__(self, idx):

        ids_in_query = self.id_list[idx]
        answer_list = self.answer_list[idx]
        unified_id_list = self.unified_id_list[idx]

        tail = np.random.choice(list(answer_list))

        positive_sample = int(tail)

        return ids_in_query, unified_id_list, positive_sample, self.query_type, self.max_session_length

    @staticmethod
    def collate_fn(data):
        positive_sample = [_[2] for _ in data]
        ids_in_query_matrix = [_[0] for _ in data]
        query_type = [_[3] for _ in data]
        max_session_length = [_[4] for _ in data]

        unified_ids = [_[1] for _ in data]

        batched_query = Instantiation(ids_in_query_matrix, max_session_length[0]).instantiate(query_type[0])

        return batched_query, unified_ids, positive_sample


class TestDatasetPE(Dataset):
    def __init__(self, nentity, nrelation, query_answers_dict):
        self.len = len(query_answers_dict)
        self.query_answers_dict = query_answers_dict
        self.nentity = nentity
        self.nrelation = nrelation


       
        self.train_answer_list = []
        self.valid_answer_list = []
        self.test_answer_list = []


        self.query_type = None

        self.graph_unified_tokens = []
        self.features_RWPE = []
        self.features_LapPE = []


        for query, answer_list in query_answers_dict.items():
            
            self.train_answer_list.append([int(ans) for ans in answer_list["train_answers"]])
            self.valid_answer_list.append([int(ans) for ans in answer_list["valid_answers"]])
            self.test_answer_list.append([int(ans) for ans in answer_list["test_answers"]])

        
            self.graph_unified_tokens.append(answer_list["graph_unified_tokens"])
            self.features_RWPE.append(answer_list["features_RWPE"])
            self.features_LapPE.append(answer_list["features_LapPE"])
        
    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        unified_graph_id_list  = self.graph_unified_tokens[idx]
        graph_RWPE_list = self.features_RWPE[idx]
        graph_LapPE_list = self.features_LapPE[idx]

        train_answer_list = self.train_answer_list[idx]
        valid_answer_list = self.valid_answer_list[idx]
        test_answer_list = self.test_answer_list[idx]
        

        return unified_graph_id_list, graph_RWPE_list, graph_LapPE_list, train_answer_list, valid_answer_list, test_answer_list


    @staticmethod
    def collate_fn(data):

        padding_token = std["[PAD]"]
        PE_size = len(data[0][1][0])
        max_query_length = max([len(_[0]) for _ in data])
        attention_mask = []

        # unified_graph_id_list = [_[0] for _ in data]
        # graph_RWPE_list = [_[1] for _ in data]
        # graph_LapPE_list = [_[2] for _ in data]
        # train_answers = [_[3] for _ in data]
        # valid_answers = [_[4] for _ in data]
        # test_answers = [_[5] for _ in data]

        unified_graph_id_list = []
        graph_RWPE_list = []
        graph_LapPE_list = []
        train_answers = []
        valid_answers = []
        test_answers = []


        for unified_graph_id, graph_RWPE, graph_LapPE, train_answer, valid_answer, test_answer in data:
            unified_graph_id_list.append(unified_graph_id + [padding_token for _ in range(max_query_length - len(unified_graph_id))])
            graph_RWPE_list.append(graph_RWPE + [[0.0 for _ in range(PE_size)] for _ in range(max_query_length - len(graph_RWPE))])
            graph_LapPE_list.append(graph_LapPE + [[0.0 for _ in range(PE_size)] for _ in range(max_query_length - len(graph_LapPE))])
            attention_mask.append([1 for _ in range(len(unified_graph_id))] + [0 for _ in range(max_query_length - len(unified_graph_id))])
            train_answers.append(train_answer)
            valid_answers.append(valid_answer)
            test_answers.append(test_answer)
        

        return unified_graph_id_list, graph_RWPE_list, graph_LapPE_list, train_answers, valid_answers, test_answers, attention_mask
    

class ValidDatasetPE(Dataset):
    
    def __init__(self, nentity, nrelation, query_answers_dict):
        self.len = len(query_answers_dict)
        self.query_answers_dict = query_answers_dict
        self.nentity = nentity
        self.nrelation = nrelation


       
        self.train_answer_list = []
        self.valid_answer_list = []
      


        self.query_type = None

        self.graph_unified_tokens = []
        self.features_RWPE = []
        self.features_LapPE = []


        for query, answer_list in query_answers_dict.items():
            
            self.train_answer_list.append([int(ans) for ans in answer_list["train_answers"]])
            self.valid_answer_list.append([int(ans) for ans in answer_list["valid_answers"]])
            
        
            self.graph_unified_tokens.append(answer_list["graph_unified_tokens"])
            self.features_RWPE.append(answer_list["features_RWPE"])
            self.features_LapPE.append(answer_list["features_LapPE"])
        
    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        unified_graph_id_list  = self.graph_unified_tokens[idx]
        graph_RWPE_list = self.features_RWPE[idx]
        graph_LapPE_list = self.features_LapPE[idx]

        train_answer_list = self.train_answer_list[idx]
        valid_answer_list = self.valid_answer_list[idx]
       
        

        return unified_graph_id_list, graph_RWPE_list, graph_LapPE_list, train_answer_list, valid_answer_list


    @staticmethod
    def collate_fn(data):

        # unified_graph_id_list = [_[0] for _ in data]
        # graph_RWPE_list = [_[1] for _ in data]
        # graph_LapPE_list = [_[2] for _ in data]
        # train_answers = [_[3] for _ in data]
        # valid_answers = [_[4] for _ in data]

        padding_token = std["[PAD]"]
        PE_size = len(data[0][1][0])
        max_query_length = max([len(_[0]) for _ in data])
        attention_mask = []

        unified_graph_id_list = []
        graph_RWPE_list = []
        graph_LapPE_list = []
        train_answers = []
        valid_answers = []

        for unified_graph_id, graph_RWPE, graph_LapPE, train_answer, valid_answer in data:
            unified_graph_id_list.append(unified_graph_id + [padding_token for _ in range(max_query_length - len(unified_graph_id))])
            graph_RWPE_list.append(graph_RWPE + [[0.0 for _ in range(PE_size)] for _ in range(max_query_length - len(graph_RWPE))])
            graph_LapPE_list.append(graph_LapPE + [[0.0 for _ in range(PE_size)] for _ in range(max_query_length - len(graph_LapPE))])
            attention_mask.append([1 for _ in range(len(unified_graph_id))] + [0 for _ in range(max_query_length - len(unified_graph_id))])
            train_answers.append(train_answer)
            valid_answers.append(valid_answer)
       
        return unified_graph_id_list, graph_RWPE_list, graph_LapPE_list, train_answers, valid_answers, attention_mask


class TrainDatasetPE(Dataset):
    
    def __init__(self, nentity, nrelation, query_answers_dict):
        self.len = len(query_answers_dict)
        self.query_answers_dict = query_answers_dict
        self.nentity = nentity
        self.nrelation = nrelation


       
        self.train_answer_list = []
    


        self.query_type = None

        self.graph_unified_tokens = []
        self.features_RWPE = []
        self.features_LapPE = []


        for query, answer_list in query_answers_dict.items():
            
            self.train_answer_list.append([int(ans) for ans in answer_list["train_answers"]])
        
        
            self.graph_unified_tokens.append(answer_list["graph_unified_tokens"])
            self.features_RWPE.append(answer_list["features_RWPE"])
            self.features_LapPE.append(answer_list["features_LapPE"])
        
    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        unified_graph_id_list  = self.graph_unified_tokens[idx]
        graph_RWPE_list = self.features_RWPE[idx]
        graph_LapPE_list = self.features_LapPE[idx]

        answer_list = self.train_answer_list[idx]

        tail = np.random.choice(list(answer_list))

        positive_sample = int(tail)

    

        return unified_graph_id_list, graph_RWPE_list, graph_LapPE_list, positive_sample


    @staticmethod
    def collate_fn(data):

        # unified_graph_id_list = [_[0] for _ in data]
        # graph_RWPE_list = [_[1] for _ in data]
        # graph_LapPE_list = [_[2] for _ in data]
        # train_answers = [_[3] for _ in data]

        padding_token = std["[PAD]"]
        PE_size = len(data[0][1][0])
        max_query_length = max([len(_[0]) for _ in data])
        attention_mask = []

        unified_graph_id_list = []
        graph_RWPE_list = []
        graph_LapPE_list = []
        train_answers = []

        for unified_graph_id, graph_RWPE, graph_LapPE, train_answer in data:
            unified_graph_id_list.append(unified_graph_id + [padding_token for _ in range(max_query_length - len(unified_graph_id))])
            graph_RWPE_list.append(graph_RWPE + [[0.0 for _ in range(PE_size)] for _ in range(max_query_length - len(graph_RWPE))])
            graph_LapPE_list.append(graph_LapPE + [[0.0 for _ in range(PE_size)] for _ in range(max_query_length - len(graph_LapPE))])
            attention_mask.append([1 for _ in range(len(unified_graph_id))] + [0 for _ in range(max_query_length - len(unified_graph_id))])
            train_answers.append(train_answer)
       
       
        return unified_graph_id_list, graph_RWPE_list, graph_LapPE_list, train_answers, attention_mask


class TestDatasetDGL(Dataset):
    def __init__(self, nentity, nrelation, query_answers_dict):
        self.len = len(query_answers_dict)
        self.query_answers_dict = query_answers_dict
        self.nentity = nentity
        self.nrelation = nrelation


       
        self.train_answer_list = []
        self.valid_answer_list = []
        self.test_answer_list = []

        self.graph_list = []


        self.query_type = None

        self.graph_unified_tokens = []
        self.node_count = []

        self.head_list = []
        self.tail_list = []

        self.node_type_list = []
        self.edge_type_list = []


        for query, answer_list in query_answers_dict.items():
            
            self.train_answer_list.append([int(ans) for ans in answer_list["train_answers"]])
            self.valid_answer_list.append([int(ans) for ans in answer_list["valid_answers"]])
            self.test_answer_list.append([int(ans) for ans in answer_list["test_answers"]])

        
            self.graph_unified_tokens.append(answer_list["graph_unified_tokens"])
            self.node_count.append(len(answer_list["graph_unified_tokens"]))

            self.head_list.append(answer_list["head_list"])
            self.tail_list.append(answer_list["tail_list"])

            self.node_type_list.append(answer_list["node_type"])
            self.edge_type_list.append(answer_list["edge_type"])

            

        
    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        unified_graph_id_list  = self.graph_unified_tokens[idx]

        head_list = self.head_list[idx]
        tail_list = self.tail_list[idx]

        # dgl_graph = dgl.DGLGraph(torch.tensor([head_list, tail_list]))
        # dgl_graph = dgl.graph(torch.tensor(head_list), torch.tensor(tail_list))
        dgl_graph = dgl.graph((torch.tensor(head_list), torch.tensor(tail_list)))

        node_type_list = self.node_type_list[idx]
        edge_type_list = self.edge_type_list[idx]

        train_answer_list = self.train_answer_list[idx]
        valid_answer_list = self.valid_answer_list[idx]
        test_answer_list = self.test_answer_list[idx]
        

        return dgl_graph, unified_graph_id_list, head_list, tail_list, node_type_list, edge_type_list, train_answer_list, valid_answer_list, test_answer_list

    @staticmethod
    def collate_fn(data):

        
        dgl_graph, \
        unified_graph_id_list, \
        head_list, tail_list, node_type_list, edge_type_list, train_answer_list, valid_answer_list, test_answer_list  = map(list, zip(*data))

        dgl_graph = dgl.batch(dgl_graph)
    
        node_type_list = [node for node_list in node_type_list for node in node_list]
        edge_type_list = [edge for edge_list in edge_type_list for edge in edge_list]
        token_list = [token for token_list in unified_graph_id_list for token in token_list]

        graph_size = [len(_) for _ in unified_graph_id_list]
        

        return dgl_graph, graph_size, token_list, node_type_list, edge_type_list, train_answer_list, valid_answer_list, test_answer_list
    

class ValidDatasetDGL(Dataset):
    
    def __init__(self, nentity, nrelation, query_answers_dict):
        self.len = len(query_answers_dict)
        self.query_answers_dict = query_answers_dict
        self.nentity = nentity
        self.nrelation = nrelation


       
        self.train_answer_list = []
        self.valid_answer_list = []
       

        self.graph_list = []


        self.query_type = None

        self.graph_unified_tokens = []
        self.node_count = []

        self.head_list = []
        self.tail_list = []

        self.node_type_list = []
        self.edge_type_list = []


        for query, answer_list in query_answers_dict.items():
            
            self.train_answer_list.append([int(ans) for ans in answer_list["train_answers"]])
            self.valid_answer_list.append([int(ans) for ans in answer_list["valid_answers"]])
            
        
            self.graph_unified_tokens.append(answer_list["graph_unified_tokens"])
            self.node_count.append(len(answer_list["graph_unified_tokens"]))

            self.head_list.append(answer_list["head_list"])
            self.tail_list.append(answer_list["tail_list"])

            self.node_type_list.append(answer_list["node_type"])
            self.edge_type_list.append(answer_list["edge_type"])
        
    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        unified_graph_id_list  = self.graph_unified_tokens[idx]

        head_list = self.head_list[idx]
        tail_list = self.tail_list[idx]

        # dgl_graph = dgl.DGLGraph(torch.tensor([head_list, tail_list]))
        # dgl_graph = dgl.graph(torch.tensor(head_list), torch.tensor(tail_list))
        dgl_graph = dgl.graph((torch.tensor(head_list), torch.tensor(tail_list)))


        node_type_list = self.node_type_list[idx]
        edge_type_list = self.edge_type_list[idx]

        train_answer_list = self.train_answer_list[idx]
        valid_answer_list = self.valid_answer_list[idx]
       

        return dgl_graph, unified_graph_id_list, head_list, tail_list, node_type_list, edge_type_list, train_answer_list, valid_answer_list

    @staticmethod
    def collate_fn(data):

        
        dgl_graph, \
        unified_graph_id_list, \
        head_list, tail_list, node_type_list, edge_type_list, train_answer_list, valid_answer_list  = map(list, zip(*data))

        dgl_graph = dgl.batch(dgl_graph)

        node_type_list = [node for node_list in node_type_list for node in node_list]
        edge_type_list = [edge for edge_list in edge_type_list for edge in edge_list]
        token_list = [token for token_list in unified_graph_id_list for token in token_list]
        
        graph_size = [len(_) for _ in unified_graph_id_list]

        return dgl_graph, graph_size, token_list, node_type_list, edge_type_list, train_answer_list, valid_answer_list
    

class TrainDatasetDGL(Dataset):
    
    def __init__(self, nentity, nrelation, query_answers_dict):
        self.len = len(query_answers_dict)
        self.query_answers_dict = query_answers_dict
        self.nentity = nentity
        self.nrelation = nrelation


       
        self.train_answer_list = []
        self.valid_answer_list = []
       

        self.graph_list = []


        self.query_type = None

        self.graph_unified_tokens = []
        self.node_count = []

        self.head_list = []
        self.tail_list = []

        self.node_type_list = []
        self.edge_type_list = []


        for query, answer_list in query_answers_dict.items():
            
            self.train_answer_list.append([int(ans) for ans in answer_list["train_answers"]])
           
        
            self.graph_unified_tokens.append(answer_list["graph_unified_tokens"])
            self.node_count.append(len(answer_list["graph_unified_tokens"]))

            self.head_list.append(answer_list["head_list"])
            self.tail_list.append(answer_list["tail_list"])

            self.node_type_list.append(answer_list["node_type"])
            self.edge_type_list.append(answer_list["edge_type"])

        
    def __len__(self):
        return self.len

    def __getitem__(self, idx):


        unified_graph_id_list  = self.graph_unified_tokens[idx]

        head_list = self.head_list[idx]
        tail_list = self.tail_list[idx]

        dgl_graph = dgl.graph((torch.tensor(head_list), torch.tensor(tail_list)))


        node_type_list = self.node_type_list[idx]
        edge_type_list = self.edge_type_list[idx]

       
     
        answer_list = self.train_answer_list[idx]

        tail = np.random.choice(list(answer_list))

        positive_sample = int(tail)

    

        return dgl_graph, unified_graph_id_list, head_list, tail_list, node_type_list, edge_type_list, positive_sample


    @staticmethod
    def collate_fn(data):


        dgl_graph, \
        unified_graph_id_list, \
        head_list, tail_list, node_type_list, edge_type_list, train_answer_list  = map(list, zip(*data))

        dgl_graph = dgl.batch(dgl_graph)

        node_type_list = [node for node_list in node_type_list for node in node_list]
        edge_type_list = [edge for edge_list in edge_type_list for edge in edge_list]
        token_list = [token for token_list in unified_graph_id_list for token in token_list]
        
        graph_size = [len(_) for _ in unified_graph_id_list]

        
        return dgl_graph, graph_size, token_list, node_type_list, edge_type_list, train_answer_list
    

    

class TestDatasetTokenGT(Dataset):
    def __init__(self, nentity, nrelation, query_answers_dict):
        self.len = len(query_answers_dict)
        self.query_answers_dict = query_answers_dict
        self.nentity = nentity
        self.nrelation = nrelation

        self.max_length = 64
        self.max_session_length = 128
       
        self.train_answer_list = []
        self.valid_answer_list = []
        self.test_answer_list = []

        

        self.query_type = None

       

        self.unified_feature_ids = []

        self.node_identifiers = []
        self.type_identifiers = []
    
        self.attention_mask = []
        

        for query, answer_list in query_answers_dict.items():
            
            self.train_answer_list.append([int(ans) for ans in answer_list["train_answers"]])
            self.valid_answer_list.append([int(ans) for ans in answer_list["valid_answers"]])
            self.test_answer_list.append([int(ans) for ans in answer_list["test_answers"]])

            
            this_node_feature_token_id = answer_list["graph_unified_tokens"]
            this_edge_feature_token_id = [edge_id + self.nentity + 100 for edge_id in answer_list["edge_type"]]

            this_unified_feature_ids = this_node_feature_token_id + this_edge_feature_token_id

            
            this_type_identifiers = [0] * len(this_node_feature_token_id) + [1] * len(this_edge_feature_token_id)

            head_list = answer_list["head_list"]
            tail_list = answer_list["tail_list"]

            this_node_identitiers = [[i, i] for i in range(len(this_node_feature_token_id))] + \
                  [[head_list[i], tail_list[i]] for i in range(len(this_edge_feature_token_id))]
            

            attention_mask = [1] * len(this_unified_feature_ids)


            if len(this_unified_feature_ids) > self.max_length:
                this_unified_feature_ids = this_unified_feature_ids[:self.max_length]
                this_type_identifiers = this_type_identifiers[:self.max_length]
                this_node_identitiers = this_node_identitiers[:self.max_length]
                attention_mask = attention_mask[:self.max_length]
            
            else:
                this_unified_feature_ids = this_unified_feature_ids + [0] * (self.max_length - len(this_unified_feature_ids))
                this_type_identifiers = this_type_identifiers + [0] * (self.max_length - len(this_type_identifiers))
                this_node_identitiers = this_node_identitiers + [[0, 0]] * (self.max_length - len(this_node_identitiers))
                attention_mask = attention_mask + [0] * (self.max_length - len(attention_mask))
                

            self.unified_feature_ids.append(this_unified_feature_ids)
            self.node_identifiers.append(this_node_identitiers)
            self.type_identifiers.append(this_type_identifiers)
            self.attention_mask.append(attention_mask)
            

        
    def __len__(self):
        return self.len

    def __getitem__(self, idx):
       

        unified_feature_ids = self.unified_feature_ids[idx]
        node_identifiers = self.node_identifiers[idx]
        type_identifiers = self.type_identifiers[idx]
        attention_mask = self.attention_mask[idx]

        train_answer_list = self.train_answer_list[idx]
        valid_answer_list = self.valid_answer_list[idx]
        test_answer_list = self.test_answer_list[idx]
        

        return unified_feature_ids, node_identifiers, type_identifiers, attention_mask, train_answer_list, valid_answer_list, test_answer_list

    @staticmethod
    def collate_fn(data):

        
        unified_feature_ids, node_identifiers, type_identifiers, attention_mask, train_answer_list, valid_answer_list, test_answer_list  = map(list, zip(*data))

        return unified_feature_ids, node_identifiers, type_identifiers, attention_mask, train_answer_list, valid_answer_list, test_answer_list
    

class ValidDatasetTokenGT(Dataset):
    
    def __init__(self, nentity, nrelation, query_answers_dict):
        self.len = len(query_answers_dict)
        self.query_answers_dict = query_answers_dict
        self.nentity = nentity
        self.nrelation = nrelation


        self.max_length = 64
        self.max_session_length = 128
       
        self.train_answer_list = []
        self.valid_answer_list = []
       

        self.query_type = None

       

        self.unified_feature_ids = []

        self.node_identifiers = []
        self.type_identifiers = []
    
        self.attention_mask = []
        

        for query, answer_list in query_answers_dict.items():
            
            self.train_answer_list.append([int(ans) for ans in answer_list["train_answers"]])
            self.valid_answer_list.append([int(ans) for ans in answer_list["valid_answers"]])
           
            
            this_node_feature_token_id = answer_list["graph_unified_tokens"]
            this_edge_feature_token_id = [edge_id + self.nentity + 100 for edge_id in answer_list["edge_type"]]

            this_unified_feature_ids = this_node_feature_token_id + this_edge_feature_token_id

            
            this_type_identifiers = [0] * len(this_node_feature_token_id) + [1] * len(this_edge_feature_token_id)

            head_list = answer_list["head_list"]
            tail_list = answer_list["tail_list"]

            this_node_identitiers = [[i, i] for i in range(len(this_node_feature_token_id))] + \
                  [[head_list[i], tail_list[i]] for i in range(len(this_edge_feature_token_id))]
            

            attention_mask = [1] * len(this_unified_feature_ids)


            if len(this_unified_feature_ids) > self.max_length:
                this_unified_feature_ids = this_unified_feature_ids[:self.max_length]
                this_type_identifiers = this_type_identifiers[:self.max_length]
                this_node_identitiers = this_node_identitiers[:self.max_length]
                attention_mask = attention_mask[:self.max_length]
            
            else:
                this_unified_feature_ids = this_unified_feature_ids + [0] * (self.max_length - len(this_unified_feature_ids))
                this_type_identifiers = this_type_identifiers + [0] * (self.max_length - len(this_type_identifiers))
                this_node_identitiers = this_node_identitiers + [[0, 0]] * (self.max_length - len(this_node_identitiers))
                attention_mask = attention_mask + [0] * (self.max_length - len(attention_mask))
                # this_unified_feature_ids = this_unified_feature_ids + [0 for _ in range(self.max_length - len(this_unified_feature_ids))]
                # this_type_identifiers = this_type_identifiers + [0 for _ in range(self.max_length - len(this_type_identifiers))]
                # this_node_identitiers = this_node_identitiers + [[0, 0] for _ in range(self.max_length - len(this_node_identitiers))]
                # attention_mask = attention_mask + [0 for _ in range(self.max_length - len(attention_mask))]


            self.unified_feature_ids.append(this_unified_feature_ids)
            self.node_identifiers.append(this_node_identitiers)
            self.type_identifiers.append(this_type_identifiers)
            self.attention_mask.append(attention_mask)
        
    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        unified_feature_ids = self.unified_feature_ids[idx]
        node_identifiers = self.node_identifiers[idx]
        type_identifiers = self.type_identifiers[idx]
        attention_mask = self.attention_mask[idx]

        train_answer_list = self.train_answer_list[idx]
        valid_answer_list = self.valid_answer_list[idx]
        

        return unified_feature_ids, node_identifiers, type_identifiers, attention_mask, train_answer_list, valid_answer_list

       

    @staticmethod
    def collate_fn(data):
        
        unified_feature_ids, node_identifiers, type_identifiers, attention_mask, train_answer_list, valid_answer_list  = map(list, zip(*data))
        return unified_feature_ids, node_identifiers, type_identifiers, attention_mask, train_answer_list, valid_answer_list
    
    

class TrainDatasetTokenGT(Dataset):
    
    def __init__(self, nentity, nrelation, query_answers_dict):
        self.len = len(query_answers_dict)
        self.query_answers_dict = query_answers_dict
        self.nentity = nentity
        self.nrelation = nrelation

        self.max_length = 64
        self.max_session_length = 128
       
        self.train_answer_list = []
    
        self.query_type = None

       

        self.unified_feature_ids = []

        self.node_identifiers = []
        self.type_identifiers = []
    
        self.attention_mask = []
        

        for query, answer_list in query_answers_dict.items():
            
            self.train_answer_list.append([int(ans) for ans in answer_list["train_answers"]])
           
            this_node_feature_token_id = answer_list["graph_unified_tokens"]
            this_edge_feature_token_id = [edge_id + self.nentity + 100 for edge_id in answer_list["edge_type"]]

            this_unified_feature_ids = this_node_feature_token_id + this_edge_feature_token_id

            
            this_type_identifiers = [0] * len(this_node_feature_token_id) + [1] * len(this_edge_feature_token_id)

            head_list = answer_list["head_list"]
            tail_list = answer_list["tail_list"]

            this_node_identitiers = [[i, i] for i in range(len(this_node_feature_token_id))] + \
                  [[head_list[i], tail_list[i]] for i in range(len(this_edge_feature_token_id))]
            

            attention_mask = [1] * len(this_unified_feature_ids)


            if len(this_unified_feature_ids) > self.max_length:
                this_unified_feature_ids = this_unified_feature_ids[:self.max_length]
                this_type_identifiers = this_type_identifiers[:self.max_length]
                this_node_identitiers = this_node_identitiers[:self.max_length]
                attention_mask = attention_mask[:self.max_length]
            
            else:
                this_unified_feature_ids = this_unified_feature_ids + [0] * (self.max_length - len(this_unified_feature_ids))
                this_type_identifiers = this_type_identifiers + [0] * (self.max_length - len(this_type_identifiers))
                this_node_identitiers = this_node_identitiers + [[0, 0]] * (self.max_length - len(this_node_identitiers))
                attention_mask = attention_mask + [0] * (self.max_length - len(attention_mask))
                # this_unified_feature_ids = this_unified_feature_ids + [0 for _ in range(self.max_length - len(this_unified_feature_ids))]
                # this_type_identifiers = this_type_identifiers + [0 for _ in range(self.max_length - len(this_type_identifiers))]
                # this_node_identitiers = this_node_identitiers + [[0, 0] for _ in range(self.max_length - len(this_node_identitiers))]
                # attention_mask = attention_mask + [0 for _ in range(self.max_length - len(attention_mask))]

            self.unified_feature_ids.append(this_unified_feature_ids)
            self.node_identifiers.append(this_node_identitiers)
            self.type_identifiers.append(this_type_identifiers)
            self.attention_mask.append(attention_mask)
            


        
    def __len__(self):
        return self.len

    def __getitem__(self, idx):


        unified_feature_ids = self.unified_feature_ids[idx]
        node_identifiers = self.node_identifiers[idx]
        type_identifiers = self.type_identifiers[idx]
        attention_mask = self.attention_mask[idx]

        answer_list = self.train_answer_list[idx]
        

        tail = np.random.choice(list(answer_list))

        positive_sample = int(tail)

    

        return unified_feature_ids, node_identifiers, type_identifiers, attention_mask, positive_sample


    @staticmethod
    def collate_fn(data):

        
        unified_feature_ids, node_identifiers, type_identifiers, attention_mask, positive_sample = map(list, zip(*data))

 

        return unified_feature_ids, node_identifiers, type_identifiers, attention_mask, positive_sample
    



class SingledirectionalOneShotIterator(object):
    def __init__(self, dataloader):
        self.iterator = self.one_shot_iterator(dataloader)
        self.step = 0
        self.len = len(dataloader)

    def __next__(self):
        self.step += 1
        data = next(self.iterator)
        return data

    @staticmethod
    def one_shot_iterator(dataloader):
        while True:
            for data in dataloader:
                yield data

def test_graph_pe_dataloader():

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

    batch_size = 5
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
        unified_graph_id_list, graph_RWPE_list, graph_LapPE_list, train_answers, attention_mask = next(iterator)
        print(unified_graph_id_list)
        print(graph_RWPE_list)
        print(graph_LapPE_list)
        print(train_answers)
        print(attention_mask)


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
        for unified_graph_id_list, graph_RWPE_list, graph_LapPE_list, train_answers, valid_answers, attention_mask in loader:
            print(unified_graph_id_list)
            print(graph_RWPE_list)
            print(graph_LapPE_list)
            print([len(_) for _ in train_answers])
            print([len(_) for _ in valid_answers])
            print(attention_mask)
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
        for  unified_graph_id_list, graph_RWPE_list, graph_LapPE_list,  train_answers, valid_answers, test_answers, attention_mask in loader:
            print(unified_graph_id_list)
            print(graph_RWPE_list)
            print(graph_LapPE_list)

            print([len(_) for _ in train_answers])
            print([len(_) for _ in valid_answers])
            print([len(_) for _ in test_answers])
            print(attention_mask)

            break

def test_original_dataloader():

    train_data_path = "../sampled_hyper_train_merged_old/amazon_train_queries_0.json"
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

    batch_size = 5
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
        for  batched_query, unified_ids, train_answers, valid_answers, test_answers in loader:
            print(batched_query)
            print(unified_ids)
            print([len(_) for _ in train_answers])
            print([len(_) for _ in valid_answers])
            print([len(_) for _ in test_answers])

            break

def test_graph_dgl_dataloader():

    train_data_path = "../sampled_hyper_train_merged_graph_dgl/amazon_train_queries_dgl.json"
    valid_data_path = "../sampled_hyper_valid_graph_dgl/amazon_valid_queries_25_dgl.json"
    test_data_path = "../sampled_hyper_test_graph_dgl/amazon_test_queries_15_dgl.json"
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

    batch_size = 5
    train_iterators = {}
    for query_type, query_answer_dict in train_data_dict.items():
        print("====================================")
        print(query_type)

        new_iterator = SingledirectionalOneShotIterator(DataLoader(
            TrainDatasetDGL(nentity, nrelation, query_answer_dict),
            batch_size=batch_size,
            shuffle=True,
            collate_fn=TrainDatasetDGL.collate_fn
        ))
        train_iterators[query_type] = new_iterator

    for key, iterator in train_iterators.items():
        print("read ", key)
        dgl_graph, graph_size, token_list, node_type_list, edge_type_list, positive_sample = next(iterator)
        print(dgl_graph)
        print(graph_size)

        # print(unified_graph_id_list)
        # print(head_list)
        # print(tail_list)
        print(node_type_list)
        print(edge_type_list)
        print(positive_sample)
        
      


    validation_loaders = {}
    for query_type, query_answer_dict in valid_data_dict.items():
        print("====================================")
        print(query_type)

        new_iterator = DataLoader(
            ValidDatasetDGL(nentity, nrelation, query_answer_dict),
            batch_size=batch_size,
            shuffle=True,
            collate_fn=ValidDatasetDGL.collate_fn
        )
        validation_loaders[query_type] = new_iterator

    for key, loader in validation_loaders.items():
        print("read ", key)
        for  dgl_graph, graph_size, token_list, node_type_list, edge_type_list, train_answer_list, valid_answer_list in loader:
            print(dgl_graph)
            print(graph_size)
            print(token_list)
            print(node_type_list)
            print(edge_type_list)
            print([len(_) for _ in train_answer_list])
            print([len(_) for _ in valid_answer_list])
          
            break

    test_loaders = {}
    for query_type, query_answer_dict in test_data_dict.items():
        print("====================================")
        print(query_type)

        new_loader = DataLoader(
            TestDatasetDGL(nentity, nrelation, query_answer_dict),
            batch_size=batch_size,
            shuffle=True,
            collate_fn=TestDatasetDGL.collate_fn
        )
        test_loaders[query_type] = new_loader

    for key, loader in test_loaders.items():
        print("read ", key)
        for dgl_graph, graph_size, token_list, node_type_list, edge_type_list, train_answer_list, valid_answer_list, test_answer_list in loader:
            print(dgl_graph)
            print(graph_size)
            print(token_list)
            print(node_type_list)
            print(edge_type_list)
            print([len(_) for _ in train_answer_list])
            print([len(_) for _ in valid_answer_list])
            print([len(_) for _ in test_answer_list])

            break

def test_graph_token_gt_dataloader():

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

    batch_size = 5
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
      
        print(type_identifiers)
        print(attention_mask)
        print(positive_sample)
        
      


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
        for  unified_feature_ids, node_identifiers, type_identifiers, attention_mask, train_answer_list, valid_answer_list in loader:
            print(unified_feature_ids)
            print(node_identifiers)
            print(type_identifiers)
            print(attention_mask)
            print([len(_) for _ in train_answer_list])
            print([len(_) for _ in valid_answer_list])
          
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
            print(type_identifiers)
            print(attention_mask)
            print([len(_) for _ in train_answer_list])
            print([len(_) for _ in valid_answer_list])
            print([len(_) for _ in test_answer_list])

            break


if __name__ == "__main__":
    # test_original_dataloader()

    test_graph_token_gt_dataloader()



