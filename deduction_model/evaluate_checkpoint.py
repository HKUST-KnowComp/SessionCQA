import argparse
from gqe import GQE
from q2b import Q2B
from q2p import Q2P
from betae import BetaE
from cone import ConE
from transformer import TransformerModel
from biqe import BiQEModel
from rnn import RNNModel
from gru import GRUModel
from lstm import LSTMModel
from tree_lstm import TreeLSTM
from tree_rnn import TreeRNN
from tcn import TCNModel
from hype import HypE
from hype_util import RiemannianAdam
from mlp import MLPMixerReasoner, MLPReasoner
from fuzzqe import FuzzQE

from graph_transformer import GTModel
from shgt import SHGT, HGT, RGCN
from tokenGT import TokenGTModel

from nqe import NQE

import torch
from dataloader import TrainDataset, ValidDataset, TestDataset, SingledirectionalOneShotIterator, TrainDatasetPE, ValidDatasetPE, TestDatasetPE
from dataloader import TrainDatasetDGL, ValidDatasetDGL, TestDatasetDGL
from dataloader import TrainDatasetTokenGT, ValidDatasetTokenGT, TestDatasetTokenGT

from torch.utils.data import DataLoader
import os
from tqdm import tqdm
import numpy as np
from datetime import datetime
from tensorboardX import SummaryWriter
import gc
import pickle
from torch.optim.lr_scheduler import LambdaLR
import pandas as pd
import json

import csv

def log_aggregation(list_of_logs):
    all_log = {}

    for __log in list_of_logs:
        # Sometimes the number of answers are 0, so we need to remove all the keys with 0 values
        # The average is taken over all queries, instead of over all answers, as is done following previous work. 
        ignore_exd = False
        ignore_ent = False
        ignore_inf = False

        if "exd_num_answers" in __log and __log["exd_num_answers"] == 0:
            ignore_exd = True
        if "ent_num_answers" in __log and __log["ent_num_answers"] == 0:
            ignore_ent = True
        if "inf_num_answers" in __log and __log["inf_num_answers"] == 0:
            ignore_inf = True
            
        
        for __key, __value in __log.items():
            if "num_answers" in __key:
                continue

            else:
                if ignore_ent and "ent_" in __key:
                    continue
                if ignore_exd and "exd_" in __key:
                    continue
                if ignore_inf and "inf_" in __key:
                    continue

                if __key in all_log:
                    all_log[__key].append(__value)
                else:
                    all_log[__key] = [__value]

    average_log = {_key: np.mean(_value) for _key, _value in all_log.items()}

    return average_log



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='The training and evaluation script for the models')

    parser.add_argument("--train_query_dir", required=True)
    parser.add_argument("--valid_query_dir", required=True)
    parser.add_argument("--test_query_dir", required=True)
    parser.add_argument('--kg_data_dir', default="graph_data_en/", help="The path the original kg data")

    parser.add_argument('--num_layers', default = 2, type=int, help="num of layers for sequential models")

    parser.add_argument('--log_steps', default=50000, type=int, help='train log every xx steps')
    parser.add_argument('-dn', '--data_name', type=str, required=True)
    parser.add_argument('-b', '--batch_size', default=64, type=int)
    parser.add_argument('-eb', '--evaluation_batch_size', default=1, type=int)

    parser.add_argument('-d', '--entity_space_dim', default=384, type=int)
    parser.add_argument('-lr', '--learning_rate', default=0.002, type=float)
    parser.add_argument('-wc', '--weight_decay', default=0.0000, type=float)

    parser.add_argument("--optimizer", default="adam", type=str)
    parser.add_argument("--dropout_rate", default=0.1, type=float)
    parser.add_argument('-ls', "--label_smoothing", default=0.0, type=float)

    parser.add_argument("--warm_up_steps", default=1000, type=int)

    parser.add_argument("-m", "--model", required=True)

    parser.add_argument("--checkpoint_path", type=str, default="./logs")
    parser.add_argument("-old", "--old_loss_fnt", action="store_true")
    parser.add_argument("-fol", "--use_full_fol", action="store_true")
    parser.add_argument("-ga", "--gradient_accumulation_steps", type=int, default=1)

    parser.add_argument("--session_encoder", type=str, default="AttnMixer")

    parser.add_argument("--few_shot", type=int, default=32)

    parser.add_argument("--pretrained_embedding", action="store_true")

    parser.add_argument("--evaluation_checkpoint", type=str, required=True)

    parser.add_argument("--no_logic", action="store_true")
    parser.add_argument("--no_order", action="store_true")

    args = parser.parse_args()

    KG_data_path = "./" + args.kg_data_dir
    data_name = args.data_name
    
  
   
    valid_query_file_names = []
    test_query_file_names = []


    
    for file in os.listdir(args.valid_query_dir):
        if file.endswith(".json"):
            valid_query_file_names.append(file)
    
    for file in os.listdir(args.test_query_dir):
        if file.endswith(".json"):
            test_query_file_names.append(file)
    test_query_file_names = sorted(test_query_file_names)


    # test_query_file_names = test_query_file_names[:1]

    print("valid_query_file_names: ", valid_query_file_names)
    print("test_query_file_names: ", test_query_file_names)

    data_dir = "./" + args.data_name
    asin_ids = json.load(open(data_dir + "/asin_ids.json", "r"))
    value_ids = json.load(open(data_dir + "/value_ids.json", "r"))
    session_dict = json.load(open(data_dir + "/session_dict.json", "r"))

    id2node = json.load(open(data_dir + "/id2node.json", "r"))
    node2id = json.load(open(data_dir + "/node2id.json", "r"))
    id2relation = json.load(open(data_dir + "/id2relation.json", "r"))
    relation2id = json.load(open(data_dir + "/relation2id.json", "r"))

    if args.pretrained_embedding:
        pretrained_weights = pickle.load(open(data_dir + "/nodeid_to_semantic_embeddings.pkl", "rb"))
    
    else:
        pretrained_weights = None


    session_encoder = args.session_encoder
   
    
    fol_type = "pinu" 
    

    if args.model == "gqe" or args.model == "q2b" or args.model=="hype":
        fol_type = "pi"
        
    loss = "new-loss"
    
    info = fol_type 
    if args.model != "nqe":
        info += "_" + session_encoder

    
    nentity = len(id2node)
    nrelation = len(id2relation)
    max_length = max([len(_) for _ in session_dict.values()])

    print("nentity: ", nentity)
    print("nrelation: ", nrelation)
    print("max_length: ", max_length)

    

    batch_size = args.batch_size

    first_round_query_types = {
        "1p": "(p,(s))",
        "1pA": "(p,(a))",
        "2p": "(p,(p,(s)))",
        "2iA": "(i,(p,(s)),(p,(a)))",
        "2iS": "(i,(p,(s)),(p,(s)))",
        "3i": "(i,(p,(s)),(p,(s)),(p,(a)))",
        "ip": "(p,(i,(p,(s)),(p,(s))))",
        "pi": "(i,(p,(p,(s))),(p,(e)))",
        "2uA": "(u,(p,(s)),(p,(a)))",
        "2uS": "(u,(p,(s)),(p,(s)))",
        "up": "(p,(u,(p,(s)),(p,(s))))",
        "2inA": "(i,(n,(p,(a))),(p,(s)))",
        "2inS": "(i,(n,(p,(s))),(p,(s)))",
        "3in": "(i,(n,(p,(s))),(p,(a)),(p,(s)))",
        "inp": "(p,(i,(n,(p,(s))),(p,(s))))",
        "pin": "(i,(p,(p,(s))),(n,(p,(e))))",
    }

    zero_shot_types = {
        "3ip": "(p,(i,(p,(s)),(p,(s)),(p,(a))))",
        "3inp": "(p,(i,(n,(p,(s))),(p,(s)),(p,(a))))",
        "3iA": "(i,(p,(s)),(p,(a)),(p,(a)))",
        "3inA": "(i,(n,(p,(a))),(p,(a)),(p,(s)))",
    }




    evaluating_query_types = list(zero_shot_types.values())
    print("Evaluating query types: ", evaluating_query_types)

    training_query_types = list(zero_shot_types.values())
    print("Training query types: ", training_query_types)


    # create model
    print("====== Initialize Model ======", args.model)
   

    if args.model == 'q2p':
        model = Q2P(num_entities=nentity, num_relations=nrelation, embedding_size=args.entity_space_dim, session_encoder=session_encoder, label_smoothing=args.label_smoothing, pretrained_weights=pretrained_weights)
    
    elif args.model == "fuzzqe":
        model = FuzzQE(num_entities=nentity, num_relations=nrelation, embedding_size=args.entity_space_dim, session_encoder=session_encoder, label_smoothing=args.label_smoothing, pretrained_weights=pretrained_weights)

    elif args.model == "mlp":
        model =  MLPReasoner(num_entities=nentity, num_relations=nrelation, embedding_size=args.entity_space_dim, session_encoder=session_encoder, label_smoothing=args.label_smoothing, pretrained_weights=pretrained_weights)
    
    elif args.model == "mlp_mixer":
        model = MLPMixerReasoner(num_entities=nentity, num_relations=nrelation, embedding_size=args.entity_space_dim, session_encoder=session_encoder, label_smoothing=args.label_smoothing, pretrained_weights=pretrained_weights)

    elif args.model == "tree_lstm":
        model = TreeLSTM(num_entities=nentity, num_relations=nrelation, embedding_size=args.entity_space_dim, session_encoder=session_encoder, label_smoothing=args.label_smoothing, pretrained_weights=pretrained_weights)

    elif args.model == "nqe":
        model = NQE(num_entities=nentity, num_relations=nrelation, embedding_size=args.entity_space_dim, label_smoothing=args.label_smoothing, pretrained_weights=pretrained_weights)
    
    elif args.model == "transformer":
        model = TransformerModel(num_entities=nentity, num_relations=nrelation, embedding_size=args.entity_space_dim ,num_layers = args.num_layers, label_smoothing=args.label_smoothing, pretrained_weights=pretrained_weights)
    
    elif args.model == "lstm":
        model = LSTMModel(num_entities=nentity, num_relations=nrelation, embedding_size=args.entity_space_dim, num_layers = args.num_layers, label_smoothing=args.label_smoothing, pretrained_weights=pretrained_weights)

    elif args.model == "gt":
        model = GTModel(num_entities=nentity, num_relations=nrelation, embedding_size=args.entity_space_dim, num_layers = args.num_layers, label_smoothing=args.label_smoothing, pretrained_weights=pretrained_weights)
    
    elif args.model == "shgt":
        model = SHGT(num_entities=nentity, num_relations=nrelation, embedding_size=args.entity_space_dim, num_layers = args.num_layers, label_smoothing=args.label_smoothing)
    
    elif args.model == "hgt":
        model = HGT(num_entities=nentity, num_relations=nrelation, embedding_size=args.entity_space_dim, num_layers = args.num_layers, label_smoothing=args.label_smoothing)
    
    elif args.model == "rgcn":
        model = RGCN(num_entities=nentity, num_relations=nrelation, embedding_size=args.entity_space_dim, num_layers = args.num_layers, label_smoothing=args.label_smoothing)
    
    elif args.model == "tokengt":
        model = TokenGTModel(num_entities=nentity, num_relations=nrelation, embedding_size=args.entity_space_dim, 
            num_layers = args.num_layers, label_smoothing=args.label_smoothing, no_logic = args.no_logic, no_order = args.no_order
        )



    PATH = args.evaluation_checkpoint

    print("loading model from ", PATH)
    checkpoint = torch.load(PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    step = checkpoint['steps']
    
    print("step: ", step)
    
    if torch.cuda.is_available():
        model.cuda()
    
    model.eval()

    model_name = args.model + "_" + session_encoder + "_" + fol_type
    
    print("====== Test ======", model_name, step)
    entailment_58_types_logs = []
    entailment_29_types_logs = []
    entailment_unseen_29_types_logs = []

    generalization_58_types_logs = []
    generalization_29_types_logs = []
    generalization_unseen_29_types_logs = []

    entailment_58_types_dict = {}
    generalization_58_types_dict = {}

    for test_query_file_name in tqdm(test_query_file_names):
        with open(args.test_query_dir + "/" + test_query_file_name , "r") as fin:
            test_data_dict = json.load(fin)
        

        test_loaders = {}
        for query_type, query_answer_dict in test_data_dict.items():

            if args.model == "gqe" or args.model == "q2b" or args.model=="hype":
                if "u" in query_type or "n" in query_type:
                    continue

            if args.use_full_fol == False:
                if "u" in query_type or "n" in query_type:
                    continue

            # new_iterator = DataLoader(
            #     TestDataset(nentity, nrelation, query_answer_dict, max_length),
            #     batch_size=args.evaluation_batch_size,
            #     shuffle=True,
            #     collate_fn=TestDataset.collate_fn
            # )
            # test_loaders[query_type] = new_iterator

            if args.model == "gt":
                new_iterator = DataLoader(
                    TestDatasetPE(nentity, nrelation, query_answer_dict),
                    batch_size=args.evaluation_batch_size,
                    shuffle=True,
                    collate_fn=TestDatasetPE.collate_fn
                )
            
            elif args.model == "shgt" or args.model == "hgt" or args.model == "rgcn":
                new_iterator = DataLoader(
                    TestDatasetDGL(nentity, nrelation, query_answer_dict),
                    batch_size=args.evaluation_batch_size,
                    shuffle=True,
                    collate_fn=TestDatasetDGL.collate_fn
                )
            
            elif args.model == "tokengt":
                new_iterator = DataLoader(
                    TestDatasetTokenGT(nentity, nrelation, query_answer_dict),
                    batch_size=args.evaluation_batch_size,
                    shuffle=True,
                    collate_fn=TestDatasetTokenGT.collate_fn
                )

            else:
                new_iterator = DataLoader(
                    TestDataset(nentity, nrelation, query_answer_dict, max_length),
                    batch_size=args.evaluation_batch_size,
                    shuffle=True,
                    collate_fn=TestDataset.collate_fn
                )
            test_loaders[query_type] = new_iterator
        

        for task_name, loader in test_loaders.items():

            all_entailment_logs = []
            all_generalization_logs = []

            if args.model == "gt":
                for unified_graph_id_list, graph_RWPE_list, graph_LapPE_list,  train_answers, valid_answers, test_answers, attention_mask in loader:
                    
                    if args.positional_encoding_type == "rwpe":
                        query_embedding = model(unified_graph_id_list, attention_mask, graph_RWPE_list)
                    elif args.positional_encoding_type == "lappe":
                        query_embedding = model(unified_graph_id_list, attention_mask, graph_LapPE_list)
                    entailment_logs = model.evaluate_entailment(query_embedding, train_answers)
                    generalization_logs = model.evaluate_generalization(query_embedding, valid_answers, test_answers)

                    all_entailment_logs.extend(entailment_logs)
                    all_generalization_logs.extend(generalization_logs)

                    if task_name in evaluating_query_types:
                        entailment_58_types_logs.extend(entailment_logs)
                        generalization_58_types_logs.extend(generalization_logs)

                    if task_name in training_query_types:
                        entailment_29_types_logs.extend(entailment_logs)
                        generalization_29_types_logs.extend(generalization_logs)

                    if task_name in evaluating_query_types and task_name not in training_query_types:
                        entailment_unseen_29_types_logs.extend(entailment_logs)
                        generalization_unseen_29_types_logs.extend(generalization_logs)
            
            elif args.model == "shgt" or args.model == "hgt" or args.model == "rgcn":
                for dgl_graph, graph_size, token_list, node_type_list, edge_type_list, train_answers, valid_answers, test_answers in loader:
                    query_embedding = model(dgl_graph, token_list, node_type_list, edge_type_list, graph_size)

                    entailment_logs = model.evaluate_entailment(query_embedding, train_answers)
                    generalization_logs = model.evaluate_generalization(query_embedding, valid_answers, test_answers)

                    all_entailment_logs.extend(entailment_logs)
                    all_generalization_logs.extend(generalization_logs)

                    if task_name in evaluating_query_types:
                        entailment_58_types_logs.extend(entailment_logs)
                        generalization_58_types_logs.extend(generalization_logs)

                    if task_name in training_query_types:
                        entailment_29_types_logs.extend(entailment_logs)
                        generalization_29_types_logs.extend(generalization_logs)

                    if task_name in evaluating_query_types and task_name not in training_query_types:
                        entailment_unseen_29_types_logs.extend(entailment_logs)
                        generalization_unseen_29_types_logs.extend(generalization_logs)
            
            elif args.model == "tokengt":
                for unified_feature_ids, node_identifiers, type_identifiers, attention_mask, train_answers, valid_answers, test_answers in loader:
                    query_embedding = model(unified_feature_ids, node_identifiers, type_identifiers, attention_mask)

                    entailment_logs = model.evaluate_entailment(query_embedding, train_answers)
                    generalization_logs = model.evaluate_generalization(query_embedding, valid_answers, test_answers)

                    all_entailment_logs.extend(entailment_logs)
                    all_generalization_logs.extend(generalization_logs)

                    if task_name in evaluating_query_types:
                        entailment_58_types_logs.extend(entailment_logs)
                        generalization_58_types_logs.extend(generalization_logs)

                    if task_name in training_query_types:
                        entailment_29_types_logs.extend(entailment_logs)
                        generalization_29_types_logs.extend(generalization_logs)

                    if task_name in evaluating_query_types and task_name not in training_query_types:
                        entailment_unseen_29_types_logs.extend(entailment_logs)
                        generalization_unseen_29_types_logs.extend(generalization_logs)

            else:

                for batched_query, unified_ids, train_answers, valid_answers, test_answers in loader:

                    if args.model == "lstm" or args.model == "transformer" or args.model == "tcn" or args.model == "rnn" or args.model == "gru" or args.model == "biqe":
                        batched_query = unified_ids

                    query_embedding = model(batched_query)
                    entailment_logs = model.evaluate_entailment(query_embedding, train_answers)
                    generalization_logs = model.evaluate_generalization(query_embedding, valid_answers, test_answers)

                    all_entailment_logs.extend(entailment_logs)
                    all_generalization_logs.extend(generalization_logs)

                    if task_name in evaluating_query_types:
                        entailment_58_types_logs.extend(entailment_logs)
                        generalization_58_types_logs.extend(generalization_logs)

                    if task_name in training_query_types:
                        entailment_29_types_logs.extend(entailment_logs)
                        generalization_29_types_logs.extend(generalization_logs)

                    if task_name in evaluating_query_types and task_name not in training_query_types:
                        entailment_unseen_29_types_logs.extend(entailment_logs)
                        generalization_unseen_29_types_logs.extend(generalization_logs)
            if task_name not in entailment_58_types_dict:
                entailment_58_types_dict[task_name] = []
            entailment_58_types_dict[task_name].extend(all_entailment_logs)


            if task_name not in generalization_58_types_dict:
                generalization_58_types_dict[task_name] = []
            generalization_58_types_dict[task_name].extend(all_generalization_logs)
        
    

    checkpoint_name = args.evaluation_checkpoint.split("/")[-1]
    if args.no_order:
        checkpoint_name += "_no_order"
    if args.no_logic:
        checkpoint_name += "_no_logic"

    with open("./evaluation_csv/" + checkpoint_name + '.csv', mode='w', newline='') as file:
        writer = csv.writer(file)

        writer.writerow(['task_name', 'setting', 'metric', 'value'])
    
        for task_name, logs in entailment_58_types_dict.items():
            aggregated_entailment_logs = log_aggregation(logs)
            for key, value in aggregated_entailment_logs.items():
                writer.writerow([task_name, 'entailment',  key, value])
                # test_summary_writer.add_scalar("z-test-" + task_name + "-" + key, value)
        
        for task_name, logs in generalization_58_types_dict.items():
            aggregated_generalization_logs = log_aggregation(logs)
            for key, value in aggregated_generalization_logs.items():
                writer.writerow([task_name, 'generalization',  key, value])
                

    
    # entailment_58_types_logs = log_aggregation(entailment_58_types_logs)
    # generalization_58_types_logs = log_aggregation(generalization_58_types_logs)
    # entailment_29_types_logs = log_aggregation(entailment_29_types_logs)
    # generalization_29_types_logs = log_aggregation(generalization_29_types_logs)
    # entailment_unseen_29_types_logs = log_aggregation(entailment_unseen_29_types_logs)
    # generalization_unseen_29_types_logs = log_aggregation(generalization_unseen_29_types_logs)

    # for key, value in entailment_58_types_logs.items():
    #     test_summary_writer.add_scalar("x-test-58-types-" + key, value, global_steps)

    # for key, value in generalization_58_types_logs.items():
    #     test_summary_writer.add_scalar("x-test-58-types-" + key, value, global_steps)

    # for key, value in entailment_29_types_logs.items():
    #     test_summary_writer.add_scalar("x-test-29-types-" + key, value, global_steps)

    # for key, value in generalization_29_types_logs.items():
    #     test_summary_writer.add_scalar("x-test-29-types-" + key, value, global_steps)

    # for key, value in entailment_unseen_29_types_logs.items():
    #     test_summary_writer.add_scalar("x-test-unseen-29-types-" + key, value, global_steps)

    # for key, value in generalization_unseen_29_types_logs.items():
    #     test_summary_writer.add_scalar("x-test-unseen-29-types-" + key, value, global_steps)


    gc.collect()

