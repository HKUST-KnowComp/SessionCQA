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

    parser.add_argument("--warm_up_steps", default=10000, type=int)

    parser.add_argument("-m", "--model", required=True)

    parser.add_argument("--checkpoint_path", type=str, default="./logs")
    parser.add_argument("-old", "--old_loss_fnt", action="store_true")
    parser.add_argument("-fol", "--use_full_fol", action="store_true")
    parser.add_argument("-ga", "--gradient_accumulation_steps", type=int, default=1)

    parser.add_argument("--session_encoder", type=str, default="AttnMixer")

    parser.add_argument("--few_shot", type=int, default=32)

    parser.add_argument("--pretrained_embedding", action="store_true")
    
    parser.add_argument("--positional_encoding_type", type=str, default="rwpe")

    args = parser.parse_args()

    
    data_name = args.data_name

    scaler = torch.cuda.amp.GradScaler()
    
  
    train_query_file_names = []
    valid_query_file_names = []
    test_query_file_names = []

    for file in os.listdir(args.train_query_dir):
        if file.endswith(".json"):
            train_query_file_names.append(file)

    for file in os.listdir(args.valid_query_dir):
        if file.endswith(".json"):
            valid_query_file_names.append(file)
    
    for file in os.listdir(args.test_query_dir):
        if file.endswith(".json"):
            test_query_file_names.append(file)

    print("train_query_file_names: ", train_query_file_names)
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


    if args.model != "nqe" and args.model != "gt" and args.model != "transformer" and args.model != "lstm" and args.model != "shgt" and args.model != "hgt" and args.model != "rgcn" and args.model != "tokengt":
        info += "_" + session_encoder
    

    if args.model == "shgt" or args.model == "hgt" or args.model == "rgcn":
        info += "_layers_" + str(args.num_layers)
    

    if args.model == "gt":
        info += "_" + args.positional_encoding_type

    
    nentity = len(id2node)
    nrelation = len(id2relation)
    max_length = max([len(_) for _ in session_dict.values()])

    print("nentity: ", nentity)
    print("nrelation: ", nrelation)
    print("max_length: ", max_length)

    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = args.checkpoint_path + '/gradient_tape/' + current_time + "_" + args.model + "_" + info + "_" + data_name + '/train'
    test_log_dir = args.checkpoint_path  + '/gradient_tape/' + current_time + "_" + args.model + "_" + info + "_" + data_name + '/test'
    train_summary_writer = SummaryWriter(train_log_dir)
    test_summary_writer = SummaryWriter(test_log_dir)

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
        "pin": "(i,(p,(p,(s))),(n,(p,(e))))"
    }



    evaluating_query_types = list(first_round_query_types.values())
    print("Evaluating query types: ", evaluating_query_types)

    training_query_types = list(first_round_query_types.values())
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
    
    elif args.model == "betae":
        model = BetaE(num_entities=nentity, num_relations=nrelation, embedding_size=args.entity_space_dim, label_smoothing=args.label_smoothing)
    
    elif args.model == "cone":
        model = ConE(num_entities=nentity, num_relations=nrelation, embedding_size=args.entity_space_dim, label_smoothing=args.label_smoothing)
    
    elif args.model == "q2b":
        model = Q2B(num_entities=nentity, num_relations=nrelation, embedding_size=args.entity_space_dim, label_smoothing=args.label_smoothing)

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
        model = TokenGTModel(num_entities=nentity, num_relations=nrelation, embedding_size=args.entity_space_dim, num_layers = args.num_layers, label_smoothing=args.label_smoothing)
    



    # add scheduler for the transformer model to do warmup, or the model will not converge at all
    optimizer = None
    if args.model == "hype":
         optimizer = RiemannianAdam(
                        filter(lambda p: p.requires_grad, model.parameters()), 
                        lr=args.learning_rate
                    )
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.learning_rate
        )

    if args.model == "transformer" or args.model == "biqe" or session_encoder == "TransRec" or args.model == "nqe" or args.model == "gt" or args.model == "tokengt" :

        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': 0.01},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

        def warmup_lambda(epoch):
            if epoch < args.warm_up_steps:
                return epoch * 1.0 / args.warm_up_steps
            else:
                return 1


        scheduler = LambdaLR(optimizer, lr_lambda=warmup_lambda)
    
    if args.model =="betae" or args.model == "cone" or args.model == "q2b":
        def warmup_lambda(epoch):
            if epoch < args.warm_up_steps:
                return epoch * 1.0 / args.warm_up_steps
            else:
                return 1
        scheduler = LambdaLR(optimizer, lr_lambda=warmup_lambda)
    
    if torch.cuda.is_available():
        model = model.cuda()

    global_steps = -1

    file_count = -1
    model_name = args.model
    
    # for train_query_file_name in train_query_file_names:
    print(train_query_file_names)
    file_count += 1
    train_query_file_name = np.random.choice(train_query_file_names)

    print("====== Training ======", model_name, train_query_file_name)

    with open(args.train_query_dir + "/" + train_query_file_name , "r") as fin:
        train_data_dict = json.load(fin)


    train_iterators = {}
    for query_type, query_answer_dict in train_data_dict.items():
            
            if args.model == "gt":
                 new_iterator = SingledirectionalOneShotIterator(DataLoader(
                    TrainDatasetPE(nentity, nrelation, query_answer_dict),
                    batch_size=batch_size,
                    shuffle=True,
                    collate_fn=TrainDatasetPE.collate_fn
                ))
            
            elif args.model == "shgt" or args.model == "hgt" or args.model == "rgcn":
                new_iterator = SingledirectionalOneShotIterator(DataLoader(
                    TrainDatasetDGL(nentity, nrelation, query_answer_dict),
                    batch_size=batch_size,
                    shuffle=True,
                    collate_fn=TrainDatasetDGL.collate_fn
                ))
            
            elif args.model == "tokengt":
                new_iterator = SingledirectionalOneShotIterator(DataLoader(
                    TrainDatasetTokenGT(nentity, nrelation, query_answer_dict),
                    batch_size=batch_size,
                    shuffle=True,
                    collate_fn=TrainDatasetTokenGT.collate_fn
                ))
                
            else: 
                new_iterator = SingledirectionalOneShotIterator(DataLoader(
                    TrainDataset(nentity, nrelation, query_answer_dict, max_length),
                    batch_size=batch_size,
                    shuffle=True,
                    collate_fn=TrainDataset.collate_fn
                ))
            train_iterators[query_type] = new_iterator

    train_iteration_names = list(train_iterators.keys())
    
    total_length = 0
    for key, value in train_iterators.items():
        print(key, value.len)
        total_length += value.len
        

    total_step = total_length

    # Train the model
    while True:
        for step in tqdm(range(total_step)):
            global_steps += 1

            model.train()
            
            
            task_name = np.random.choice(train_iteration_names)
            iterator = train_iterators[task_name]

            with torch.cuda.amp.autocast():

                if args.model == "gt":
                    unified_graph_id_list, graph_RWPE_list, graph_LapPE_list, positive_sample, attention_mask = next(iterator)

                    if args.positional_encoding_type == "rwpe":
                        loss = model(unified_graph_id_list, attention_mask, graph_RWPE_list, positive_sample)
                    elif args.positional_encoding_type == "lappe":
                        loss = model(unified_graph_id_list, attention_mask, graph_LapPE_list, positive_sample)     

                elif args.model == "shgt" or args.model == "hgt" or args.model == "rgcn":
                    dgl_graph, graph_size, token_list, node_type_list, edge_type_list, positive_sample = next(iterator)
                    loss = model(dgl_graph, token_list, node_type_list, edge_type_list, graph_size, positive_sample)
                
                elif args.model == "tokengt":
                    unified_feature_ids, node_identifiers, type_identifiers, attention_mask, positive_sample = next(iterator)
                    loss = model(unified_feature_ids, node_identifiers, type_identifiers, attention_mask, positive_sample)
                    
                
                else: 
                    batched_query, unified_ids, positive_sample = next(iterator)
                    
                    if args.model == "lstm" or args.model == "transformer" or args.model == "tcn" or args.model == "rnn" or args.model == "gru" or args.model == "biqe" :
                        batched_query = unified_ids
                    
                    loss = model(batched_query, positive_sample)

                if args.gradient_accumulation_steps > 1:
                    # loss = loss / args.gradient_accumulation_steps
                    # loss.backward()
                    scaler.scale(loss).backward()

                    if (global_steps + 1) % args.gradient_accumulation_steps == 0:
                        # optimizer.step()
                        scaler.step(optimizer)
                        scaler.update()

                        optimizer.zero_grad()

                else: 
                    # loss.backward()
                    scaler.scale(loss).backward()
                    # optimizer.step()
                    scaler.step(optimizer)
                    optimizer.zero_grad()
            
            if args.model == "transformer" or args.model == "betae" or args.model == "cone" or \
                  args.model == "biqe" or session_encoder == "TransRec" or args.model == "nqe" or args.model == "gt" or \
                     args.model == "tokengt":
                scheduler.step()
            
            if global_steps % 100 == 0:
                train_summary_writer.add_scalar("y-train-" + task_name, loss.item(), global_steps)
            
            save_step = args.log_steps
            model_name = args.model

            
            # Evaluate the model
            if global_steps % args.log_steps == 0:

                # Save the model
                model.eval()
                general_checkpoint_path = args.checkpoint_path + "/" + model_name + "_" + str(global_steps) + "_" + info + "_" + data_name + ".bin"
                if args.model == "gt":
                    general_checkpoint_path = args.checkpoint_path + "/" + model_name + "_" + str(global_steps) + "_" + info + "_" + data_name + "_" + args.positional_encoding_type + ".bin"

                
                torch.save({
                    'steps': global_steps,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, general_checkpoint_path)
            

                # Validation
                print("====== Validation ======", model_name)
                entailment_58_types_logs = []
                entailment_29_types_logs = []
                entailment_unseen_29_types_logs = []

                generalization_58_types_logs = []
                generalization_29_types_logs = []
                generalization_unseen_29_types_logs = []

                entailment_58_types_dict = {}
                generalization_58_types_dict = {}
        
                for valid_query_file_name in tqdm(valid_query_file_names):

                    with open(args.valid_query_dir + "/" + valid_query_file_name , "r") as fin:
                        valid_data_dict = json.load(fin)
                    

                    validation_loaders = {}
                    for query_type, query_answer_dict in valid_data_dict.items():

                        if args.model == "gqe" or args.model == "q2b" or args.model=="hype":
                            if "u" in query_type or "n" in query_type:
                                continue

                        if args.use_full_fol == False:
                            if "u" in query_type or "n" in query_type:
                                continue
                        if args.model == "gt":
                            new_iterator = DataLoader(
                                ValidDatasetPE(nentity, nrelation, query_answer_dict),
                                batch_size=args.evaluation_batch_size,
                                shuffle=True,
                                collate_fn=ValidDatasetPE.collate_fn
                            )
                        
                        elif args.model == "shgt" or args.model == "hgt" or args.model == "rgcn":
                            new_iterator = DataLoader(
                                ValidDatasetDGL(nentity, nrelation, query_answer_dict),
                                batch_size=args.evaluation_batch_size,
                                shuffle=True,
                                collate_fn=ValidDatasetDGL.collate_fn
                            )
                        
                        elif args.model == "tokengt":
                            new_iterator = DataLoader(
                                ValidDatasetTokenGT(nentity, nrelation, query_answer_dict),
                                batch_size=args.evaluation_batch_size,
                                shuffle=True,
                                collate_fn=ValidDatasetTokenGT.collate_fn
                            )
                        else: 
                            new_iterator = DataLoader(
                                ValidDataset(nentity, nrelation, query_answer_dict, max_length),
                                batch_size=args.evaluation_batch_size,
                                shuffle=True,
                                collate_fn=ValidDataset.collate_fn
                            )
                        validation_loaders[query_type] = new_iterator

                    

                    for task_name, loader in validation_loaders.items():

                        all_entailment_logs = []
                        all_generalization_logs = []
                        

                        if args.model == "gt":
                            for unified_graph_id_list, graph_RWPE_list, graph_LapPE_list, train_answers, valid_answers, attention_mask  in loader:

                                if args.positional_encoding_type == "rwpe":
                                    query_embedding = model(unified_graph_id_list, attention_mask, graph_RWPE_list)
                                elif args.positional_encoding_type == "lappe":
                                    query_embedding = model(unified_graph_id_list, attention_mask, graph_LapPE_list)

                                entailment_logs = model.evaluate_entailment(query_embedding, train_answers)
                                generalization_logs = model.evaluate_generalization(query_embedding, train_answers, valid_answers)

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
                            for dgl_graph, graph_size, token_list, node_type_list, edge_type_list, train_answers, valid_answers in loader:
                                query_embedding = model(dgl_graph, token_list, node_type_list, edge_type_list, graph_size)

                                entailment_logs = model.evaluate_entailment(query_embedding, train_answers)
                                generalization_logs = model.evaluate_generalization(query_embedding, train_answers, valid_answers)

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
                            for unified_feature_ids, node_identifiers, type_identifiers, attention_mask, train_answers, valid_answers in loader:
                                query_embedding = model(unified_feature_ids, node_identifiers, type_identifiers, attention_mask)

                                entailment_logs = model.evaluate_entailment(query_embedding, train_answers)
                                generalization_logs = model.evaluate_generalization(query_embedding, train_answers, valid_answers)

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
                            for batched_query, unified_ids, train_answers, valid_answers in loader:

                                if args.model == "lstm" or args.model == "transformer" or args.model == "tcn" or args.model == "rnn" or args.model == "gru" or args.model == "biqe":
                                    batched_query = unified_ids

                                query_embedding = model(batched_query)
                                entailment_logs = model.evaluate_entailment(query_embedding, train_answers)
                                generalization_logs = model.evaluate_generalization(query_embedding, train_answers, valid_answers)

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

                            
                for task_name, logs in entailment_58_types_dict.items():
                    aggregated_entailment_logs = log_aggregation(logs)
                    for key, value in aggregated_entailment_logs.items():
                        test_summary_writer.add_scalar("z-valid-" + task_name + "-" + key, value, global_steps)
                
                for task_name, logs in generalization_58_types_dict.items():
                    aggregated_generalization_logs = log_aggregation(logs)
                    for key, value in aggregated_generalization_logs.items():
                        test_summary_writer.add_scalar("z-valid-" + task_name + "-" + key, value, global_steps)
                
                
                entailment_58_types_logs = log_aggregation(entailment_58_types_logs)
                generalization_58_types_logs = log_aggregation(generalization_58_types_logs)
                entailment_29_types_logs = log_aggregation(entailment_29_types_logs)
                generalization_29_types_logs = log_aggregation(generalization_29_types_logs)
                entailment_unseen_29_types_logs = log_aggregation(entailment_unseen_29_types_logs)
                generalization_unseen_29_types_logs = log_aggregation(generalization_unseen_29_types_logs)

                for key, value in entailment_58_types_logs.items():
                    test_summary_writer.add_scalar("x-valid-58-types-" + key, value, global_steps)

                for key, value in generalization_58_types_logs.items():
                    test_summary_writer.add_scalar("x-valid-58-types-" + key, value, global_steps)

                for key, value in entailment_29_types_logs.items():
                    test_summary_writer.add_scalar("x-valid-29-types-" + key, value, global_steps)

                for key, value in generalization_29_types_logs.items():
                    test_summary_writer.add_scalar("x-valid-29-types-" + key, value, global_steps)

                for key, value in entailment_unseen_29_types_logs.items():
                    test_summary_writer.add_scalar("x-valid-unseen-29-types-" + key, value, global_steps)

                for key, value in generalization_unseen_29_types_logs.items():
                    test_summary_writer.add_scalar("x-valid-unseen-29-types-" + key, value, global_steps)

                
                print("====== Test ======", model_name)
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
                    
                
                for task_name, logs in entailment_58_types_dict.items():
                    aggregated_entailment_logs = log_aggregation(logs)
                    for key, value in aggregated_entailment_logs.items():
                        test_summary_writer.add_scalar("z-test-" + task_name + "-" + key, value, global_steps)
                
                for task_name, logs in generalization_58_types_dict.items():
                    aggregated_generalization_logs = log_aggregation(logs)
                    for key, value in aggregated_generalization_logs.items():
                        test_summary_writer.add_scalar("z-test-" + task_name + "-" + key, value, global_steps)
                

                
                entailment_58_types_logs = log_aggregation(entailment_58_types_logs)
                generalization_58_types_logs = log_aggregation(generalization_58_types_logs)
                entailment_29_types_logs = log_aggregation(entailment_29_types_logs)
                generalization_29_types_logs = log_aggregation(generalization_29_types_logs)
                entailment_unseen_29_types_logs = log_aggregation(entailment_unseen_29_types_logs)
                generalization_unseen_29_types_logs = log_aggregation(generalization_unseen_29_types_logs)

                for key, value in entailment_58_types_logs.items():
                    test_summary_writer.add_scalar("x-test-58-types-" + key, value, global_steps)

                for key, value in generalization_58_types_logs.items():
                    test_summary_writer.add_scalar("x-test-58-types-" + key, value, global_steps)

                for key, value in entailment_29_types_logs.items():
                    test_summary_writer.add_scalar("x-test-29-types-" + key, value, global_steps)

                for key, value in generalization_29_types_logs.items():
                    test_summary_writer.add_scalar("x-test-29-types-" + key, value, global_steps)

                for key, value in entailment_unseen_29_types_logs.items():
                    test_summary_writer.add_scalar("x-test-unseen-29-types-" + key, value, global_steps)

                for key, value in generalization_unseen_29_types_logs.items():
                    test_summary_writer.add_scalar("x-test-unseen-29-types-" + key, value, global_steps)

        
                gc.collect()
        

















