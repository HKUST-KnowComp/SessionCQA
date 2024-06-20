import json
from pickle import FALSE

import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F

from dataloader import TestDataset, ValidDataset, TrainDataset, SingledirectionalOneShotIterator
from session_model import SRGNNRec, AttentionMixerRec, GRURec, TransRec


number_negative_samples = 10000

class GeneralModel(nn.Module):

    def __init__(self, num_entities, num_relations, embedding_size):
        super(GeneralModel, self).__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.embedding_size = embedding_size

    def loss_fnt(self, sub_query_encoding, labels):
        raise NotImplementedError

    def scoring(self, query_encoding):
        """
        :param query_encoding:
        :return: [batch_size, num_entities]
        """
        raise NotImplementedError

    
    def selected_scoring(self, query_encoding, selected_entities):
        """

        :param query_encoding: [num_particles, embedding_size]
        :param selected_entities: [num_entities]
        :return: [ num_particles]
        """
        if isinstance(query_encoding, tuple):
            # For BoxE and ConE
            center, offset = query_encoding
            concated = torch.cat([center, offset], dim=-1)
            query_encoding = self.scoring_projection(concated)

            
        # [batch_size, num_entities, embedding_size]
        entity_embeddings = self.entity_embedding(selected_entities)

        # [batch_size, num_particles, num_entities]
        # print("query encoding shape: ", query_encoding.shape)
        # print("entity embedding shape: ", entity_embeddings.shape)
        

        query_scores = torch.matmul(query_encoding, entity_embeddings.transpose(-2, -1))
        # print("query scores shape: ", query_scores.shape)
       
        return query_scores



    def evaluate_entailment(self, query_encoding, entailed_answers):

    
        log_list = []

        # This is a permutation from the coverted indices to the original indices, and we select the first k elements as negative samples
        try: 
            device = query_encoding.device
        
        except:
            # For the case of BoxE and ConE
            device = query_encoding[0].device
        
        if isinstance(query_encoding, tuple):
            # For BoxE and ConE
            center, offset = query_encoding
            concated = torch.cat([center, offset], dim=-1)
            query_encoding = self.scoring_projection(concated)

       
        random_negative_indices = torch.randperm(self.num_entities, device=device)

        # This is a permutation from the original indices to the converted indices
        reversed_permutation = torch.argsort(random_negative_indices).to(device)

        selected_negative_entities = random_negative_indices[:number_negative_samples].to(device)

        selected_negative_scores = self.selected_scoring(query_encoding, selected_negative_entities)


        with torch.no_grad():


            for i in range(len(entailed_answers)):
                
                if len(entailed_answers[i]) == 0:
                    continue
                
                entailed_answer_set = torch.tensor(entailed_answers[i], device=device, dtype=torch.long)
                postive_scores = self.selected_scoring(query_encoding[i], entailed_answer_set)

    
                postive_positions = reversed_permutation[entailed_answer_set]
                postive_positions = postive_positions[postive_positions < number_negative_samples]

                mask = torch.ones(number_negative_samples, dtype=torch.bool, device=device)
                mask[postive_positions] = False

                selected_negative_scores_filtered = selected_negative_scores[i][mask]

                # [num_entailed_answers, 1]
                entailed_answers_scores = postive_scores.unsqueeze(1)
                # [1, num_negative_samples]
                negative_scores = selected_negative_scores_filtered.unsqueeze(0)

                # [num_entailed_answers, num_negative_samples]
                scores = entailed_answers_scores - negative_scores
                # print("scores: ", scores)
                
                rankings = torch.sum(scores < 0, dim=1) + 1

                rankings = rankings.float()

                mrr = torch.mean(torch.reciprocal(rankings)).cpu().numpy().item()
                hit_at_1 = torch.mean((rankings < 1.5).double()).cpu().numpy().item()
                hit_at_3 = torch.mean((rankings < 3.5).double()).cpu().numpy().item()
                hit_at_10 = torch.mean((rankings < 10.5).double()).cpu().numpy().item()

                num_answers = len(entailed_answers[i])

                logs = {
                    "ent_mrr": mrr,
                    "ent_hit_at_1": hit_at_1,
                    "ent_hit_at_3": hit_at_3,
                    "ent_hit_at_10": hit_at_10,
                    "ent_num_answers": num_answers
                }

                log_list.append(logs)
        
        return log_list


                             


    # def evaluate_entailment(self, query_encoding, entailed_answers):
    #     """
    #     :param query_encoding:
    #     :param entailed_answers:
    #     :return:
    #     """

    #     # [batch_size, num_entities]
    #     with torch.no_grad():
            
    #         all_scoring = self.scoring(query_encoding)

    #         # [batch_size, num_entities]
    #         original_scores = all_scoring.clone()

    #         log_list = []

    #         for i in range(len(entailed_answers)):

    #             if len(entailed_answers[i]) == 0:
    #                 continue
    #             entailed_answer_set = torch.tensor(entailed_answers[i])
                

    #             # [num_entities]
    #             not_answer_scores = all_scoring[i]
    #             not_answer_scores[entailed_answer_set] = - 10000000

    #             # [1, num_entities]
    #             not_answer_scores = not_answer_scores.unsqueeze(0)

    #             # [num_entailed_answers, 1]
    #             entailed_answers_scores = original_scores[i][entailed_answer_set].unsqueeze(1)

    #             # [num_need_to_inferred_answers, num_entities]
    #             answer_rankings_list = []

    #             batch_size = 128
    #             for j in range(0, entailed_answers_scores.shape[0], batch_size):
    #                 batch = entailed_answers_scores[j:j+batch_size]
    #                 answer_is_smaller_matrix = ((batch - not_answer_scores) < 0)
    #                 answer_rankings = answer_is_smaller_matrix.sum(dim=-1) + 1

    #                 answer_rankings_list.append(answer_rankings)


    #             # [num_need_to_inferred_answers]
    #             answer_rankings = torch.cat(answer_rankings_list, dim = 0)


    #             # [num_entailed_answers]
    #             rankings = answer_rankings.float()

    #             mrr = torch.mean(torch.reciprocal(rankings)).cpu().numpy().item()
    #             hit_at_1 = torch.mean((rankings < 1.5).double()).cpu().numpy().item()
    #             hit_at_3 = torch.mean((rankings < 3.5).double()).cpu().numpy().item()
    #             hit_at_10 = torch.mean((rankings < 10.5).double()).cpu().numpy().item()

    #             num_answers = len(entailed_answers[i])

    #             logs = {
    #                 "ent_mrr": mrr,
    #                 "ent_hit_at_1": hit_at_1,
    #                 "ent_hit_at_3": hit_at_3,
    #                 "ent_hit_at_10": hit_at_10,
    #                 "ent_num_answers": num_answers
    #             }

    #             log_list.append(logs)
    #     return log_list


    def evaluate_generalization(self, query_encoding, entailed_answers, generalized_answers):
       

        
        log_list = []

        try:
            device = query_encoding.device
        except:
            # For the case of BoxE and ConE
            device = query_encoding[0].device

        random_negative_indices = torch.randperm(self.num_entities, device=device)
        # This is a permutation from the original indices to the converted indices
        reversed_permutation = torch.argsort(random_negative_indices).to(device)

        selected_negative_entities = random_negative_indices[:number_negative_samples].to(device)
        selected_negative_scores = self.selected_scoring(query_encoding, selected_negative_entities)


        if isinstance(query_encoding, tuple):
            # For BoxE and ConE
            center, offset = query_encoding
            concated = torch.cat([center, offset], dim=-1)
            query_encoding = self.scoring_projection(concated)

        with torch.no_grad():


            for i in range(len(entailed_answers)):

                all_answers = list(set(entailed_answers[i]) | set(generalized_answers[i]))
                need_to_inferred_answers = list(set(generalized_answers[i]) - set(entailed_answers[i]))
                num_answers = len(need_to_inferred_answers)
                if num_answers == 0:
                    continue

                all_answers = torch.tensor(all_answers, device=device, dtype=torch.long)
                need_to_inferred_answers = torch.tensor(need_to_inferred_answers, device=device, dtype=torch.long)

                postive_scores = self.selected_scoring(query_encoding[i], need_to_inferred_answers)

                postive_positions = reversed_permutation[all_answers]
                postive_positions = postive_positions[postive_positions < number_negative_samples]

                mask = torch.ones(number_negative_samples, dtype=torch.bool, device=device)
                mask[postive_positions] = False

                selected_negative_scores_filtered = selected_negative_scores[i][mask]

               
                # [num_entailed_answers, 1]
                entailed_answers_scores = postive_scores.unsqueeze(1)
                # [1, num_negative_samples]
                negative_scores = selected_negative_scores_filtered.unsqueeze(0)

                # [num_entailed_answers, num_negative_samples]
                scores = entailed_answers_scores - negative_scores
                rankings = torch.sum(scores < 0, dim=1) + 1

                rankings = rankings.float()

                mrr = torch.mean(torch.reciprocal(rankings)).cpu().numpy().item()
                hit_at_1 = torch.mean((rankings < 1.5).double()).cpu().numpy().item()
                hit_at_3 = torch.mean((rankings < 3.5).double()).cpu().numpy().item()
                hit_at_10 = torch.mean((rankings < 10.5).double()).cpu().numpy().item()

                
                logs = {
                    "inf_mrr": mrr,
                    "inf_hit_at_1": hit_at_1,
                    "inf_hit_at_3": hit_at_3,
                    "inf_hit_at_10": hit_at_10,
                    "inf_num_answers": num_answers
                }

                log_list.append(logs)
        
        return log_list



    # def evaluate_generalization(self, query_encoding, entailed_answers, generalized_answers):
    #     """


    #     :param query_encoding:
    #     :param entailed_answers:
    #     :param generalized_answers:
    #     :return:
    #     """

    #     # [batch_size, num_entities]
    #     with torch.no_grad():
    #         all_scoring = self.scoring(query_encoding)

    #         # [batch_size, num_entities]
    #         original_scores = all_scoring.clone()

    #         log_list = []

    #         for i in range(len(entailed_answers)):

    #             all_answers = list(set(entailed_answers[i]) | set(generalized_answers[i]))
    #             need_to_exclude_answers = list(set(entailed_answers[i]) - set(generalized_answers[i]))
    #             need_to_inferred_answers = list(set(generalized_answers[i]) - set(entailed_answers[i]))

    #             if len(need_to_inferred_answers) == 0:
    #                 continue

    #             all_answers_set = torch.tensor(all_answers)

    #             # [num_entities]
    #             not_answer_scores = all_scoring[i]
    #             not_answer_scores[all_answers_set] = - 10000000

    #             # [1, num_entities]
    #             not_answer_scores = not_answer_scores.unsqueeze(0)

    #             logs = {}

    #             if len(need_to_inferred_answers) > 0:
    #                 num_answers = len(need_to_inferred_answers)

    #                 need_to_inferred_answers = torch.tensor(need_to_inferred_answers)

    #                 # [num_need_to_inferred_answers, 1]
    #                 need_to_inferred_answers_scores = original_scores[i][need_to_inferred_answers].unsqueeze(1)

    #                 # [num_need_to_inferred_answers, num_entities]
    #                 answer_rankings_list = []

    #                 batch_size = 128
    #                 for j in range(0, need_to_inferred_answers_scores.shape[0], batch_size):
    #                     batch = need_to_inferred_answers_scores[j:j+batch_size]
    #                     answer_is_smaller_matrix = ((batch - not_answer_scores) < 0)
    #                     answer_rankings = answer_is_smaller_matrix.sum(dim=-1) + 1

    #                     answer_rankings_list.append(answer_rankings)


    #                 # [num_need_to_inferred_answers]
    #                 answer_rankings = torch.cat(answer_rankings_list, dim = 0)

               

    #                 # [num_need_to_inferred_answers]
    #                 rankings = answer_rankings.float()

    #                 mrr = torch.mean(torch.reciprocal(rankings)).cpu().numpy().item()
    #                 hit_at_1 = torch.mean((rankings < 1.5).double()).cpu().numpy().item()
    #                 hit_at_3 = torch.mean((rankings < 3.5).double()).cpu().numpy().item()
    #                 hit_at_10 = torch.mean((rankings < 10.5).double()).cpu().numpy().item()

    #                 logs["inf_mrr"] = mrr
    #                 logs["inf_hit_at_1"] = hit_at_1
    #                 logs["inf_hit_at_3"] = hit_at_3
    #                 logs["inf_hit_at_10"] = hit_at_10
    #                 logs["inf_num_answers"] = num_answers
    #             else:
    #                 logs["inf_mrr"] = 0
    #                 logs["inf_hit_at_1"] = 0
    #                 logs["inf_hit_at_3"] = 0
    #                 logs["inf_hit_at_10"] = 0
    #                 logs["inf_num_answers"] = 0

    #             if len(need_to_exclude_answers) > 0:
    #                 num_answers = len(need_to_exclude_answers)

    #                 need_to_exclude_answers = torch.tensor(need_to_exclude_answers)

    #                 # [num_need_to_exclude_answers, 1]
    #                 need_to_exclude_answers_scores = original_scores[i][need_to_exclude_answers].unsqueeze(1)

    #                 # [num_need_to_inferred_answers, num_entities]
    #                 answer_rankings_list = []

    #                 batch_size = 128
    #                 for j in range(0, need_to_exclude_answers_scores.shape[0], batch_size):
    #                     batch = need_to_exclude_answers_scores[j:j+batch_size]
    #                     answer_is_smaller_matrix = ((batch - not_answer_scores) < 0)
    #                     answer_rankings = answer_is_smaller_matrix.sum(dim=-1) + 1

    #                     answer_rankings_list.append(answer_rankings)


    #                 # [num_need_to_inferred_answers]
    #                 answer_rankings = torch.cat(answer_rankings_list, dim = 0)
    #                 # [num_need_to_exclude_answers]
    #                 rankings = answer_rankings.float()

    #                 mrr = torch.mean(torch.reciprocal(rankings)).cpu().numpy().item()
    #                 hit_at_1 = torch.mean((rankings < 1.5).double()).cpu().numpy().item()
    #                 hit_at_3 = torch.mean((rankings < 3.5).double()).cpu().numpy().item()
    #                 hit_at_10 = torch.mean((rankings < 10.5).double()).cpu().numpy().item()

    #                 logs["exd_mrr"] = mrr
    #                 logs["exd_hit_at_1"] = hit_at_1
    #                 logs["exd_hit_at_3"] = hit_at_3
    #                 logs["exd_hit_at_10"] = hit_at_10
    #                 logs["exd_num_answers"] = num_answers
    #             else:
    #                 logs["exd_mrr"] = 0
    #                 logs["exd_hit_at_1"] = 0
    #                 logs["exd_hit_at_3"] = 0
    #                 logs["exd_hit_at_10"] = 0
    #                 logs["exd_num_answers"] = 0

    #             log_list.append(logs)
    #     return log_list


class IterativeModel(GeneralModel):

    def __init__(self, num_entities, num_relations, embedding_size, use_old_loss = False, session_encoder = "SRGNN"):
        super(IterativeModel, self).__init__(num_entities, num_relations, embedding_size)
        self.use_old_loss = use_old_loss
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.embedding_size = embedding_size


    def projection(self, relation_ids, sub_query_encoding):
        raise NotImplementedError

    def higher_projection(self, relation_ids, sub_query_encoding):
        raise NotImplementedError

    def intersection(self, sub_query_encoding_list):
        raise NotImplementedError

    def union(self, sub_query_encoding_list):
        raise NotImplementedError

    def negation(self, sub_query_encoding):
        raise NotImplementedError

    def forward(self, batched_structured_query, label=None):

        assert batched_structured_query[0] in ["p", "e", "i", "u", "n", "s"]

        if batched_structured_query[0] == "p":

            sub_query_result = self.forward(batched_structured_query[2])
            if batched_structured_query[2][0] == 'e' or batched_structured_query[2][0] == 's':
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
            this_query_result = self.entity_embedding(entity_ids)
        
        elif batched_structured_query[0] == "s":

            entity_ids = torch.tensor(batched_structured_query[1])
            entity_ids = entity_ids.to(self.entity_embedding.weight.device)

            sequence_length = (entity_ids != self.num_entities).sum(dim=1).to(self.entity_embedding.weight.device)
            this_query_result = self.session_encoder(entity_ids, sequence_length)

        else:
            print("Error: unknown query type")
            print(batched_structured_query)
            this_query_result = None

        if label is None:
            return this_query_result

        else:
            if self.use_old_loss == False:
                return self.loss_fnt(this_query_result, label)
            else:
                return self.old_loss_fnt(this_query_result, label)


class SequentialModel(GeneralModel):

    def __init__(self, num_entities, num_relations, embedding_size):
        super().__init__(num_entities, num_relations, embedding_size)

    def encode(self, batched_structured_query):
        raise NotImplementedError

    def forward(self, batched_structured_query, label=None):
        
        batched_structured_query = torch.tensor(batched_structured_query)
        if torch.cuda.is_available():
            batched_structured_query = batched_structured_query.cuda()

        representations = self.encode(batched_structured_query)

        if label is not None:
            return self.loss_fnt(representations, label)

        else:
            return representations
    


class LabelSmoothingLoss(torch.nn.Module):
    def __init__(self, smoothing: float = 0.1,
                 reduction="mean", weight=None):
        super(LabelSmoothingLoss, self).__init__()
        self.smoothing = smoothing
        self.reduction = reduction
        self.weight = weight

    def reduce_loss(self, loss):
        return loss.mean() if self.reduction == 'mean' else loss.sum() \
            if self.reduction == 'sum' else loss

    def linear_combination(self, x, y):
        return self.smoothing * x + (1 - self.smoothing) * y

    def forward(self, preds, target):
        assert 0 <= self.smoothing < 1

        if self.weight is not None:
            self.weight = self.weight.to(preds.device)

        n = preds.size(-1)
        log_preds = F.log_softmax(preds, dim=-1)
        
        loss = self.reduce_loss(-log_preds.sum(dim=-1))

        nll = F.nll_loss(
            log_preds, target, reduction=self.reduction, weight=self.weight
        )
        return self.linear_combination(loss / n, nll)


if __name__ == "__main__":

    pass