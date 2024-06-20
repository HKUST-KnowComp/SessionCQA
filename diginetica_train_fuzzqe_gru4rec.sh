CUDA_VISIBLE_DEVICES=3 python ./deduction_model/train.py \
-dn hyper_graph_diginetica \
-m fuzzqe \
--train_query_dir ./sampled_hyper_train_diginetica_merged \
--valid_query_dir ./sampled_hyper_valid_diginetica \
--test_query_dir ./sampled_hyper_test_diginetica \
--checkpoint_path ./logs \
-fol  \
-b 512 \
--log_steps 120000 \
-lr 0.001 \
--session_encoder GRURec
