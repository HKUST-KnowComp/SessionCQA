CUDA_VISIBLE_DEVICES=6 python ./deduction_model/train.py \
-dn hyper_graph_data_en \
-m q2p \
--train_query_dir ./sampled_hyper_train_merged \
--valid_query_dir ./sampled_hyper_valid \
--test_query_dir ./sampled_hyper_test \
--checkpoint_path ./logs \
-fol  \
-b 96 \
--log_steps 120000 \
-lr 0.001 \
--session_encoder SRGNN \
-ga 4
