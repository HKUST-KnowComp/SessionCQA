CUDA_VISIBLE_DEVICES=7 python ./deduction_model/train.py \
-dn hyper_graph_dressipi \
-m q2p \
--train_query_dir ./sampled_hyper_train_dressipi_merged \
--valid_query_dir ./sampled_hyper_valid_dressipi \
--test_query_dir ./sampled_hyper_test_dressipi \
--checkpoint_path ./logs \
-fol  \
-b 384 \
--log_steps 120000 \
-lr 0.001 \
--session_encoder SRGNN 

