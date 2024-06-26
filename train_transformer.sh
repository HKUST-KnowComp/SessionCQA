CUDA_VISIBLE_DEVICES=2 python ./deduction_model/train.py \
-dn hyper_graph_data_en \
-m transformer \
--train_query_dir ./sampled_hyper_train_merged \
--valid_query_dir ./sampled_hyper_valid \
--test_query_dir ./sampled_hyper_test \
--checkpoint_path ./logs \
-fol  \
-b 384 \
--log_steps 120000 \
-lr 0.0005
