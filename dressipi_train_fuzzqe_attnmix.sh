CUDA_VISIBLE_DEVICES=2 python ./deduction_model/train.py \
-dn hyper_graph_dressipi \
-m fuzzqe \
--train_query_dir ./sampled_hyper_train_dressipi_merged \
--valid_query_dir ./sampled_hyper_valid_dressipi \
--test_query_dir ./sampled_hyper_test_dressipi \
--checkpoint_path ./logs \
-fol  \
-b 512 \
--log_steps 120000 \
-lr 0.001 \
--session_encoder AttnMixer