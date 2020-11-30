CUDA_VISIBLE_DEVICES=0 python3 src/eval.py \
	--algorithm pad \
	--num_shared_layers 8 \
	--num_head_layers 3 \
	--eval_episodes 100 \
	--seed 0