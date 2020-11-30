CUDA_VISIBLE_DEVICES=0 python3 src/train.py \
	--algorithm pad \
	--num_shared_layers 8 \
	--num_head_layers 3 \
	--seed 0