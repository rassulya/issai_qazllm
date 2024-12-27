export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONPATH=.
export NCCL_IB_GID_INDEX=1
export NCCL_SOCKET_IFNAME=eth0 
export NCCL_IB_HCA=mlx5_0,mlx5_10,mlx5_11,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_9
export NCCL_IB_TIMEOUT=22
export NCCL_IB_DISABLE=0
export NCCL_IB_RETRY_CNT=7
export NCCL_NET=IB
export PYTHONPATH=.
# export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=6000
# export TORCH_NCCL_ENABLE_MONITORING=0
# export NCCL_DEBUG=INFO 

# torchrun --nproc_per_node=8 --nnodes=8 --master_addr=<ip_address> --master_port=1234 --node_rank=1  recipes/full_finetune_distributed_loop.py --config config_train/8B_3.1_base_val_loop.yaml
# python tune.py run --nproc_per_node=8 --nnodes=6 --master_addr=<ip_address> --master_port=1234 --node_rank=0  full_finetune_distributed_loop_no_val --config config_train/8B_3.1_inst_noval_loop.yaml
python3 tune.py run --nproc_per_node=8 --nnodes=1 --node_rank=0  full_finetune_distributed_loop_no_val --config config_train/8B_3.1_inst_noval_loop.yaml