export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="./dataset/maps"
export NAVSIM_EXP_ROOT="..."
export NAVSIM_DEVKIT_ROOT="..."
export OPENSCENE_DATA_ROOT="..."
# 20000
split=navhard_two_stage
agent=flowdrive_agent
dir=test_flow_${split}
metric_cache_path="${NAVSIM_EXP_ROOT}/${split}_metric_cache"
ckpt="/nas/users/perception-users/liulin/playground/hfd/models--Zzxxxxxxxx--gtrs/epoch99_epc.ckp" 
#"/nas/users/perception-users/liulin/workspace/DiffusionDrive/exp/training_flow_agent_withoutlidar/2025.08.24.10.55.29/lightning_logs/version_0/checkpoints/epoch19.ckpt"
experiment_name=test_flowdrive
# epoch99_ep.ckpt
export DP_PREDS=none
export SUBSCORE_PATH=${NAVSIM_EXP_ROOT}/${dir}/${split}.pkl # save path for the dp-generated trajectories

PYTHONPATH=$PYTHONPATH:./ python ${NAVSIM_DEVKIT_ROOT}/navsim_test/planning/script/run_pdm_score_gpu_v2.py \
agent=$agent \
dataloader.params.batch_size=12 \
agent.checkpoint_path=${ckpt} \
trainer.params.precision=32 \
experiment_name=${experiment_name} \
+cache_path=null \
metric_cache_path=${metric_cache_path} \
train_test_split=${split}
