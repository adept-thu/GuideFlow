export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT=""
export NAVSIM_EXP_ROOT=""
export NAVSIM_DEVKIT_ROOT=""
export OPENSCENE_DATA_ROOT=""

#s_split=navhard_two_stage
split=navhard_two_stage
agent=gtrs_dense_vov
dir=test_gtrs_dense_${split}
metric_cache_path="${NAVSIM_EXP_ROOT}/${split}_metric_cache"
experiment_name=test_dense

ckpt= # this can also be the checkpoint we provided
SYNTHETIC_SENSOR_PATH=$OPENSCENE_DATA_ROOT/navhard_two_stage/sensor_blobs
SYNTHETIC_SCENES_PATH=$OPENSCENE_DATA_ROOT/navhard_two_stage/openscene_meta_datas
export DP_PREDS='navhard_two_stage.pkl' #./exp/test_flow_navhard_two_stage_1/navhard_two_stage.pkl' #'./exp/test_flow_navhard_two_stage/navhard_two_stage.pkl' 
export SUBSCORE_PATH=${NAVSIM_EXP_ROOT}/${dir}/${split}.pkl; # save path for the scores

PYTHONPATH=$PYTHONPATH:./ python ${NAVSIM_DEVKIT_ROOT}/navsim/planning/script/run_pdm_score_gpu_v2.py \
+combined_inference=true \
agent=$agent \
agent.config.vocab_path=${NAVSIM_DEVKIT_ROOT}/traj_final/8192.npy \
dataloader.params.batch_size=12 \
agent.checkpoint_path=${ckpt} \
trainer.params.precision=32 \
experiment_name=${experiment_name} \
+cache_path=null \
metric_cache_path=${metric_cache_path} \
train_test_split=${split} \
#synthetic_sensor_path=$SYNTHETIC_SENSOR_PATH \
#synthetic_scenes_path=$SYNTHETIC_SCENES_PATH \
# +combined_inference=true \
# agent.config.vocab_path=${NAVSIM_DEVKIT_ROOT}/traj_final/8192.npy \
