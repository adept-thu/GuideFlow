export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="/nas/users/perception-users/yuguanyi/playground_2506/code/GTRS/dataset/maps"
export NAVSIM_EXP_ROOT="/nas/users/perception-users/yuguanyi/playground_2506/code/GTRS/exp"
export NAVSIM_DEVKIT_ROOT="/nas/users/perception-users/yuguanyi/playground_2506/code/GTRS"
export OPENSCENE_DATA_ROOT="/nas/users/perception-users/yuguanyi/playground_2506/code/GTRS/dataset"

TRAIN_TEST_SPLIT=navtrain #navhard_two_stage
CACHE_PATH=$NAVSIM_EXP_ROOT/${TRAIN_TEST_SPLIT}_metric_cache

PYTHONPATH=$PYTHONPATH:./ python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_metric_caching.py \
train_test_split=$TRAIN_TEST_SPLIT \
metric_cache_path=$CACHE_PATH
