TRAIN_TEST_SPLIT=navtrain

python $NAVSIM_DEVKIT_ROOT/navsim_train/planning/script/run_training.py \
agent=flowdrive_agent \
experiment_name=training_flowdrive_agent \
train_test_split=$TRAIN_TEST_SPLIT \
