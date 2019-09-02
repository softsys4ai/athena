#! /bin/bash

rootDir="experiment"
modelsDir="models"

dataSetName="cifar10" #"mnist"
numOfClasses=10

# Change training and testing paramters accordingly
# [Training parameters]
trainSamplesDir="training_samples"
numOfTrainSamples=10000
kFold=5

# [Testing paramters]
testSamplesDir="testing_samples"
numOfTestSamples=10000
testResultFoldName="test"

python train.py "$trainSamplesDir" "$rootDir" "$modelsDir" "$numOfTrainSamples" "$kFold" "$dataSetName" "$numOfClasses"

experimentRootDir=$(cat "current_experiment_root_dir_name.txt")

python test_ensemble_model_on_all_types_of_AEs.py "$testSamplesDir" "$experimentRootDir" "$modelsDir" "$numOfTestSamples" "$testResultFoldName" "$dataSetName" "$numOfClasses"

# collect new labels from each ensemble model for the give image dataset
#inputImagesFP="${testSamplesDir}/BS-${dataSetName}-clean.npy" # change it accordingly
#python generate_data_from_ensemble_models.py "$inputImagesFP" "$modelsDir" "$experimentRootDir" "$dataSetName" "$numOfClasses"

# train new target models
#python train_new_target_models.py "$inputImagesFP" "$experimentRootDir"  "$dataSetName" "$numOfClasses"
