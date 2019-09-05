#! /bin/bash

sampleType=$1

rootDir="experiment"
modelsDir="models"

dataSetName="mnist"
numOfClasses=10

# [Testing paramters]
testSamplesDir="testing_samples"
numOfTestSamples=10000
testResultFoldName="test"


experimentRootDir=$(cat "current_experiment_root_dir_name.txt")

mkdir "$rootDir"
mkdir "$experimentRootDir"

python -u predict_testset.py "$testSamplesDir" "$experimentRootDir" "$modelsDir" "$numOfTestSamples" "$testResultFoldName" "$dataSetName" "$numOfClasses" "$sampleType"

