#! /bin/bash

rootDir="experiment"
modelsDir="models"

# [Testing paramters]
testSamplesDir="testing_samples"
numOfTestSamples=10000
testResultFoldName="test"

rootDir="detectionExperiment"

python detection_as_defense.py "$testSamplesDir" "$rootDir" "$modelsDir" "$numOfTestSamples" "$testResultFoldName"



