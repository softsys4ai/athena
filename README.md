# Adversarial Transformers

## Table of Contents
1. [**Dependencies**](#1-dependencies)
2. [**Attacks and Transformations**](#2-attacks-and-transformations)
3. [**Project Structure**](#3-project-structure)
4. [**Getting Started**](#4-getting-started)
5. [**How to Contribute**](#5-how-to-contribute)
6. [**Citation and References**](#6-citation-and-references)

## Introduction
This repository hosts the Adversarial Transformers project source code. This code is currently being developed to support the "Ensembles of Many Weak Defenses are Strong: Defending Deep Neural Networks Against Adversarial Attacks" publication. The main finding of this publication is that many weak learners, neural networks trained on disjointly transformed datasets, can be combined into an ensemble network to effectively and efficiently defend against neural adversarial attacks. This codebase provides a framework for training weak learners with transformations, building ensemble of weak learners, provides implemented attack methods to test the effectiveness of ensembled weak learners, and provides all other subsequent source code and tooling to replicate the results of the experiments that will be presented in this publication.

## Dependencies
### Software Requirements
**Package** | **Version**
--- | ---
pillow | n/a
scipy | n/a
scikit-learn | n/a
cleverhans | 3.0.1
tensorboard | 1.13.1
tensorflow | 1.13.1
tensorflow-gpu | 1.13.1
numpy | n/a
scikit-image | 0.15.0
opencv-python | 4.1.0
keras | 2.2.5

#### Manual Installation

Clone the repository:
```
git clone git@github.com:softsys4ai/adversarial_transformers.git
```
Navigate to the directory where the repository was downloaded and run the following commend to install software dependencies:
```
pip install requirements.txt
```

### Hardware Requirements
Processor (recommended): 2 x 3.0 GHz CPU cores</br>
Memory: >=16 GB RAM</br>
Disk: >=30 GB available on disk partition</br>
Storage: >=20 MBps


## Attacks and Transformations
This section provides a brief overview of currently implemented transformations, attacks, and defenses.</br>

Each of the transformations are used to transform the entire training dataset. Once transformed, another neural network is trained on the transformed dataset. Each neural network trained on a transformed dataset is what we have coined as a Weak Learner. A composition of these networks trained on transformed datasets is coined as an Ensemble of Weak Learners.</br>

Each of the attacks listed below are used to test an Ensemble of Weak Learners. These ensembles are attacked with these methods to determine the effectiveness of an Ensemble of Weak Learner defenses against each type of adversarial attack.</br>

### Attacks

**Attacks** | **Type** | **Description**
--- | --- | ---
Deep Fool | whitebox | Simple algorithm to efficiently fool deep neural networks. [(Moosavi-Dezfooli et al., 2015)](https://arxiv.org/abs/1511.04599)
FGSM (Fast Gradient Signed Method) | whitebox | Simple and fast method of generating adverserial examples. [(Goodfellow et al., 2014)](https://arxiv.org/abs/1412.6572)
BIM (Basic Iterative Method) | whitebox | Basic iterative white box attack method to fool deep neural networks [(Kurakin et al., 2016)](https://arxiv.org/abs/1607.02533)
CW (Carlini and Wagner) | whitebox | Attack algorithm effective against distillation as a defense. [(Carlini and Wagner, 2016)](https://arxiv.org/abs/1608.04644)
JSMA (Jacobian-based Saliency Map Attack) | whitebox | Uses an understanding of the mapping between input and output to trick deep neural nets. [(Papernot et al., 2016)](https://arxiv.org/abs/1511.07528)
PGD (Projected Gradient Descent) | whitebox | Generates adverserial examples through gradient ascent. [(Madry et al., 2017)](https://arxiv.org/abs/1706.06083)
One Pixel | whitebox | Low cost generation of adverserial images, by only modifying one pixel of an image, with differential evolution. [(Su et al., 2019)](https://arxiv.org/abs/1710.08864)
MIM (Momentum Iterative Method) | whitebox | Uses momentum to stabilize update directions for crafting advserial examples, escaping poor local maxima. [(Dong et al., 2018)](https://arxiv.org/abs/1710.06081)

### Transformations
**Transform** | **Description**
--- | ---
Rotate | Rotation of images by 90, 180, 270 degrees
Shift | Shift of images left, right, up, down, top left, top right, bottom left, bottom right
Flip | Flip of images horizontally, vertically, and both
Affine Transform | Vertical compression, vertical stretch, horizontal compress, vertical stretch, both compress, both stretch
Morph Transform | erosion, dilation, opening, closing, gradient
Augment | samplewise standard norm and featurewise standard norm
Cartoonify | Mean type 1, mean type 2, mean type 3, mean type 4, gaussian type 1, gaussian type 2, gaussian type 3, gaussian type 4
Quantization | 2 clusters, 4 clusters, 8 clusters, 16 clusters, 32 clusters, 64 clusters
Distortion | x-axis and y-axis
Filter | entropy, gaussian, maximum, median, minimum prewitt, rank, roberts, scharr, sobel
Noising | gaussian, localvar, pepper, poison, salt, salt & pepper
Compression | jpeg quality 10, 30, 50, 80 and png compression 1, png compression 5, png compression 8
De-noising | Non-local means fast, non-local means, Bregmann total variation denoising, 
Geometric | iradon, iradon sart, swirl
Segmentation | gradient

### Defenses
**Defenses** | **Description**
--- | ---
EDMV-probs (Efficient Two-Step Defense for Deep Neural Networks) | Cheap defense strategy to combat adversarial examples that has a robustness comparable to using expensive, multi-step adversarial examples. [(Chang et al., 2018)](https://arxiv.org/abs/1810.03739)
EDAP-probs (Defense Against Universal Adversarial Perturbations) | Defense strategy to effectively defend against unseen adversarial examples in the real world by introducing pre-input detection of purturbations. [Akhtar et al., 2018](https://arxiv.org/abs/1711.05929)
EDMV-logits | The same as EDMV-probs but substituting DNN logit outputs for output probabilities.
EDAP-logits | The same as EDAP-probs but substituting DNN logit outputs for output probabilities.

## 3. Project Structure
Navigation of project and component hierarchy

project<br>
  |-- attacks (python files implement adversarial attack approaches)<br>
  |      &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|-- attacker.py (main entrance of AE generation, orchestrates all latter scripts)<br>
  |      &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|-- whitebox.py (generating AEs using APIs provided by cleverhans)<br>
  |      &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|-- one_pixel.py (implements one-pixel attack)<br>
  |      &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|-- carlini_wagner_l2.py (implements CW attack)<br>
  |      &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|-- momentum_iterative_method.py (implements MIM attack)<br>
  |<br>
  |-- utils (utils for all)<br>
  |      &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|-- config.py (stores project wide configurations and object definitions)<br>
  |      &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|-- file.py (contains helper functions to read/write evalution results from/to disk)<br>
  |<br>
  |-- scripts (scripts for experiments)<br>
  |      &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|-- craft_adversarial_examples.py (creates AEs given a model, dataset, and attack type)<br>
  |<br>
  |-- data.py (contains helper functions to load and normalize all datasets)
  |<br>
  |-- models.py (builds, compiles, trains, and evaluates models)
  |<br>
  |-- train.py (trains weak learners on each transformation and compositions of transformations)<br>
  |<br>
  |-- test_ensemble_model_on_all_types_of_AEs.py (creates and evaluates ensembles of weak defenses on all attack types)<br>
  |</br>


## Getting Started

### How do I install the project?
1. Navigate to the ["Manual Installation"](#manual-installation) instructions sub-section to install all software requirements.
2. Use the following tutorials to get up and running


### How do I configure/change project parameters?

```
    Script: adversarial_transformers/utils/config.py
    Description:
    This class contains a plethora of variables and class definitions for use across the project.
    
    Command Line Arguments
    ----------------------
    none.
    
    Methods
    -------
    class DATA(object)
        Class to contain information about selected datasets.

    class TRANSFORMATION(object)
        Contains a dictionary of supported transformations and strings used to reference them.

    class ATTACK(object)
        Contains a dictionary of supported attacks and strings used to reference them and attack specific parameter values; 

    class MODEL(object)
        Contains a model's architecture, training/testing dataset, and training hyperparameters.
    
    class MODE(object)
        Contains a "DEBUG" boolean value (default is false) to set the project to toggle the project in and out of debug mode.

    class PATH(object)
        Contains variables containing the absolute path of the project as well as the relative paths of important project resources, logging, and save locations.
```


### How do I load a dataset?
#### Available datasets
**Dataset Name** | **Description**
--- | ---
MNIST | Grayscale 28x28 pixel handwritten digit dataset (10 classes) containing 60,000 training and 10,000 validation examples.
Fashion-MNIST | Grayscale 28x28 pixel clothing dataset (10 classes) containing 60,000 training and 10,000 validation examples. 
CIFAR-10 | RGB 32x32 pixel dataset (10 classes) containing 50,000 training and 10,000 validation examples.
CIFAR-100 | RGB 32x32 pixel dataset (100 classes) containing 50,000 training and 10,000 validation examples.

```
    Script: adversarial_transformers/data.py
    Description:
    Generally, the use of this dataset loading class should be left for usage in our scripts. However, if you desire to load a dataset for your own experimentation, you may use this class to do so.
    
    Command Line Arguments
    ----------------------
    none.
    
    Methods
    -------
    load_data(dataset)
        Returns four variables: (X_train, Y_train), (X_test, Y_test). "X_train" and "Y_train" contain neural network inputs and outputs, respectively, for training a neural network. "X_test" and "Y_test" contain neural network inputs and outputs, respectively, for validating the accuracy of the neural network.
    normalize(X)
        Returns one variable: X. Normalizes the four dimensional (num. data samples, width, height, depth) input dataset X in order to use it for training. Normalization scales down dataset values, while preserving relative differences, for use as input to a neural network.
```


### How do I use attack methods and craft adversarial examples?

```
    Script: adversarial_transformers/scripts/craft_adversarial_examples.py
    Description:
    To craft adversarial examples provide this script with the name of a dataset and the name of the attack method you would like to use to generate examples.
    
    Command Line Arguments
    ----------------------
    none.
    
    Methods
    -------
    craft(dataset, method)
        Saves adversarial examples to the ADVERSARIAL_FILE path specified in the config.py file.
```

### How do I create and train a vanilla model on a vanilla dataset?

```
    Script: adversarial_transformers/models.py
    Description:
    To create and train a weak defense you may use this script. To properly train a weak defense, create a model and then train the model on your dataset with an applied transformation.
    
    Command Line Arguments
    ----------------------
    none.
    
    Methods
    -------
    create_model(dataset, input_shape, nb_classes)
        Returns a convolutional neural network model built with a representational capacity best suited for the specific dataset that you are using.

    train_model(model, dataset, model_name, need_augment=False, **kwargs)
        Returns a trained version of the model provided. Training hyperparameters can be found both in this method and the config.py file if you would like to change any of them for your use case.

    train(model, X, Y, model_name, need_augment=False, **kwargs)
        Returns a trained model. This method is used by the train_model() method but can be called as a standalone method if you choose to train a model on a custom dataset.

    evaluate_model(model, X, Y)
        Returns the following variables: acc, ave_conf_correct, ave_conf_miss. This method consumes a model and the test dataset in order to evaluate the accuracy of the model. Accuracy is defined as the percentage of correct classifications by the provided model.

    save_model(model, model_name, director=PATH.MODEL)
        Serializes and saves the model to disk at the path created by concatenating the paths provided in the director and model_name parameters. 

    load_model(model_name, director=PATH.MODEL)
        Returns a tensorflow model. Loads and compiles a saved model from disk at the path created by concatenating the paths provided in the director and model_name parameters.
```

### How do I create and train weak defenses?

```
    Script: adversarial_transformers/train.py
    Description:
    This script trains a weak defense for each type of transformation and saves each model to the specified models directory to be used to build an ensemble.
    
    Command Line Arguments
    ----------------------
    samplesDir : str
        File path of input images to train the weak defense
    rootDir : str
        File path of the directory of the project
    modelsDir : str
        File path of directory to store trained weak defenses
    numOfSamples : int
        Upper bound of the number of the dataset input indices to be used for training/validation.
    kFold : int
        Number of folds, used to determine the validation training set size.
    datasetName : str
        Name of the dataset to be used for training (mnist, fmnist, cifar10, cifar100)
    numOfClasses
        Number of output classes of the provided dataset

    Methods
    -------
    usage()
        Call this method to print instructions for how to use this script to your standard output stream.
```

### How do I construct and evaluate an ensemble of weak defenses?

    Now that you have trained weak defenses, you are likely wondering, how do I build an ensemble of these weak defenses to defend against adversarial attacks? In our project, we construct and use ensembles by loading all of the trained weak defenses from a specified models directory, perform inference on a given subset of a dataset with each of the weak defenses, and then save all of the models' output probabilities and logits to disk at a specified prediction result directory for further analysis.

```
    Script: adversarial_transformers/test_ensemble_model_on_all_types_of_AEs.py
    Description:
    This script evaluates the ensemble model, the model composed of all of the models trained with train.py, and saves all of the results to a specified test result folder.
    
    Command Line Arguments
    ----------------------
    samplesDir : str
        Path to directory containing the correct output labels for the network.
    experimentRootDir : str
        Path to directory of the root of the project.
    modelsDir : str
        Path to directory where all of the trained models are saved.
    numOfSamples : int
        Upper bound of the number of samples to use for evaluation of the ensemble.
    testResultFoldName : str
        Path to directory where test results are to be stored.
    datasetName : str
        Name of the dataset to be used for training (mnist, fmnist, cifar10, cifar100)
    numOfClasses : int
        Number of output classes for the provided dataset.
    
    Methods
    -------
    none.
```

Evaluation Strategy | Description
--- | ---
Black box | Attacker has no knowledge of the internal model architecture.
Gray box | Attacker only knowledge of weak defense architectures but does not know how the ensemble combines the outputs of the weak defenses.
White box | Attacker has knowledge of the entire ensemble architecture, including how the ensemble combines the outputs of the weak defenses.

Each evaluation, performed for each attack type, is done with an ensemble network constructed in three different ways: with the k-best weak defenses, with the k-worst weak defenses, and with k-random weak defenses. The value k starts at one and is incremented by one after each evaluation in order to test the accuracy of the ensemble with an increasing number of weak defenses. The best and worst weak defenses are determined by testing weak defenses against a subset of adversarial examples produced by a given attack type, ranking them, and choosing the k-best and k-worst. k-random weak defenses are chosen as one would expect, randomly.


### What other scripts should I be aware of?

```
    Script: adversarial_transformers/attacks/attacker.py
    Description:
    This script calls orchestrates and calls all other attack scripts in its directory to produce adversarial examples with a given attack type on a given, trained model.
    
    Command Line Arguments
    ----------------------
    none.

    Methods
    -------
    get_adversarial_examples(model_name, attack_method, X, Y, **kwargs)
        Returns the following variables: X_adv and Y. "X_adv" is a vector containing all of the generated adversarial examples and Y is a vector of the same length containing the correct class labels for each created adversarial example.
```
```
    Script: adversarial_transformers/utils/file.py
    Description:
    This is a helper script that provides methods to read and write ensemble model evaluations from and to disk. The methods contained within this script may be called on their own. Currently, this script's functions are leveraged by the "test_ensemble_model_on_all_types_of_AEs.py" script for recording produced ensemble evaluation results.
    
    Command Line Arguments
    ----------------------
    none.

    Methods
    -------
    dict2csv(dictionary, file_name, list_as_value=False)
        Saves a given dictionary to a csv file located at the path specified by the "file_name" parameter. If only a file name is specified, the csv will be saved in the adversarial_transformers/utils directory.
    csv2dict(file_name, orient=ORIENT.COL, dtype='float')
        Reads a csv at the path specified by the "file_name" parameter and returns the dictionary stored in the CSV file.
    save_adv_examples(data, **kwargs)
        Saves adversarial examples provided in the "data" parameter to the path specified by the "ADVERSARIAL_FILE" variable in the "config.py" script.
```

## How to Contribute


We welcome new features or enhancements of this framework. Bug fixes can be initiated through GitHub pull requests. 


## Citation

```
@article{ying2020,
  title={Ensembles of Many Weak Defenses can be Strong: Defending Deep Neural Networks Against adversarial Attacks},
  author={Meng, Ying and Su, Jianhai and O’Kane, Jason and Jamshidi, Pooyan},
  year={2020}
}
```

## Contacts

* Pooyan Jamshidi (pooyan.jamshidi@gmail.com)
* Ying Meng (y.meng201011@gmail.com)

## Contributors

* [Ying Meng](https://github.com/MENG2010)
* [Jianhai Su](https://github.com/oceank)
* [Blake Edwards](https://github.com/blakeedwards823)
* [Stephen Baione](https://github.com/StephenBaione)
* [Pooyan Jamshidi](https://github.com/pooyanjamshidi)


## Acknowledgements
This project has been partially supported by:
* Google via GCP cloud research credits
* NASA (EPSCoR 521340-SC001) 