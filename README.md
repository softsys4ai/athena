# Adversarial Transformers

## Table of Contents
1. [**Dependencies**](#1-dependencies)
2. [**Attacks and Transformations**](#2-attacks-and-transformations)
3. [**Project Structure**](#3-project-structure)
4. [**Getting Started**](#4-getting-started)
5. [**How to Contribute**](#5-how-to-contribute)
6. [**Citation and References**](#6-citation-and-references)

## 1. Dependencies
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
TODO understand maximum memory usage of the program


## 2. Attacks and Transformations
Overview of implemented transformations, attacks, defenses, detections, metrics, certifications, and verifications

### Attacks

**Attacks** | **Type** | **Description**
--- | --- | ---
Deep Fool | whitebox | Simple algorithm to efficiently fool deep neural networks. [(Moosavi-Dezfooli et al., 2015)](https://arxiv.org/abs/1511.04599)
FGSM (Fast Gradient Signed Method) | whitebox | Simple and fast method of generating adverserial examples. [(Goodfellow et al., 2014)](https://arxiv.org/abs/1412.6572)
BIM (Basic Iterative Method) | blackbox | Basic iterative black box attack method to fool deep neural networks [(Kurakin et al., 2016)](https://arxiv.org/abs/1607.02533)
CW (Carlini and Wagner) | whitebox | Attack algorithm effective against distillation as a defense. [(Carlini and Wagner, 2016)](https://arxiv.org/abs/1608.04644)
JSMA (Jacobian-based Saliency Map Attack) | whitebox | Uses an understanding of the mapping between input and output to trick deep neural nets. [(Papernot et al., 2016)](https://arxiv.org/abs/1511.07528)
PGD (Projected Gradient Descent) | whitebox | Generates adverserial examples through gradient ascent. [(Madry et al., 2017)](https://arxiv.org/abs/1706.06083)
One Pixel | blackbox | Low cost generation of adverserial images, by only modifying one pixel of an image, with differential evolution. [(Su et al., 2019)](https://arxiv.org/abs/1710.08864)
MIM (Momentum Iterative Method) | blackbox | Uses momentum to stabilize update directions for crafting advserial examples, escaping poor local maxima. [(Dong et al., 2018)](https://arxiv.org/abs/1710.06081)

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
  |      &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|-- attacker.py (main entrance of AE generation)<br>
  |      &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|-- whitebox.py (generating AEs using APIs provided by cleverhans)<br>
  |      &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|-- one_pixel.py (implements one-pixel attack)<br>
  |<br>
  |-- defense (defense strategies)<br>
  |<br>
  |-- evaluation (files evaluate models, attacks, ect.)<br>
  |<br>
  |-- scripts (scripts for experiments)<br>
  |<br>
  |-- test (``unit`` test scripts for this project)<br>
  |<br>
  |-- utils (utils for all)<br>
  |<br>
  |-- visualization (visualization representation approaches)<br>
  |<br>
  |<br>
  |<br>
  |<br>

attacker.py
whitebox.py       --> return the advserial examples with true labels --> all have generate() method
one_pixel
MIM 


## 4. Getting Started

### How do I install the project?
1. Navigate to the ["Manual Installation"](#manual-installation) instructions sub-section to install all software requirements.
2. Use the following tutorials to get up and running

### How do I load a dataset?

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

#### Available datasets
**Dataset Name** | **Description**
--- | ---
MNIST | Grayscale 28x28 pixel handwritten digit dataset (10 classes) containing 60,000 training and 10,000 validation examples.
Fashion-MNIST | Grayscale 28x28 pixel clothing dataset (10 classes) containing 60,000 training and 10,000 validation examples. 
CIFAR-10 | RGB 32x32 pixel dataset (10 classes) containing 50,000 training and 10,000 validation examples.
CIFAR-100 | RGB 32x32 pixel dataset (100 classes) containing 50,000 training and 10,000 validation examples.

### How do I configure/change project parameters?

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


### How do I craft adversarial examples?

    Script: adversarial_transformers/scripts/craft_adversarial_examples.py
    Description:
    To craft adversarial examples provide this script with the name of a dataset and the name of the attack method you would like to use to generate examples.
    
    Command Line Arguments
    ----------------------
    none.
    
    Methods
    -------
    craft(dataset, method)
        Saves adverserial examples to the ADVERSARIAL_FILE path specified in the config.py file.


### How do I create and train a weak defense?

    Script: adversarial_transformers/models.py
    Description:
    To create and train a weak defense you may use this script. To properly train a weak defense, first generate adversarial examples, load them, and porovide them 
    
    Command Line Arguments
    ----------------------
    none.
    
    Methods
    -------
    create_model(dataset, input_shape, nb_classes)
        Returns a convolutional neural network model built with a representational capacity best suited for the specific dataset that you are using.

    train_model(model, dataset, model_name, need_augment=False, **kwargs)
        Returns a trained version of the model provided. Training hyperparameters can be found both in this method and the config.py file if you would like to change any of them for your use case.

    evaluate_model(model, X, Y)
        Returns the following variables: acc, ave_conf_correct, ave_conf_miss. This method consumes a model and the test dataset in order to evaluate the accuracy of the model. Accuracy is defined as the percentage of correct classifications by the provided model.

    save_model(model, model_name, director=PATH.MODEL)
        Serializes and saves the model to disk at the path created by concatenating the paths provided in the director and model_name parameters. 

    load_model(model_name, director=PATH.MODEL)
        Returns a tensorflow model. Loads and compiles a saved model from disk at the path created by concatenating the paths provided in the director and model_name parameters.
    

### main idea of the work
target model (regular model trained on the clean dataset) --> original target model
offline
    image --> transformation --> new image --> weak defense model 
        each weak model are trained on one or a composition of transformation
online
    Transformation 1    Weak defense 1  -->     |
    Transformation 2    Weak defense 2  -->     |
    .                                           |--> Ensemble of strategies
    .                                           |
    .                                           |
    Transformation N    Weak defense N  -->     |

### Important Functions


### Main entrances
- attacks subfolder
- scripts --> tutorial / demo in this folder
    - craft adversarial examples --> explains how to call attacks subfolder, how to get AE, how to call parameters, etc.
- utils -->
    - config.py --> contains all of the configuration parameters, implemented attacks, implemented defenses
    - file.py --> saves adverserial examples 
    - plot.py --> helper functions related plotting for the publications
- ensemble is still in progress
- evaluation subfolder
    - not a general feature, ignore for now
- models.py
    - defines the neural network architectures
    - train the model here
- data.py --> loads data from the datasets
- visualizations --> production of visuals for the publication
- old_scripts 
    

- evaluating the final ensemble model
    - black box --> 
        - build their own dataset of inputs and outputs 
    - gray box --> attackers know the list of transformations, know everything about the weak defenses
        - the only thing they do not know is how the combination of the results work
    - white box --> attackers know everything, including the ensemble strategies

    - still working on gray box and white box
        - currently using any strategy to choose a weak defense target
        - for each of the weak defenses, 
        - concrete example
            - 72 types of transformations <-> 72 weak defenses
            - least effective --> choose one with lowest test accuracy 
            - most effective, least effective, random



### In what ways are models trained? What data are they trained on?


### How are weak defenses trained?


### How are models evaluated? How are ensembles evaluated?


### How are the transformations called? Where are they called? How are they used?


### How are the attack algorithms called? Where are they called? How are they used?


### 




### How to train a model of weak defenses?
each weak defense is a model trained on a transformation (train_new_target_models.py)
each of these models is saved to a specified training directory ()
train.py is then used to "train" the ensemble, aka determine the relationship of weak defenses for best defense
the ensemble can be then be used as a while with ()

### How to evaluate ensemble of weak defenses?
  util.py + CL parameters

    
### Script: train_new_target_models.py
    Description: 
    hello world
    
    Command Line Arguments
    ----------------------
    inputImagesFP : str
        file path of input images to train the weak defense
    experimentRootDir : str
        file path of the directory to save experiment
    datasetName : str
        name of the dataset to be used for training (mnist, fmnist, cifar10, cifar100)
    numOfClasses : int
        number of output classes of the provided dataset
    
    Methods
    -------
    
### Script: 
    Description: 
    hello world
    
    Command Line Arguments
    ----------------------
    
    
    Methods
    -------

### Script: 
    Description: 
    hello world
    
    Command Line Arguments
    ----------------------
    
    
    Methods
    -------

### Script: 
    Description: 
    hello world
    
    Command Line Arguments
    ----------------------
    
    
    Methods
    -------


## 5. How to Contribute

Adding new features (attacks, defenses, transformations), improving documentation, squashing bugs, or creating tutorials are examples of helpful contributions that could be made to this project. Additionally, if you are publishing a new attack, defense, or transformation, we highly encourage you to add it to this project or bring it to our communities attention by creating an enhancement issue.</br>

Bug fixes can be initiated through GitHub pull requests. When making code contributions to the project, we ask that you write descriptive comments for all significant functions of your addition, follow the [Python PEP 8 style guidelines](https://www.python.org/dev/peps/pep-0008/), and sign all of your commits using the -s flag or by appending "Signed-off-by: <Name>,<Email>" to your commit message.</br>


## 6. Citation and References

### Cite this work
```
@article{ying2019,
  title={Ensembles of Many Weak Defenses are Strong: Defending Deep Neural Networks Against Adverserial Attacks},
  author={},
  journal={},
  year={2019}
}
```

### References
[1] MariuszBojarski,DavideDelTesta,DanielDworakowski,Bern- hard Firner, Beat Flepp, Prasoon Goyal, Lawrence D Jackel, Mathew Monfort, Urs Muller, Jiakai Zhang, et al. End to end learning for self-driving cars. arXiv preprint arXiv:1604.07316, 2016. <br><br>
[2] N. Carlini and D. Wagner. Towards evaluating the robustness of neural networks. In 2017 IEEE Symposium on Security and Privacy (SP), pages 39–57, May 2017. <br><br>
[3] Nicholas Carlini and David Wagner. Magnet and” efficient de- fenses against adversarial attacks” are not robust to adversarial examples. arXiv preprint arXiv:1711.08478, 2017. <br><br>
[4] Moustapha Cisse, Piotr Bojanowski, Edouard Grave, Yann Dauphin, and Nicolas Usunier. Parseval networks: Improving robustness to adversarial examples. In Proceedings of the 34th In- ternational Conference on Machine Learning-Volume 70, pages 854–863. JMLR. org, 2017. <br><br>
[5] XiaoDing,YueZhang,TingLiu,andJunwenDuan.Deeplearn- ing for event-driven stock prediction. In Twenty-fourth interna- tional joint conference on artificial intelligence, 2015. <br><br>
[6] Gintare Karolina Dziugaite, Zoubin Ghahramani, and Daniel M Roy. A study of the effect of jpg compression on adversarial images. arXiv preprint arXiv:1608.00853, 2016. <br><br>
[7] Alhussein Fawzi, Omar Fawzi, and Pascal Frossard. Analysis of classifiers robustness to adversarial perturbations. Machine Learning, 107(3):481–508, 2018. <br><br>
[8] Ian J Goodfellow, Jonathon Shlens, and Christian Szegedy. Ex- plaining and harnessing adversarial examples. arXiv preprint arXiv:1412.6572, 2014. <br><br>
[9] Kathrin Grosse, Praveen Manoharan, Nicolas Papernot, Michael Backes, and Patrick McDaniel. On the (statistical) detection of adversarial examples. arXiv preprint arXiv:1702.06280, 2017. <br><br>
[10] Chuan Guo, Mayank Rana, Moustapha Cisse, and Laurens Van Der Maaten. Countering adversarial images using input transfor- mations. arXiv preprint arXiv:1711.00117, 2017. <br><br>
[11] KaimingHe,XiangyuZhang,ShaoqingRen,andJianSun.Deep residual learning for image recognition. In The IEEE Conference on Computer Vision and Pattern Recognition (CVPR), June 2016. <br><br>
[12] Warren He, James Wei, Xinyun Chen, Nicholas Carlini, and Dawn Song. Adversarial example defense: Ensembles of weak defenses are not strong. In 11th {USENIX} Workshop on Offen- sive Technologies ({WOOT} 17), 2017. <br><br>
[13] Warren He, James Wei, Xinyun Chen, Nicholas Carlini, and Dawn Song. Adversarial example defenses: Ensembles of weak defenses are not strong. CoRR, abs/1706.04701, 2017. <br><br>
[14] Geoffrey Hinton, Li Deng, Dong Yu, George Dahl, Abdel- rahman Mohamed, Navdeep Jaitly, Andrew Senior, Vincent Van- houcke, Patrick Nguyen, Brian Kingsbury, et al. Deep neural net- works for acoustic modeling in speech recognition. IEEE Signal processing magazine, 29, 2012. <br><br>
[15] Andrew Ilyas, Shibani Santurkar, Dimitris Tsipras, Logan En- gstrom, Brandon Tran, and Aleksander Madry. Adversar- ial examples are not bugs, they are features. arXiv preprint arXiv:1905.02175, 2019. <br><br>
[16] Tim Kraska, Alex Beutel, Ed H Chi, Jeffrey Dean, and Neoklis Polyzotis. The case for learned index structures. In Proceedings of the 2018 International Conference on Management of Data, pages 489–504. ACM, 2018. <br><br>
[17] Alexey Kurakin, Ian Goodfellow, and Samy Bengio. Adversar- ial machine learning at scale. arXiv preprint arXiv:1611.01236, 2016. <br><br>
[18] Yanpei Liu, Xinyun Chen, Chang Liu, and Dawn Song. Delv- ing into transferable adversarial examples and black-box attacks. arXiv preprint arXiv:1611.02770, 2016. <br><br>
[19] Jiajun Lu, Hussein Sibai, Evan Fabry, and David Forsyth. No need to worry about adversarial examples in object detection in autonomous vehicles. arXiv preprint arXiv:1707.03501, 2017. <br><br>
[20] AleksanderMadry,AleksandarMakelov,LudwigSchmidt,Dim- itris Tsipras, and Adrian Vladu. Towards deep learning models resistant to adversarial attacks. In International Conference on Learning Representations, 2018. <br><br>
[21] Dongyu Meng and Hao Chen. Magnet: a two-pronged defense against adversarial examples. In Proceedings of the 2017 ACM SIGSAC Conference on Computer and Communications Security, pages 135–147. ACM, 2017. <br><br>
[22] JanHendrikMetzen,TimGenewein,VolkerFischer,andBastian Bischoff. On detecting adversarial perturbations. arXiv preprint arXiv:1702.04267, 2017. <br><br>
[23] Seyed-Mohsen Moosavi-Dezfooli, Alhussein Fawzi, and Pascal Frossard. Deepfool: a simple and accurate method to fool deep neural networks. CoRR, abs/1511.04599, 2015. <br><br>
[24] Nicolas Papernot, Patrick McDaniel, Ian Goodfellow, Somesh Jha, Z. Berkay Celik, and Ananthram Swami. Practical black- box attacks against machine learning. In Proceedings of the 2017 ACM on Asia Conference on Computer and Communications Se- curity, ASIA CCS ’17, pages 506–519, New York, NY, USA, 2017. ACM. <br><br>
[25] Nicolas Papernot, Patrick McDaniel, Somesh Jha, Matt Fredrik- son, Z Berkay Celik, and Ananthram Swami. The limitations of deep learning in adversarial settings. In 2016 IEEE European Symposium on Security and Privacy (EuroS&P), pages 372–387. IEEE, 2016. <br><br>
[26] Razvan Pascanu, Jack W Stokes, Hermineh Sanossian, Mady Marinescu, and Anil Thomas. Malware classification with recur- rent networks. In 2015 IEEE International Conference on Acous- tics, Speech and Signal Processing (ICASSP), pages 1916–1920. IEEE, 2015. <br><br>
[27] Alexander Ratner, Dan Alistarh, Gustavo Alonso, Peter Bailis, Sarah Bird, Nicholas Carlini, Bryan Catanzaro, Eric Chung, Bill Dally, Jeff Dean, et al. Sysml: The new frontier of machine learn- ing systems. arXiv preprint arXiv:1904.03257, 2019. <br><br>
[28] Uri Shaham, Yutaro Yamada, and Sahand Negahban. Under- standing adversarial training: Increasing local stability of su- pervised models through robust optimization. Neurocomputing, 307:195–204, 2018. <br><br>
[29] Dinggang Shen, Guorong Wu, and Heung-Il Suk. Deep learning in medical image analysis. Annual review of biomedical engi- neering, 19:221–248, 2017. <br><br>
[30] Christian Szegedy, Wojciech Zaremba, Ilya Sutskever, Joan Bruna, Dumitru Erhan, Ian Goodfellow, and Rob Fergus. Intriguing properties of neural networks. arXiv preprint arXiv:1312.6199, 2013. <br><br>
[31] Wayne Xiong, Jasha Droppo, Xuedong Huang, Frank Seide, Mike Seltzer, Andreas Stolcke, Dong Yu, and Geoffrey Zweig. Achieving human parity in conversational speech recognition. CoRR, abs/1610.05256, 2016. <br><br>
