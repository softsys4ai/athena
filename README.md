# Adversarial Transformers

## Table of Contents
1. [**Dependencies**](#1-dependencies)
2. [**Attacks and Transformations**](#2-attacks-and-transformations)
3. [**Project Structure**](#3-project-structure)
4. [**Getting Started**](#4-getting-started)
5. [**How to Contribute**](#5-how-to-contribute)
6. [**References**](#6-references)

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

Download or clone the repository:
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

### White box attacks

**Attacks** | **Description**
--- | ---
Deep Fool | n/a
FGSM | n/a
BIM | n/a
CW | n/a
JSMA | n/a
PGD | n/a
One Pixel | n/a
MIM | n/a

### Transformations
**Transform** | **Description**
--- | ---
Rotate | n/a
Shift | n/a
Flip | n/a
Affine Transform | n/a
Morph Transform | n/a
Augment | n/a
Cartoonify | n/a
Quantize | n/a
Distort | n/a
Filter | n/a
Add noise | n/a
Compress | n/a
De-noising | n/a
Geometric transformations | n/a
Segmentations | n/a

### Defenses

**Defenses** | **Description**
--- | ---
 | n/a
 | n/a
 | n/a
 | n/a


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
  |-- utils (utils for all)<br>
  |<br>
  |-- visualization (visualization representation approaches)<br>
  |<br>
  |<br>
  |<br>
  |<br>

TODO

## 4. Getting Started

1. Follow the "Manual Installation" instructions in the "Dependencies" section to install all software requirements.
2. TODO

## 5. How to Contribute
TODO

## 6. References
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