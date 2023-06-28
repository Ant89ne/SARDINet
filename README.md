# SARDINet

This repository is dedicated to the implementation of the *SAR Distorted Image translator Network* (SARDINet). This neural network aims at translating SAR distorted images into optical ones. 

One can find a detailed description of the network and the results in [Deep Learning of Radiometrical and Geometrical Sar Distorsions for Image Modality translations](https://ieeexplore.ieee.org/document/9897713).

The code is the baseline to compute the results obtained in the paper. 

### Guidelines

Data used in the paper are available on IEEE Dataport :

*Abdourrahmane ATTO, January 9, 2022, "Can Artificial Intelligence Untangle Distorted and Compressed Geometries Associated with SAR Images of 3D Objects ? ", IEEE Dataport, doi: https://dx.doi.org/10.21227/y3pm-1113.*



The code was computed using Python 3.10.6 and the packages versions detailed in the file `requirements.txt`.

In order to run the code please follow the next steps :

* Open the `main.py` file
* Choose the path where your data are located and the path where you want to save the results
* Choose you hyperparameters, whether to compute an adversarial training and your loss functions
* If you want to modify the architecture of the network, you'll find all you need in the `TransNet.py` file. 
* Save your changes
* Run the `main.py` file

### Citation

If this work was useful for you, please ensure citing our works : 

*BRALET, Antoine, ATTO, Abdourrahmane M., CHANUSSOT, Jocelyn, et al. Deep Learning of Radiometrical and Geometrical Sar Distorsions for Image Modality translations. In : 2022 IEEE International Conference on Image Processing (ICIP). IEEE, 2022. p. 1766-1770.*

Thank you for your support

### Any troubles ?

If you have any troubles with the article or the code, do not hesitate to contact us !