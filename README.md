# A Comparative Analysis of Data Synthesis Techniques to Improve Classification Accuracy of Raman Spectroscopy Data

---

# Installation

The software dependancies for this project are open-source and managed using the Anaconda Distribution v23.1.0.

An enviroment.yml is provided to import the required dependencies.

Download [Anaconda](https://www.anaconda.com/download)

Dependencies:
* Python v3.9.12
* Tensorflow v2.9.1
* Keras v2.9.0
* NumPy v1.21.5
* Pandas v1.4.3

# Directory

This repository contains three folders required to reproduce the results.

The *data* folder contains the original data and synthetic data, which is produced from the *synthesis* folder. Text files including row (index) numbers are provided, which indicates the portion of the original data that compose each training fold. This was process was done manually to insure uniqueness and a maintained class prior probability in each fold.

The *synthesis* folder contains the python code for generating the synthetic data sets for the three folds of original data, for both the *Weighted Blending* and *Variational Autoencoder (VAE)*

The *experiments* folder contains the code to load the data, augment synthetic data and train the Deep Learning algorithms independantly. The user must update the code where appropriate to select the correct path for loading the data and storing results.


# Contact Us
For any questions related to this work please contact the authors:
* A.flanagan18@universityofgalway.ie
* frank.glavin@universityofgalway.ie

# Acknowledgements
* This work was supported by the Science Foundation of Ireland Centre for Research Training in Artificial Intelligence (Science Foundation Ireland Grant No-18/CRT/6223)
* We acknowledge the provision of the chlorinated dataset files by [Analyze IQ](https://www.analyzeiq.com/) Limited