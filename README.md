# CID
# Kaggle competition: IEEE's Signal Processing Society - Camera Model Identification

This code provides a simple solution based on transfer learning (ResNet50) for the camera model identification kaggle competion. The solution:
* achieves 50% accuracy on public leaderboard
* achieves 90% validation accuracy for training set and 95% accuracy for manipulated training set.
* takes about 16 hours to run on a macbook pro (no GPUs) -- most of the time is for one time computation
* the data files created by the code will take about 30 GB of space on the hard disk

The code has been tested, but there is no guarantee of correctness. If you do find bugs, please let me know.

How to use the code:
1. This code used keras/tensorflow with python 3.5 -- see the conda environment file
2. Place the training data in folder data/train, place testing data in the folder data/test. The folder data/train should have 10 subdirectories corresponding to 10 camera models.
3. Generate manipulated training data by running data/create_manip_data_py - this should take about one hour to run
4. Extract patches and compute resnet features by running data/create_resnet.py -- this should take about 12 hours to run 
5. Run training and make predictions by running Network/run_network.py -- this should take about an hour


The overall solution simply computes resnet features for patches extracted from close to center of the image. We train two models, one for unaltered image and one for manipulated images and use the corresponding model to make predictions.

