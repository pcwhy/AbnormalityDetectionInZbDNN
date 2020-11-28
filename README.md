# Abnormality Detection and Continual Learning on Zero-bias Deep Neural Networks (DNN)

This repository hosts vital files of my work using zero-bias deep neural networks for abnormality detection. 

Please be noted that this repository will be updated continuously.

## Datasets
Remember to download dataset (real ADS-B signals, in I/Q format with a sample rate 8MHz, at Daytona Beach International Airport) from the following urls:

Direct Link:

https://drive.google.com/uc?export=download&id=1N-eImoAA3QFPu3cBJd-1WIUH0cqw2RoT

Or:
https://ieee-dataport.org/open-access/24-hour-signal-recording-dataset-labels-cybersecurity-and-iot

## Code files descriptions
### zbAbnormalityDetectionRealData.m
is the main entry using zero-bias DNN for abnormality detection as well as visualizing the decision boundaries of zero-bias DNN

### decisionVoronoiMNISTWithAbnormal.m
is a toy example using MNIST data to show the normal and abnormal data in zero-bias DNN with Voronoi diagram to depict the decision boundaries.

### abContinualLearning.m
Implement continual learning on zero-bias DNN. Especially, it contains codes for Elastic Weight Consolidation and how to use custom loop to train a DNN on real data.

### hypersphereCoverageTest.m
A numerical simulation to show theoretically how many different new classes (fingerprints) can be learned from the perspective of zero-bias DNN.
