# Abnormality detection on Zero-bias Deep Neural Networks

This repository hosts vital files of my work using zero-bias deep neural networks for abnormality detection.

Remember to download dataset (real ADS-B signals, in I/Q format with a sample rate 8MHz, at Daytona Beach International Airport) from the following urls:

Direct Link:

https://drive.google.com/uc?export=download&id=1N-eImoAA3QFPu3cBJd-1WIUH0cqw2RoT

Or:
https://ieee-dataport.org/open-access/24-hour-signal-recording-dataset-labels-cybersecurity-and-iot

In which
zbAbnormalityDetectionRealData.m
is the main entry using zero-bias DNN for abnormality detection as well as visualizing the decision boundaries of zero-bias DNN

decisionVoronoiMNISTWithAbnormal.m
is a toy example using MNIST data to show the normal and abnormal data in zero-bias DNN with Voronoi diagram to depict the decision boundaries.
