% Derive from: https://www.mathworks.com/help/deeplearning/ug/gradcam-explains-why.html
% https://www.mathworks.com/help/deeplearning/ug/investigate-classification-decisions-using-gradient-attribution-techniques.html
% https://www.mathworks.com/help/deeplearning/ug/visualize-image-classifications-using-maximal-and-minimal-activating-images.html
% https://www.mathworks.com/help/deeplearning/ug/visualize-activations-of-a-convolutional-neural-network.html
close all;
clear;
clc;

net = googlenet;
inputSize = net.Layers(1).InputSize(1:2);
img = imread("sherlock.jpg");
img = imresize(img,inputSize);
[classfn,score] = classify(net,img);
imshow(img);
title(sprintf("%s (%.2f)", classfn, score(classfn)));

lgraph = layerGraph(net);
lgraph = removeLayers(lgraph, lgraph.Layers(end).Name);
dlnet = dlnetwork(lgraph);
softmaxName = 'prob';
% featureLayerName = 'inception_5b-output';
featureLayerName = 'conv2-norm2';

dlImg = dlarray(single(img),'SSC');
[featureMap, dScoresdMap] = dlfeval(@gradcam, dlnet, dlImg, softmaxName, featureLayerName, classfn);

gradcamMap = sum(featureMap .* sum(dScoresdMap, [1 2]), 3);
gradcamMap = extractdata(gradcamMap);
gradcamMap = rescale(gradcamMap);
gradcamMap = imresize(gradcamMap, inputSize, 'Method', 'bicubic');

imshow(img);
hold on;
imagesc(gradcamMap,'AlphaData',0.2);
colormap jet
hold off;
title("Grad-CAM");

function [featureMap,dScoresdMap] = gradcam(dlnet, dlImg, softmaxName, featureLayerName, classfn)
    [scores,featureMap] = predict(dlnet, dlImg, 'Outputs', {softmaxName, featureLayerName});
    classScore = scores(classfn);
    dScoresdMap = dlgradient(classScore,featureMap);
end

