clc;
clear;
close all;
addpath('../../../');
load('../Data/RF_Data2.mat');
% RFData = table2array(RFData);
raw = DATA';
l1 = Label(1,:)';
l2 = Label(2,:)';
l3 = Label(3,:)';
% raw = RFData(1:2048,:)';
% l1 = RFData(2049,:)'; % RF activities or not
% l2 = RFData(2050,:)'; % RF background and three drones
% l3 = RFData(2051,:)'; % RF background and three drones with different activities.
mask = logical((sum(raw.^2,2)>0.00001)+(l1==0));
raw = raw(mask,:);
YTrain = categorical(l3);
YTrain = YTrain(mask,:);
count = size(raw,1);
XTrain = reshape(raw',[32,32,2,count]);

perm = randperm(numel(YTrain));
% XTrain = XTrain(:,:,:,perm);
XTrain = raw(perm,:);
YTrain = YTrain(perm);
% XVal = XTrain(:,:,:,end - round(0.3*size(XTrain,4)):end);
XVal = XTrain(end - round(0.3*size(XTrain,1)):end,:);
YVal = YTrain(end - round(0.3*size(YTrain,1)):end);
% XTrain = XTrain(:,:,:,1:round(0.7*size(XTrain,4)));
XTrain = XTrain(1:round(0.7*size(XTrain,1)),:);
YTrain = YTrain(1:round(0.7*size(YTrain,1)));

layers = [
        featureInputLayer(2048)
%      imageInputLayer([32 32 2])
    
%     convolution2dLayer(3,8,'Padding','same')
     batchNormalizationLayer
%     reluLayer
    
%     maxPooling2dLayer(2,'Stride',2)
%     
%     convolution2dLayer(3,16,'Padding','same')
%     batchNormalizationLayer
%     reluLayer
%     
%     maxPooling2dLayer(2,'Stride',2)
%     
%     convolution2dLayer(3,32,'Padding','same')
%     batchNormalizationLayer
%     reluLayer
%     tensorVectorLayer('Flatten')
    fullyConnectedLayer(128)
    reluLayer
    batchNormalizationLayer

    fullyConnectedLayer(128)
    reluLayer
    batchNormalizationLayer

    fullyConnectedLayer(128)
    reluLayer
    batchNormalizationLayer

    fullyConnectedLayer(numel(categories(YTrain)))
    dropoutLayer(0.2)

%     FCLayer(1568,10,'fc_bf_fp',[])
    zeroBiasFCLayer(numel(categories(YTrain)),numel(categories(YTrain)),'zb_fp',[])
    softmaxLayer
    classificationLayer];

options = trainingOptions('sgdm', ...
    'InitialLearnRate',0.01, ...
    'MiniBatchSize',1024,...
    'MaxEpochs',1, ...
    'Shuffle','every-epoch', ...
    'ValidationData',{XVal,YVal}, ...
    'ValidationFrequency',30, ...
    'Verbose',false, ...
    'Plots','training-progress');
net = trainNetwork(XTrain, YTrain,layers,options);
YPred = classify(net,XVal);
accuracy = sum(YPred == YVal)/numel(YVal)

lgraph2 = layerGraph(net.Layers); % Also collect old weights
% % OR:
lgraph2 = lgraph2.removeLayers('classoutput');
dlnet = dlnetwork(lgraph2);