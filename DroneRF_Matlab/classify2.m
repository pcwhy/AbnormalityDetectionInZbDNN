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
XTrain = XTrain(:,:,:,perm);
% XTrain = raw(perm,:);
YTrain = YTrain(perm);
XVal = XTrain(:,:,:,end - round(0.3*size(XTrain,4)):end);
% XVal = XTrain(end - round(0.3*size(XTrain,1)):end,:);
YVal = YTrain(end - round(0.3*size(YTrain,1)):end);
XTrain = XTrain(:,:,:,1:round(0.7*size(XTrain,4)));
% XTrain = XTrain(1:round(0.7*size(XTrain,1)),:);
YTrain = YTrain(1:round(0.7*size(YTrain,1)));
numClasses = size(unique(YTrain),1);

layers = [
    imageInputLayer([32,32,2], 'Name', 'input')
    convolution2dLayer(5,10, 'Name', 'conv2d_1')
    batchNormalizationLayer('Name', 'batchNorm_1')
    reluLayer('Name', 'relu_1')
    convolution2dLayer(3, 10, 'Padding', 1, 'Name', 'conv2d_2')
    reluLayer('Name', 'relu_2')    
    convolution2dLayer(3, 10, 'Padding', 1, 'Name', 'conv2d_3')
    reluLayer('Name', 'relu_3')    
    additionLayer(2,'Name', 'add_1')
%     depthConcatenationLayer(2,'Name','add_1')    
%     batchNormalizationLayer('Name', 'batchNorm_2')
    fullyConnectedLayer(numClasses, 'Name', 'fc_bf_fp') % 11th
    %tanhLayer('Name','relu_4')
    %reluLayer('Name','relu_4')
    %preluLayer(numClasses,'prelu')
    
    zeroBiasFCLayer(numClasses,numClasses,'Fingerprints',[])
    
    %fullyConnectedLayer(numClasses, 'Name', 'Fingerprints') % 11th
    %dropoutLayer(0.3,'Name','dropOut_1')            
    softmaxLayer('Name', 'softmax_1')
    classificationLayer('Name', 'classify_1')
    ];


lgraph = layerGraph(layers);
lgraph = connectLayers(lgraph, 'relu_1', 'add_1/in2');
%lgraph = connectLayers(lgraph, 'relu_2', 'add_1/in3');
plot(lgraph);
options = trainingOptions('sgdm',...
    'Plots', 'training-progress',...
    'ExecutionEnvironment','auto',...
    'ValidationData',{XVal,categorical(YVal)},...
    'MaxEpochs', 3, ...
    'MiniBatchSize',128,...
    'L2Regularization',0.05);
[net,info] = trainNetwork(XTrain, categorical(YTrain), lgraph, options);

YPred = classify(net,XVal);
accuracy = sum(YPred == YVal)/numel(YVal)

lgraph2 = layerGraph(net); % Also collect old weights
% % OR:
lgraph2 = lgraph2.removeLayers('classify_1');
dlnet = dlnetwork(lgraph2);
% save dlnet








