clear;
clc;

[XTrain,YTrain] = digitTrain4DArrayData;
YTrain = double(YTrain);
cond = YTrain >=9;
revCond = ~cond;
uX = XTrain(:,:,:,cond);
uY = YTrain(cond);
% uX = 2*randn(size(uX));
XTrain = XTrain(:,:,:,revCond);
YTrain = YTrain(revCond);
YTrain = categorical(YTrain);

perm = randperm(numel(YTrain));
XTrain = XTrain(:,:,:,perm);
YTrain = YTrain(perm);
XVal = XTrain(:,:,:,end - round(0.3*size(XTrain,4)):end);
YVal = YTrain(end - round(0.3*size(YTrain,1)):end);
XTrain = XTrain(:,:,:,1:round(0.7*size(XTrain,4)));
YTrain = YTrain(1:round(0.7*size(YTrain,1)));

layers = [
    imageInputLayer([28 28 1])
    
    convolution2dLayer(3,8,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,16,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,32,'Padding','same')
    batchNormalizationLayer
    reluLayer
    tensorVectorLayer('Flatten')
    fullyConnectedLayer(numel(categories(YTrain)))
%     FCLayer(1568,10,'fc_bf_fp',[])
    zeroBiasFCLayer(numel(categories(YTrain)),numel(categories(YTrain)),'zb_fp',[])
    softmaxLayer
    classificationLayer];

options = trainingOptions('sgdm', ...
    'InitialLearnRate',0.01, ...
    'MaxEpochs',20, ...
    'Shuffle','every-epoch', ...
    'ValidationData',{XVal,YVal}, ...
    'ValidationFrequency',30, ...
    'Verbose',false, ...
    'Plots','training-progress');
net = trainNetwork(XTrain, YTrain,layers,options);
YPred = classify(net,XVal);
accuracy = sum(YPred == YVal)/numel(YVal)

lgraph = layerGraph(net.Layers);
lgraph = lgraph.removeLayers('classoutput');
dlnet = dlnetwork(lgraph);

%%%%%% Visualize the decision voronoi
tYfp = extractdata(squeeze(predict(dlnet, dlarray(XVal(:,:,:,1:end),'SSCB'),...
    'Outputs','fc')))';    
tY2fp = extractdata(squeeze(predict(dlnet, dlarray(uX(:,:,:,1:end),'SSCB'),...
		'Outputs','fc')))';  
tYfp = tYfp./sqrt(sum(tYfp.^2,2));
tY2fp = tY2fp./sqrt(sum(tY2fp.^2,2));
tsnefp = tsne(gather([(dlnet.Layers(15).Weights);tYfp;tY2fp]),...
    'Distance','cosine','NumDimensions',2,'Algorithm','barneshut'); 
% tsnefp = tsne(gather([(dlnet.Layers(15).Weights);tYfp]),...
%     'Distance','cosine','NumDimensions',2,'Algorithm','barneshut'); 
set(0,'DefaultTextFontName','Times','DefaultTextFontSize',18,...
   'DefaultAxesFontName','Times','DefaultAxesFontSize',18,...
   'DefaultLineLineWidth',1,'DefaultLineMarkerSize',7.75)
figure
voronoi(double(tsnefp(1:size(dlnet.Layers(15).Weights,1),1)),...
    double(tsnefp(1:size(dlnet.Layers(15).Weights,1),2)))
hold on;
beginOftYFp = size(dlnet.Layers(15).Weights,1) + 1;
% plot(double(tsnefp(beginOftYFp:beginOftYFp+size(tYfp,1),1)),...
%     double(tsnefp(beginOftYFp:beginOftYFp+size(tYfp,1),2)),'g.')
gscatter(double(tsnefp(beginOftYFp:beginOftYFp+size(tYfp,1)-1,1)),...
    double(tsnefp(beginOftYFp:beginOftYFp+size(tYfp,1)-1,2)),...
    YVal);
%plot(double(tsnefp(29+size(tY2fp,1):end,1)),double(tsnefp(29+size(tY2fp,1):end,2)),'+')
scatter(double(tsnefp(1:beginOftYFp-1,1)),...
    double(tsnefp(1:beginOftYFp-1,2)),...
    'filled','DisplayName','Fingerprints','Marker','^');
scatter(double(tsnefp(end - size(tY2fp,1)+1:end,1)),...
    double(tsnefp(end - size(tY2fp,1)+1:end,2)),0.2,...
    'DisplayName','Abnormalities','Marker','.');
axis equal
% set(gca,'FontSize',12)
%%%%%% End of visualizing decision voronoi



