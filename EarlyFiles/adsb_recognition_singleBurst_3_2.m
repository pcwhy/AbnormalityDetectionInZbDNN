clc;
close all;
clear;
rng default;

addpath('./matplotlib')  


load('adsb_records_qt.mat');
%load('adsb_bladerf2_10M_qt');
payloadMatrix = reshape(payloadMatrix', ...
    length(payloadMatrix)/length(msgIdLst), length(msgIdLst))';
rawIMatrix = reshape(rawIMatrix', ...
    length(rawIMatrix)/length(msgIdLst), length(msgIdLst))';
rawQMatrix = reshape(rawQMatrix', ...
    length(rawQMatrix)/length(msgIdLst), length(msgIdLst))';
rawCompMatrix = rawIMatrix + rawQMatrix.*1j;
if size(rawCompMatrix,2) < 1024
    appendingBits = (ceil(sqrt(size(rawCompMatrix,2))))^2 - size(rawCompMatrix,2);
    rawCompMatrix = [rawCompMatrix, zeros(size(rawCompMatrix,1), appendingBits)];
else
   rawCompMatrix = rawCompMatrix(:,1:1024); 
end
uIcao = unique(icaoLst);
c = countmember(uIcao,icaoLst);
icaoOccurTb = [uIcao,c];
icaoOccurTb = sortrows(icaoOccurTb,2,'descend');
cond1 = icaoOccurTb(:,2)>=300;
cond2 = icaoOccurTb(:,2)<=5000;
cond = logical(cond1.*cond2);
selectedPlanes = icaoOccurTb(cond,:);
unknowPlanes = icaoOccurTb(~cond,:);
allTrainData = [icaoLst, abs(rawCompMatrix)];
selectedBasebandData = [];
selectedRawCompData = [];

unknownBasebandData = [];
unknownRawCompData = [];

minTrainChance = 100;
maxTrainChance = 500;

for i = 1:size(selectedPlanes,1)
    selection = allTrainData(:,1)==selectedPlanes(i,1);
    localBaseband = allTrainData(selection,:);
    localComplex = rawCompMatrix(selection,:);
    localAngles = (angle(localComplex(:,:)));

%     figure
%     for k = 1:size(localAngles,1)
%         plot(localAngles(k,:),'.');
%         title(strcat(num2str(k), ' / ', num2str(size(localAngles,1))));
%         pause(30/1000);
%     end    
    
    if size(localBaseband,1) < minTrainChance
        continue;
    elseif size(localBaseband,1) >= maxTrainChance
        rndSeq = randperm(size(localBaseband,1));
        rndSeq = rndSeq(1:maxTrainChance);
        localBaseband = localBaseband(rndSeq,:);
        localComplex = localComplex(rndSeq,:);
    else
        %Nothing to do
    end
    selectedBasebandData = [selectedBasebandData; localBaseband];
    selectedRawCompData = [selectedRawCompData; localComplex];    
end

for i = 1:size(unknowPlanes,1)
    selection = allTrainData(:,1)==unknowPlanes(i,1);
    localBaseband = allTrainData(selection,:);
    localComplex = rawCompMatrix(selection,:);
    localAngles = (angle(localComplex(:,:)));
    unknownBasebandData = [unknownBasebandData; localBaseband];
    unknownRawCompData = [unknownRawCompData; localComplex];    
end

randSeries = randperm(size(selectedBasebandData,1));
selectedBasebandData = selectedBasebandData(randSeries,:);
selectedRawCompData = selectedRawCompData(randSeries,:);

randSeries = randperm(size(unknownBasebandData,1));
unknownBasebandData = unknownBasebandData(randSeries,:);
unknownRawCompData = unknownRawCompData(randSeries,:);

[X,cX,Y,cY] = makeDataTensor(selectedBasebandData,selectedRawCompData);
[uX,cuX,uY,cuY] = makeDataTensor(unknownBasebandData,unknownRawCompData);

inputSize = [size(X,1) size(X,2) size(X,3)];
numClasses = size(unique(selectedBasebandData(:,1)),1);

layers = [
    imageInputLayer(inputSize, 'Name', 'input')
    convolution2dLayer(5,10, 'Name', 'conv2d_1')
    batchNormalizationLayer('Name', 'batchNorm_1')
    reluLayer('Name', 'relu_1')
    convolution2dLayer(3, 10, 'Padding', 1, 'Name', 'conv2d_2')
    reluLayer('Name', 'relu_2')    
    convolution2dLayer(3, 10, 'Padding', 1, 'Name', 'conv2d_3')
    reluLayer('Name', 'relu_3')    
    %additionLayer(2,'Name', 'add_1')
    depthConcatenationLayer(2,'Name','add_1')    

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
    'ValidationData',{cX,categorical(cY)},...
    'MaxEpochs', 60, ...
    'MiniBatchSize',128,...
    'L2Regularization',0.005);
[net,info] = trainNetwork(X, categorical(Y), lgraph, options);

%iacc2 = find(~isnan(info.ValidationAccuracy) == 1)
%vacc2 = info.ValidationAccuracy(~isnan(info.ValidationAccuracy) == 1);

YPred = classify(net, cX);
accuracy = sum(categorical(cY) == YPred)/numel(cY)
cm = confusionmat(categorical(cY),YPred);
cm = cm./sum(cm,2);
imagesc(cm);

% %plotconfusion(categorical(cY),YPred);
% tY = activations(net, cX(:,:,:,1:500), 'softmax_1');
% tY2 = squeeze(tY)';
% tY3 = tsne(tY2,'Algorithm','barneshut','NumPCAComponents',30,'NumDimensions',3);
% %gscatter(tY3(:,1),tY3(:,2),categorical(cY(1:500)))
% scatter3(tY3(:,1),tY3(:,2),tY3(:,3),15,categorical(cY(1:500)),'filled');
% %scatter3(tY3(:,1),tY3(:,2),tY3(:,3),5,categorical(cY(1:500)),'filled');

% vx = 1:length(info.ValidationAccuracy);
% vm = logical(~isnan(info.ValidationAccuracy));
% plot(info.TrainingAccuracy);
% hold;
% plot(vx(vm),info.ValidationAccuracy(vm),'LineWidth',1.5);
% legend('training','validation');
% xlim([1 4000]);
% legend('training','validation');

% 405   330   528   279

% cv1 = net.Layers(2).Weights;
% figure
% for i = 1:size(cv1,4)
%     subplot(2,5,i);
%     cv11 = cv1(:,:,:,i);
%     cv11 = (cv11 ./ max(max(max(cv11))));
%     imagesc(cv11);
% end
% 
% tY = squeeze(activations(net, cX(:,:,:,1:2500), 'fc_bf_fp'))';
% tY2 = squeeze(activations(net, uX(:,:,:,1:2500), 'fc_bf_fp'))';
cond = classify(net, cX) == categorical(cY);
tY = squeeze(activations(net, cX(:,:,:,cond), 'Fingerprints'))';
tY2 = squeeze(activations(net, uX(:,:,:,1:2500), 'Fingerprints'))';
magtY = max(tY,[],2)./net.Layers(11).normMag;
magtY2 = max(tY2,[],2)./net.Layers(11).normMag;
% magtY = max(tY,[],2);
% magtY2 = max(tY2,[],2);

figure
subplot(2,1,1);
hold on;
histogram(magtY,'NumBins',100,'Normalization','probability','BinLimits',[1,2]);
histogram(magtY2,'NumBins',100,'Normalization','probability','BinLimits',[1,2]);
xlim([1,2])
subplot(2,1,2);
hold on;
histogram(magtY,'NumBins',100,'Normalization','cdf','BinLimits',[1,2]);
histogram(magtY2,'NumBins',100,'Normalization','cdf','BinLimits',[1,2]);
xlim([1,2])

figure
[cdfKnown,esKnown] = histcounts(magtY,'NumBins',100,'Normalization','cdf','BinLimits',[1,2]);
%es = (es(1:end-1)+es(2:end))./2;
esKnown = (esKnown(1:end-1)+esKnown(2:end))./2;
[cdfUnknown,esUnknown] = histcounts(magtY2,'NumBins',100,'Normalization','cdf','BinLimits',[1,2]);
esUnknown = (esUnknown(1:end-1)+esUnknown(2:end))./2;
hold on;
plot(esKnown,cdfKnown,'LineWidth',1.5);
plot(esUnknown,cdfUnknown,'LineWidth',1.5);
plot(esUnknown,((cdfUnknown)-(cdfKnown)),'LineWidth',1.5);
xlabel('Decision threshold')
ylabel('cdf');
legend('Known','Unknown','Decision margin');
set(gca,'FontSize',12)
%set(gcf,'position',[405   330   528   279]);
set(gcf,'position',[438   382   441   181]);
xlim([1,2])

figure
hold on;
histogram(magtY,'NumBins',100,'Normalization','probability','BinLimits',[1,2]);
histogram(magtY2,'NumBins',100,'Normalization','probability','BinLimits',[1,2]);
xlim([1,2])
xlabel('Similarities')
ylabel('Probability');
legend('Known','Unknown');
set(gca,'FontSize',12)
%set(gcf,'position',[405   330   528   279]);
set(gcf,'position',[438   382   441   181]);
xlim([1,2])
%Let's regard 'known' as positive and 'unknown' as negative
%load('adsb_records_qt.mat');
%TPR = 0.8107
%FPR = 0.1893
%TNR = 0.7893
%FNR = 0.2107

figure
ttY = squeeze(activations(net, X(:,:,:,1:end), 'Fingerprints'))';
tY = squeeze(activations(net, cX(:,:,:,1:2500), 'Fingerprints'))';
tY2 = squeeze(activations(net, uX(:,:,:,1:2500), 'Fingerprints'))';
svm = fitcsvm(ttY,ones(size(ttY,1),1), ...
    "KernelFunction","rbf",'OutlierFraction',0.01);
svm = svm.compact;
[~,knownscore]=svm.predict(tY);
[~,unknownscore]=svm.predict(tY2);
histogram(knownscore,'NumBins',50,'BinLimits',[-2,5],'Normalization','probability');
hold on;
histogram(unknownscore,'NumBins',50,'BinLimits',[-2,5],'Normalization','probability');
legend('Known','Unknown');
%title('One-class SVM (Using real samples)');
xlabel('Score')
ylabel('Probability');
legend('Known','Unknown');
set(gca,'FontSize',12)
%set(gcf,'position',[405   330   528   279]);
set(gcf,'position',[438   382   441   181]);

figure
[cdfKnown,esKnown] = histcounts(knownscore,'NumBins',100,'Normalization','cdf','BinLimits',[-2,5]);
%es = (es(1:end-1)+es(2:end))./2;
esKnown = (esKnown(1:end-1)+esKnown(2:end))./2;
[cdfUnknown,esUnknown] = histcounts(unknownscore,'NumBins',100,'Normalization','cdf','BinLimits',[-2,5]);
esUnknown = (esUnknown(1:end-1)+esUnknown(2:end))./2;
hold on;
plot(esKnown,cdfKnown,'LineWidth',1.5);
plot(esUnknown,cdfUnknown,'LineWidth',1.5);
plot(esUnknown,((cdfUnknown)-(cdfKnown)),'LineWidth',1.5);
xlabel('Decision threshold')
ylabel('cdf');
legend('Known','Unknown','Decision margin');
set(gca,'FontSize',12)
%set(gcf,'position',[405   330   528   279]);
set(gcf,'position',[438   382   441   181]);
%xlim([-1,1])

% 
% imagesc(weightCorr);
% fingerprintMatrices = {};
% fpCorrMatrices = {};
% corrSNR = [];
% for coeffL2Reg = 0:0.001:0.5
%     options = trainingOptions('sgdm',...
%         'Plots', 'training-progress',...
%         'ExecutionEnvironment','auto',...
%         'ValidationData',{cX,categorical(cY)},...
%         'MaxEpochs', 20, ...
%         'MiniBatchSize',128,...
%         'L2Regularization',coeffL2Reg);
%     [net,info] = trainNetwork(X, categorical(Y), lgraph, options);
%     finalMask = net.Layers(11).Weights;
%     fingerprintMatrices{end+1} = finalMask;
%     weightCorr=[];
%     for i = 1:size(finalMask,1)
%         weightCorr(end+1,:) = sum(finalMask(i,:).*finalMask,2)';
%     end
%     fpCorrMatrices{end+1} = weightCorr;
%     corrSNR(end+1) = mean(max(abs(weightCorr),[],2)./sum(abs(weightCorr),2));
%     
% end

