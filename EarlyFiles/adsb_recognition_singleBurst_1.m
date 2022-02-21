clc;
close all;
clear;
rng default;

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
cond1 = icaoOccurTb(:,2)>=100;
cond2 = icaoOccurTb(:,2)<=5000;
cond = logical(cond1.*cond2);
selectedPlanes = icaoOccurTb(cond,:);
unknowPlanes = icaoOccurTb(~cond,:);
allTrainData = [icaoLst, abs(rawCompMatrix)];
selectedBasebandData = [];
selectedRawCompData = [];

minTrainChance = 50;
maxTrainChance = 200;

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

    
%     for k = 1:25
%         subplot(5,5,k);
%         plot(localAngles(k,:),'.')
%     end
    
    % Might be useful.
    %instAng = unwrap(atan2(imag(rawCompMatrix(selection,:)),real(rawCompMatrix(selection,:))));
    %plot(ang);
end

% for i = 1:size(unknowPlanes,1)
%     selection = allTrainData(:,1)==unknowPlanes(i,1);
%     localBaseband = allTrainData(selection,:);
%     localComplex = rawCompMatrix(selection,:);
%     localAngles = (angle(localComplex(:,:)));
%     
%     if size(localBaseband,1) < minTrainChance
%         continue;
%     elseif size(localBaseband,1) >= maxTrainChance
%         rndSeq = randperm(size(localBaseband,1));
%         rndSeq = rndSeq(1:maxTrainChance);
%         localBaseband = localBaseband(rndSeq,:);
%         localComplex = localComplex(rndSeq,:);
%     else
%         %Nothing to do
%     end
%     
%     selectedBasebandData = [selectedBasebandData; localBaseband];
%     selectedRawCompData = [selectedRawCompData; localComplex];    
% 
% end



randSeries = randperm(size(selectedBasebandData,1));
selectedBasebandData = selectedBasebandData(randSeries,:);
selectedRawCompData = selectedRawCompData(randSeries,:);

selectedFFTVector = zeros(size(selectedBasebandData));
selectedFFTVector(:,1) = selectedBasebandData(:,1);
for i = 1:size(selectedBasebandData,1)
%   selectedFFTVector(i,2:end) = fftshift(fft(selectedBasebandData(i,2:end)));
    selectedFFTVector(i,2:end) = fftshift(fft(selectedRawCompData(i,:),1024));
end
cvFFT = selectedFFTVector(ceil(0.7*size(selectedFFTVector,1)):size(selectedFFTVector,1),:);
selectedFFTVector = selectedFFTVector(1:ceil(0.7*size(selectedFFTVector,1))-1,:);
selectedFFTmag = abs(selectedFFTVector);
selectedFFTang = angle(selectedFFTVector);
%selectedFFTang = unwrap(atan2(real(selectedFFTVector),imag(selectedFFTVector)));

cvFFTmag = abs(cvFFT);
cvFFTang = angle(cvFFT);
%cvFFTang = unwrap(atan2(real(cvFFT),imag(cvFFT)));

selectedNoiseVector = zeros(size(selectedBasebandData));
selectedNoiseVector(:,1) = selectedBasebandData(:,1);
for i = 1:size(selectedBasebandData,1)
    selectedNoiseVector(i,2:size(selectedBasebandData,2))...
        = extractNoise(selectedBasebandData(i,2:size(selectedBasebandData,2)));
%     plot(selectedNoiseVector(i,2:size(selectedBasebandData,2)))
end
cvNoise = selectedNoiseVector(ceil(0.7*size(selectedNoiseVector,1)):size(selectedNoiseVector,1),:);
selectedNoiseVector = selectedNoiseVector(1:ceil(0.7*size(selectedNoiseVector,1))-1,:);

featureDims = 3;

trainDataTensor = zeros(size(selectedNoiseVector,1)*featureDims,...
    size(selectedNoiseVector,2)-1);

for i = 1:size(selectedNoiseVector,1)
    cursor = i*featureDims-(featureDims-1);
    trainDataTensor(cursor,:) = selectedNoiseVector(i,2:end);
     trainDataTensor(cursor+1,:) = real(selectedFFTVector(i,2:end));
     trainDataTensor(cursor+2,:) = imag(selectedFFTVector(i,2:end));    
end

cvDataTensor = zeros(size(cvNoise,1)*featureDims, size(cvNoise,2)-1);
for i = 1:size(cvNoise,1)
    cursor = i*featureDims-(featureDims-1);    
    cvDataTensor(cursor,:) = cvNoise(i,2:end);    
    cvDataTensor(cursor+1,:) = real(cvFFT(i,2:end));
    cvDataTensor(cursor+2,:) = imag(cvFFT(i,2:end));
end

Y = selectedNoiseVector(:,1);
cY = cvNoise(:,1);

X2 = reshape(trainDataTensor',[sqrt(1024), sqrt(1024),...
    featureDims, size(selectedNoiseVector,1)]);
cX2 = reshape(cvDataTensor',[sqrt(1024), sqrt(1024),...
    featureDims, size(cY,1)]);
% One way to restore the origin signal is:
% sig = X2(:,:,:,1);
% sig2 = reshape(sig,[1,1024]);

inputSize = [size(X2,1) size(X2,2) size(X2,3)];
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

    fullyConnectedLayer(numClasses, 'Name', 'fc_1')
    %fullyConnectedLayer(numClasses, 'Name', 'fc_2')
    dropoutLayer(0.2,'Name','dropOut_1')
    softmaxLayer('Name', 'softmax_1')
    classificationLayer('Name', 'classify_1')];

lgraph = layerGraph(layers);

lgraph = connectLayers(lgraph, 'relu_1', 'add_1/in2');
%lgraph = connectLayers(lgraph, 'relu_2', 'add_1/in3');

plot(lgraph);

options = trainingOptions('sgdm',...
    'Plots', 'training-progress',...
    'ExecutionEnvironment','gpu',...
    'ValidationData',{cX2,categorical(cY)},...
    'MaxEpochs', 30);
[net,info] = trainNetwork(X2, categorical(Y), lgraph, options);

%iacc2 = find(~isnan(info.ValidationAccuracy) == 1)
%vacc2 = info.ValidationAccuracy(~isnan(info.ValidationAccuracy) == 1);

YPred = classify(net, cX2);
accuracy = sum(categorical(cY) == YPred)/numel(cY)
cm = confusionmat(categorical(cY),YPred);
cm = cm./sum(cm,2);
imagesc(cm);
%plotconfusion(categorical(cY),YPred);
tY = activations(net, cX2(:,:,:,1:500), 'softmax_1');
tY2 = squeeze(tY)';
tY3 = tsne(tY2,'Algorithm','barneshut','NumPCAComponents',30,'NumDimensions',3);
%gscatter(tY3(:,1),tY3(:,2),categorical(cY(1:500)))
scatter3(tY3(:,1),tY3(:,2),tY3(:,3),15,categorical(cY(1:500)),'filled');
%scatter3(tY3(:,1),tY3(:,2),tY3(:,3),5,categorical(cY(1:500)),'filled');

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

