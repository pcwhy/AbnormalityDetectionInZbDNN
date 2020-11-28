clc;
%close all;
clear;
rng default;

% addpath('./matplotlib')  
addpath('./HypersphereLib/')  

% Please download the dataset from
% https://drive.google.com/uc?export=download&id=1N-eImoAA3QFPu3cBJd-1WIUH0cqw2RoT

load('adsb_records_qt.mat');

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
dateTimeLst = datetime(uint64(timeStampLst),'ConvertFrom','posixtime','TimeZone','America/New_York','TicksPerSecond',1e3,'Format','dd-MMM-yyyy HH:mm:ss.SSS');

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
% 
% % 
for i = 1:size(selectedBasebandData,1)
    mu = mean(selectedBasebandData(i,2:end));
    sigma = std(selectedBasebandData(i,2:end));
    
    % Use probabilistic filtering to restore the rational signal
    mask = selectedBasebandData(i,2:end) <= mu + 3*sigma;
    cleanInput = selectedBasebandData(i,2:end).*mask;
    selectedBasebandData(i,2:end) = cleanInput;
end
for i = 1:size(selectedRawCompData,1)
    muReal = mean(real(selectedRawCompData(i,1:end)));
    sigmaReal = std(real(selectedRawCompData(i,1:end)));
    muImag = mean(imag(selectedRawCompData(i,1:end)));
    sigmaImag = std(imag(selectedRawCompData(i,1:end)));
    
    % Use probabilistic filtering to restore the rational signal
    maskReal = abs(real(selectedRawCompData(i,1:end))) <= muReal + 3*sigmaReal;
    cleanInputReal = real(selectedRawCompData(i,1:end)).*maskReal;
    
    maskImag = abs(imag(selectedRawCompData(i,1:end))) <= muImag + 3*sigmaImag;
    cleanInputImag = imag(selectedRawCompData(i,1:end)).*maskImag;
    
    selectedRawCompData(i,1:end) = cleanInputReal+cleanInputImag.*1j;    
end


[X,cX,Y,cY] = makeDataTensor(selectedBasebandData,selectedRawCompData);
[uX,cuX,uY,cuY] = makeDataTensor(unknownBasebandData,unknownRawCompData);

lookUpTab = [unique([Y;cY]),[1:length(unique([Y;cY]))]'];
Y2 = Y;
for i = 1:size(Y)
    Y2(i) = lookUpTab(find(lookUpTab(:,1) == Y(i)),2);
end
cY2 = cY;
for i = 1:size(cY)
    cY2(i) = lookUpTab(find(lookUpTab(:,1) == cY(i)),2);
end
% uY2 = uY;
% for i = 1:size(uY)
%     uY2(i) = lookUpTab(find(lookUpTab(:,1) == uY(i)),2);
% end
% cuY2 = cuY;
% for i = 1:size(cuY)
%     cuY2(i) = lookUpTab(find(lookUpTab(:,1) == cuY(i)),2);
% end
Y = Y2;
cY = cY2;
% uY = uY2;
% cuY = cuY2;
inputSize = [size(X,1) size(X,2) size(X,3)];
numClasses = size(unique(selectedBasebandData(:,1)),1);

layers = [
    imageInputLayer(inputSize, 'Name', 'input', 'Mean', 0)
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
%     fullyConnectedLayer(numClasses, 'Name', 'Fingerprints') 
    zeroBiasFCLayer(numClasses,numClasses,'Fingerprints',[])
    yxSoftmax('softmax_1')
    classificationLayer('Name', 'classify_1')
    ];

lgraph = layerGraph(layers);
lgraph = connectLayers(lgraph, 'relu_1', 'add_1/in2');
% figure; plot(lgraph);

% options = trainingOptions('sgdm',...
%     'Plots', 'training-progress',...
%     'ExecutionEnvironment','auto',...
%     'ValidationData',{cX,categorical(cY)},...
%     'MaxEpochs', 60, ...
%     'MiniBatchSize',128,...
%     'L2Regularization',0.005);
% [net,info] = trainNetwork(X, categorical(Y), lgraph, options);
% % %iacc2 = find(~isnan(info.ValidationAccuracy) == 1)
% % %vacc2 = info.ValidationAccuracy(~isnan(info.ValidationAccuracy) == 1);
% YPred = classify(net, cX);
% accuracy = sum(categorical(cY) == YPred)/numel(cY)
% cm = confusionmat(categorical(cY),YPred);
% cm = cm./sum(cm,2);
% imagesc(cm);

XTrain = X;
YTrain = Y;
numEpochs = 3;
miniBatchSize = 128;
plots = "training-progress";
executionEnvironment = "auto";
if plots == "training-progress"
    figure(10);
    lineLossTrain = animatedline('Color','#0072BD','lineWidth',1.5);
    lineClassificationLoss = animatedline('Color','#EDB120','lineWidth',1.5);
      
    ylim([-inf inf])
    xlabel("Iteration")
    ylabel("Loss")
    legend('Loss','classificationLoss');
    grid on;
    
    figure(11);  
    lineCVAccuracy = animatedline('Color','#D95319','lineWidth',1.5);
    ylim([0 1.1])
    xlabel("Iteration")
    ylabel("Loss")    
    legend('CV Acc.','Avg. Kernel dist.');
    grid on;    
end
L2RegularizationFactor = 0.01;
initialLearnRate = 0.01;
decay = 0.01;
momentumSGD = 0.9;
velocities = [];
learnRates = [];
momentums = [];
gradientMasks = [];
numObservations = numel(YTrain);
numIterationsPerEpoch = floor(numObservations./miniBatchSize);
iteration = 0;
start = tic;
classes = categorical(YTrain);
% lgraph2 = layerGraph(net); % Also collect old weights
% % OR:
lgraph2 = lgraph; % No old weights
lgraph2 = lgraph2.removeLayers('classify_1');
dlnet = dlnetwork(lgraph2);

% Loop over epochs.
totalIters = 0;
abandonFlg = 0;

for epoch = 1:numEpochs
    if abandonFlg == 1
        break;
    end
    idx = randperm(numel(YTrain));
    XTrain = XTrain(:,:,:,idx);
    YTrain = YTrain(idx); 
    % Loop over mini-batches.
    for i = 1:numIterationsPerEpoch
        iteration = iteration + 1;
        totalIters = totalIters + 1;
        % Read mini-batch of data and convert the labels to dummy
        % variables.
        idx = (i-1)*miniBatchSize+1:i*miniBatchSize;
        Xb = XTrain(:,:,:,idx);
        Yb = zeros(numClasses, miniBatchSize, 'single');
        for c = 1:numClasses
            Yb(c,YTrain(idx)==(c)) = 1;
        end
        % Convert mini-batch of data to dlarray.
        dlX = dlarray(single(Xb),'SSCB');
        % If training on a GPU, then convert data to gpuArray.
        if (executionEnvironment == "auto" && canUseGPU) || executionEnvironment == "gpu"
            dlX = gpuArray(dlX);
        end
        % Evaluate the model gradients, state, and loss using dlfeval and the
        % modelGradients function and update the network state.
        [gradients,state,loss,classificationLoss] = dlfeval(@modelGradientsOnWeights,dlnet,dlX,Yb);
%         [gradients,state,loss] = dlfeval(@modelGradientsOnWeights,dlnet,dlX,Yb);        
        dlnet.State = state;
        % Determine learning rate for time-based decay learning rate schedule.
        learnRate = initialLearnRate/(1 + decay*iteration);
        % Update the network parameters using the SGDM optimizer.
        %[dlnet, velocity] = sgdmupdate(dlnet, gradients, velocity, learnRate, momentum);
        % Update the network parameters using the SGD optimizer.
        %dlnet = dlupdate(@sgdFunction,dlnet,gradients);
        if isempty(velocities)
            velocities = packScalar(gradients, 0);
            learnRates = packScalar(gradients, learnRate);
%             momentums = packScalar(gradients, momentumSGD);
            momentums = packScalar(gradients, 0);
            L2Foctors = packScalar(gradients, 0);            
            gradientMasks = packScalar(gradients, 1);   
%             % Let's lock some weights
%             for k = 1:2
%                 gradientMasks.Value{k}=dlarray(zeros(size(gradientMasks.Value{k})));
%             end
        end

%         [dlnet, velocities] = dlupdate(@sgdmFunctionL2, ...
%             dlnet, gradients, velocities, ...
%             learnRates, momentums, L2Foctors, gradientMasks);
        totalIterPk = packScalar(gradients, totalIters);
        [dlnet, velocities, momentums] = dlupdate(@adamFunction, ...
                    dlnet, gradients, velocities, ...
                    learnRates, momentums, L2Foctors, gradientMasks, ...
                    totalIterPk);        
%         [dlnet] = dlupdate(@sgdFunction, ...
%             dlnet, gradients);
        % Display the training progress.
        if plots == "training-progress"
            D = duration(0,0,toc(start),'Format','hh:mm:ss');
            XTest = cX;
            YTest = categorical(cY);
            if mod(iteration,5) == 0 
                accuracy = cvAccuracy(dlnet, XTest,YTest,miniBatchSize,executionEnvironment,0);
                addpoints(lineCVAccuracy,iteration, accuracy);
                [~,idx] = max(extractdata(dlnet.predict(dlarray(cX,'SSCB'))),[],1);
                cond = (idx(:) == cY(:));
                tYpositive = extractdata(squeeze(predict(dlnet, dlarray(cX(:,:,:,cond),'SSCB'),...
                    'Outputs','Fingerprints')))';

                if accuracy > 0.95
                   abandonFlg = 1;
                   break;
                end
            end
            addpoints(lineLossTrain,iteration,double(gather(extractdata(loss))))
            addpoints(lineClassificationLoss,iteration,double(gather(extractdata(classificationLoss))));
            title("Epoch: " + epoch + ", Elapsed: " + string(D))
            drawnow
        end
    end
end
accuracy = cvAccuracy(dlnet, cX, categorical(cY), miniBatchSize, executionEnvironment, 1)
tmpCx = cX;
tmpCy = cY;
cX = X;
cY = Y;
[~,idx] = max(extractdata(dlnet.predict(dlarray(cX,'SSCB'))),[],1);
cond = (idx(:) == cY(:));
tYfp = extractdata(squeeze(predict(dlnet, dlarray(cX(:,:,:,1:end),'SSCB'),...
    'Outputs','fc_bf_fp')))';    
tY2fp = extractdata(squeeze(predict(dlnet, dlarray(uX(:,:,:,1:end),'SSCB'),...
		'Outputs','fc_bf_fp')))';  
tYfp = tYfp./sqrt(sum(tYfp.^2,2));
tY2fp = tY2fp./sqrt(sum(tY2fp.^2,2));
classCenters = [];
classMahalModels = {};
classMahalDistrib = {};
for i = 1:size(unique(Y),1)
    condOneClass = logical((cY(1:end) == i).*cond);
    oneClassFp = tYfp(condOneClass,:);
    avgOneClassFp = mean(oneClassFp);
    avgOneClassFp = avgOneClassFp./sqrt(sum(avgOneClassFp.^2,2));
    [C,m] = covmatrix(oneClassFp);
    classMahalModels{end+1} = {C,m};
    mahalDists = gather(mahalanobis(oneClassFp,classMahalModels{end}{1},classMahalModels{end}{2}));
    [~,den,xm,aClusterCDF]=kde(mahalDists,512,min(mahalDists),max(mahalDists));     
    classMahalDistrib{end+1} = {xm,den,aClusterCDF};

%     centerDistDistrib = sum(oneClassFp.* avgOneClassFp,2);
%     filterCond = centerDistDistrib >= 0.3;
%     oneClassFp = oneClassFp(filterCond,:);
%     avgOneClassFp = mean(oneClassFp);
%     avgOneClassFp = avgOneClassFp./sqrt(sum(avgOneClassFp.^2,2));    
    classCenters(end+1,:) = gather(avgOneClassFp);
end
cX = tmpCx;
cY = tmpCy;

%%%%%% Visualize the decision voronoi
tYfp = extractdata(squeeze(predict(dlnet, dlarray(cX(:,:,:,1:end),'SSCB'),...
    'Outputs','fc_bf_fp')))';    
tY2fp = extractdata(squeeze(predict(dlnet, dlarray(uX(:,:,:,1:end),'SSCB'),...
		'Outputs','fc_bf_fp')))';  
tYfp = tYfp./sqrt(sum(tYfp.^2,2));
tY2fp = tY2fp./sqrt(sum(tY2fp.^2,2));
tsnefp = tsne(gather([(dlnet.Layers(11).Weights);tYfp;tY2fp;classCenters]),...
    'Distance','cosine','NumDimensions',2,'Algorithm','barneshut'); 
%tsnefp = tsne([(dlnet.Layers(11).Weights);tYfp],'Distance','cosine'); 
figure
pe = voronoi(double(tsnefp(1:size(dlnet.Layers(11).Weights,1),1)),...
    double(tsnefp(1:size(dlnet.Layers(11).Weights,1),2)))
hold on;
beginOftYFp = size(dlnet.Layers(11).Weights,1) + 1;
% plot(double(tsnefp(beginOftYFp:beginOftYFp+size(tYfp,1),1)),...
%     double(tsnefp(beginOftYFp:beginOftYFp+size(tYfp,1),2)),'g.')
gscatter(double(tsnefp(beginOftYFp:beginOftYFp+size(tYfp,1)-1,1)),...
    double(tsnefp(beginOftYFp:beginOftYFp+size(tYfp,1)-1,2)),...
    cY);
%plot(double(tsnefp(29+size(tY2fp,1):end,1)),double(tsnefp(29+size(tY2fp,1):end,2)),'+')
plot(double(tsnefp(1:beginOftYFp-1,1)),...
    double(tsnefp(1:beginOftYFp-1,2)),...
    'ro','lineWidth',2)
plot(double(tsnefp(29+size(tYfp,1):round(end/2.5),1)),...
    double(tsnefp(29+size(tYfp,1):round(end/2.5),2)),'x')
plot(double(tsnefp(end-size(classCenters,1) + 1:end,1)),...
    double(tsnefp(end-size(classCenters,1) + 1:end,2)),'rx','lineWidth',2)
legend('Fingerprint','Decision boundaries','Mapped inputs')
axis equal
set(gca,'FontSize',12)

figure
pe = voronoi(double(tsnefp(1:size(dlnet.Layers(11).Weights,1),1)),...
    double(tsnefp(1:size(dlnet.Layers(11).Weights,1),2)))
hold on;
scatter(double(tsnefp(beginOftYFp:beginOftYFp+size(tYfp,1)-1,1)),...
  double(tsnefp(beginOftYFp:beginOftYFp+size(tYfp,1)-1,2)),3,[0.3010 0.7450 0.9330],'filled')
scatter(double(tsnefp(29+size(tYfp,1):round(end/2.5),1)),...
    double(tsnefp(29+size(tYfp,1):round(end/2.5),2)),3,'m','filled')
legend('Class fingerprint','Class boundaries','Normal data','Abnormalities')

%%%%%% End of visualizing decision voronoi


tmpCx = cX;
tmpCy = cY;


[~,idx] = max(extractdata(dlnet.predict(dlarray(cX(:,:,:,1:end),'SSCB'))),[],1);
cond = (idx(:) == cY(1:end));
idx2 = idx(cond);
classFN = [];
for i = 1:numel(unique(cY(:)))
    gtCond = cY(1:end/2)==i;
    actualDecisions = idx(gtCond);
    FP = sum(actualDecisions ~= i,'all')./sum(gtCond,'all');
    FN = 1-sum(actualDecisions == i,'all')./sum(gtCond,'all');
    classFN(end+1,:) = [i,gather(FN)];
end

newzb = zeroBiasFCLayer(numClasses,numClasses,'Fingerprints',classCenters);
lgraph = layerGraph(dlnet);
lgraph = lgraph.replaceLayer('Fingerprints',newzb);
dlnet = dlnetwork(lgraph);
% accuracy = cvAccuracy(dlnet, cX, categorical(cY), miniBatchSize, executionEnvironment, 0);

tY2positive = extractdata(squeeze(predict(dlnet, dlarray(uX(:,:,:,1:2500),'SSCB'),...
    'Outputs','Fingerprints')))';
[magtY2positive,pos2] = max(tY2positive,[],2);         
magtY2positive = magtY2positive./(dlnet.Layers(11).normMag)-1;


tYpositive = extractdata(squeeze(predict(dlnet, dlarray(cX(:,:,:,1:end),'SSCB'),...
    'Outputs','Fingerprints')))';
[magtYpositive,pos] = max(tYpositive,[],2);  
magtYpositive = magtYpositive./(dlnet.Layers(11).normMag)-1;


tYfp = extractdata(squeeze(predict(dlnet, dlarray(cX(:,:,:,1:end),'SSCB'),...
    'Outputs','fc_bf_fp')))';    
tY2fp = extractdata(squeeze(predict(dlnet, dlarray(uX(:,:,:,1:2500),'SSCB'),...
        'Outputs','fc_bf_fp')))';  
tYfp = tYfp./sqrt(sum(tYfp.^2,2));
tY2fp = tY2fp./sqrt(sum(tY2fp.^2,2));
roc = [];
metrics = [];
for cutOffThreshold = 0:0.02:1
    abnormalityCount = 0;
    regularCount = 0;
    tp = 0;
    tn = 0;
    fp = 0;
    fn = 0;
    for i = 1:size(tY2fp,1)
        mhDist = mahalanobis(tY2fp(i,:),classMahalModels{pos2(i)}{1},classMahalModels{pos2(i)}{2});
        xm = classMahalDistrib{pos2(i)}{1};
        cutOffDist = xm(...
            max(1,sum(classMahalDistrib{pos2(i)}{3}<=cutOffThreshold))...
            );

        if mhDist > cutOffDist
            abnormalityCount = abnormalityCount + 1;
            tp = tp+1;
        else
            regularCount = regularCount + 1;
            fn = fn+1;
        end
    end
    for i = 1:size(tYfp,1)
        mhDist = mahalanobis(tYfp(i,:),classMahalModels{pos(i)}{1},classMahalModels{pos(i)}{2});
        xm = classMahalDistrib{pos(i)}{1};
        cutOffDist = xm(...
            max(1,sum(classMahalDistrib{pos(i)}{3}<=cutOffThreshold))...
            );    
    %     cutOffDist = xm(...
    %         sum(classMahalDistrib{pos(i)}{3}<=1-classFN(pos(i),2))...
    %         );
        if mhDist <= cutOffDist
            regularCount = regularCount + 1;
            tn = tn + 1;
        else
            abnormalityCount = abnormalityCount + 1;
            fp = fp + 1;
        end
    end
    tpr = tp./(tp + fn);
    tnr = tn./(tn + fp);
    fpr = fp./(tn + fp);
    fnr = fn./(tp + fn);
    anAcc = (tp+tn)./(tp+tn+fp+fn);
    roc(end+1,:) = [fpr,tpr,anAcc];
    metrics(end+1,:) = [tpr,tnr,fpr,fnr];
end
figure;
hold on;
plot(roc(:,1),roc(:,2),'*','LineWidth',1.0);
plot(roc(:,1),roc(:,1),'--','LineWidth',1.0);
figure;
hold on;
plot([0:0.02:1],metrics(:,1),'LineWidth',1.0);
plot([0:0.02:1],metrics(:,2),'LineWidth',1.0);
plot([0:0.02:1],metrics(:,3),'LineWidth',1.0);
plot([0:0.02:1],metrics(:,4),'LineWidth',1.0);
legend('tpr','tnr','fpr','fnr');

cX = tmpCx;
cY = tmpCy;

[~,idx] = max(extractdata(dlnet.predict(dlarray(cX,'SSCB'))),[],1);
cond = (idx(:) == cY(:));
tYpositive = extractdata(squeeze(predict(dlnet, dlarray(cX(:,:,:,cond),'SSCB'),...
    'Outputs','Fingerprints')))';
hyperSphereCoverage = occupiedRatio(dlnet.Layers(11).Weights,tYpositive,dlnet.Layers(11).normMag)




function ratio = occupiedRatio(zbFingerprints,tYpositive,normMag)
    uFingerprints = zbFingerprints./sqrt(sum(zbFingerprints.^2,2));
    cosines = (tYpositive-normMag)./normMag;
    [maxCosines,posMax] = max(cosines,[],2);  
    [minCosines,posMin] = min(cosines,[],2);    
    minAcceptableClassBoundaries = [];
    for i = 1:size(uFingerprints,1)
        classMinCosines = cosines(posMax==i,i);
        minAcceptableClassBoundaries(end+1,:) = min(classMinCosines,[],1);
    end
    featureDims = size(uFingerprints,2);
    N = 50000;
    X = 2*pi*rand(featureDims-1,N);
    r = 1;
    mcSamples = HyperSphere([X],r);
    classificationOutput = (uFingerprints*mcSamples)';
    capturedMcSamples = 0;
    for i = 1:size(classificationOutput,1)
        singleMatch = classificationOutput(i,:) >= (minAcceptableClassBoundaries)';
        if sum(singleMatch,'all') > 0
            capturedMcSamples = capturedMcSamples + 1;
        end
    end
    ratio = capturedMcSamples./N;
end

function accuracy = cvAccuracy(dlnet, XTest, YTest, miniBatchSize, executionEnvironment, confusionChartFlg)
    dlXTest = dlarray(XTest,'SSCB');
    if (executionEnvironment == "auto" && canUseGPU) || executionEnvironment == "gpu"
        dlXTest = gpuArray(dlXTest);
    end
    dlYPred = modelPredictions(dlnet,dlXTest,miniBatchSize);
    [~,idx] = max(extractdata(dlYPred),[],1);
    YPred = categorical(gather(idx));
    accuracy = mean(YPred(:) == YTest(:));
    if confusionChartFlg == 1
        figure
        confusionchart(YPred(:),YTest(:));
    end
end

function dlYPred = modelPredictions(dlnet,dlX,miniBatchSize)
    numObservations = size(dlX,4);
    numIterations = ceil(numObservations / miniBatchSize);
    numClasses = size(dlnet.Layers(end-1).Weights,1);
    dlYPred = zeros(numClasses,numObservations,'like',dlX);
    for i = 1:numIterations
        idx = (i-1)*miniBatchSize+1:min(i*miniBatchSize,numObservations);
        dlYPred(:,idx) = predict(dlnet,dlX(:,:,:,idx));
    end
end


function [gradients,state,loss,classificationLoss] = modelGradientsOnWeights(dlnet,dlX,Y)
%   %This is only used with softmax of matlab which only applies softmax
%   on 'C' and 'B' channels.
    [rawPredictions,state] = forward(dlnet,dlX,'Outputs', 'Fingerprints');
    dlYPred = softmax(dlarray(squeeze(rawPredictions),'CB'));
%     [dlYPred,state] = forward(dlnet,dlX);
    penalty = 0;
    scalarL2Factor = 0;
    if scalarL2Factor ~= 0
        paramLst = dlnet.Learnables.Value;
        for i = 1:size(paramLst,1)
            penalty = penalty + sum((paramLst{i}(:)).^2);
        end
    end
    
    classificationLoss = crossentropy(squeeze(dlYPred),Y) + scalarL2Factor*penalty;

    loss = classificationLoss;
%     loss = classificationLoss + 0.2*(max(max(rawPredictions))-min(max(rawPredictions)));
    gradients = dlgradient(loss,dlnet.Learnables);
    %gradients = dlgradient(loss,dlnet.Learnables(4,:));
end

function [params,velocityUpdates,momentumUpdate] = adamFunction(params, rawParamGradients,...
    velocities, learnRates, momentums, L2Foctors, gradientMasks, iters)
    % https://arxiv.org/pdf/2010.07468.pdf %%AdaBelief
    % https://arxiv.org/pdf/1711.05101.pdf  %%DeCoupled Weight Decay 
    b1 = 0.5; 
    b2 = 0.999;
    e = 1e-8;
    curIter = iters(:);
    curIter = curIter(1);
    
    gt = rawParamGradients;
    mt = (momentums.*b1 + ((1-b1)).*gt);
    vt = (velocities.*b2 + ((1-b2)).*((gt-mt).^2));

     momentumUpdate = mt;
     velocityUpdates = vt;
    h_mt = mt./(1-b1.^curIter);
    h_vt = (vt+e)./(1-b2.^curIter);
    params = params - 0.0001.*(mt./(sqrt(vt)+e)).*gradientMasks...
        -L2Foctors.*params.*gradientMasks; %This works better for zero-bias dense layer
%     params = params - 0.001.*(h_mt./(sqrt(h_vt)+e)).*gradientMasks...
%         -L2Foctors.*params.*gradientMasks;

end

function param = sgdFunction(param,paramGradient)
    learnRate = 0.01;
    param = param - learnRate.*paramGradient;
end

function [params, velocityUpdates] = sgdmFunction(params, paramGradients,...
    velocities, learnRates, momentums)
% https://towardsdatascience.com/stochastic-gradient-descent-momentum-explanation-8548a1cd264e
%     velocityUpdates = momentums.*velocities+learnRates.*paramGradients;
    velocityUpdates = momentums.*velocities+0.001.*paramGradients;
    params = params - velocityUpdates;
end

function [params, velocityUpdates] = sgdmFunctionL2(params, rawParamGradients,...
    velocities, learnRates, momentums, L2Foctors, gradientMasks)
% https://towardsdatascience.com/stochastic-gradient-descent-momentum-explanation-8548a1cd264e
% https://towardsdatascience.com/intuitions-on-l1-and-l2-regularisation-235f2db4c261
    paramGradients = rawParamGradients + 2*L2Foctors.*params;
    velocityUpdates = momentums.*velocities+learnRates.*paramGradients;
    params = params - (velocityUpdates).*gradientMasks;
end

function tabVars = packScalar(target, scalar)
% The matlabs' silly design results in such a strange function
    tabVars = target;
    for row = 1:size(tabVars(:,3),1)
        tabVars{row,3} = {...
            dlarray(...
            ones(size(tabVars.Value{row})).*scalar...%ones(size(tabVars(row,3).Value{1,1})).*scalar...
            )...
            };
    end
end