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
% XTrain = reshape(raw',[32,32,2,count]);

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

load('droneRFFC_60.mat')
% dlnet = droneRFCNN;
dlnet = droneRFFC_60;

%%%%%% Visualize the decision voronoi
tYfp = extractdata(squeeze(predict(dlnet, dlarray(XVal','CB'),...
    'Outputs','dropout')))';    
% tY2fp = extractdata(squeeze(predict(dlnet, dlarray(uX(:,:,:,1:end),'SSCB'),...
% 		'Outputs','fc')))';  
tYfp = tYfp./sqrt(sum(tYfp.^2,2));
% tY2fp = tY2fp./sqrt(sum(tY2fp.^2,2));
% tsnefp = tsne(gather([(dlnet.Layers(15).Weights);tYfp;tY2fp]),...
%     'Distance','cosine','NumDimensions',2,'Algorithm','barneshut'); 
tsnefp = tsne(gather([(dlnet.Layers(14).Weights);tYfp]),...
    'Distance','cosine','NumDimensions',2,'Algorithm','barneshut'); 
set(0,'DefaultTextFontName','Times','DefaultTextFontSize',18,...
   'DefaultAxesFontName','Times','DefaultAxesFontSize',18,...
   'DefaultLineLineWidth',1,'DefaultLineMarkerSize',7.75)
figure

voronoi(double(tsnefp(1:size(dlnet.Layers(14).Weights,1),1)),...
        double(tsnefp(1:size(dlnet.Layers(14).Weights,1),2)))
hold on;
beginOftYFp = size(dlnet.Layers(14).Weights,1) + 1;
% plot(double(tsnefp(beginOftYFp:beginOftYFp+size(tYfp,1),1)),...
%     double(tsnefp(beginOftYFp:beginOftYFp+size(tYfp,1),2)),'g.')
gscatter(double(tsnefp(beginOftYFp:beginOftYFp+size(tYfp,1)-1,1)),...
         double(tsnefp(beginOftYFp:beginOftYFp+size(tYfp,1)-1,2)),...
         YVal);
%plot(double(tsnefp(29+size(tY2fp,1):end,1)),double(tsnefp(29+size(tY2fp,1):end,2)),'+')
scatter(double(tsnefp(1:beginOftYFp-1,1)),...
    double(tsnefp(1:beginOftYFp-1,2)),...
    'filled','DisplayName','Fingerprints','Marker','^');
% scatter(double(tsnefp(end - size(tY2fp,1)+1:end,1)),...
%     double(tsnefp(end - size(tY2fp,1)+1:end,2)),...
%     'DisplayName','Abnormalities');
axis equal
% set(gca,'FontSize',12)
%%%%%% End of visualizing decision voronoi


