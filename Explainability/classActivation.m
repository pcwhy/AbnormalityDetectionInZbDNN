% netName = "squeezenet";
% netName = "googlenet";
% netName = "mobilenetv2";
netName = "resnet18";

net = eval(netName);

% camera = webcam;
im = imread('./shoes.jpg');

inputSize = net.Layers(1).InputSize(1:2);
classes = net.Layers(end).Classes;
layerName = activationLayerName(netName);
% h = figure('Units','normalized','Position',[0.05 0.05 0.9 0.8],'Visible','on');

% while ishandle(h)
%     im = snapshot(camera);
    imResized = imresize(im,[inputSize(1), NaN]);
    imageActivations = activations(net,imResized,layerName);    
    pooling = squeeze(mean(imageActivations,[1 2]));
    
    if netName ~= "squeezenet"
        fcWeights = net.Layers(end-2).Weights;
        fcBias = net.Layers(end-2).Bias;
        scores =  fcWeights*pooling + fcBias;
        
        [~,classIds] = maxk(scores,3);
        
        classTemplate = shiftdim(fcWeights(classIds(1),:),-1);
        classActivationMap = sum(imageActivations.*classTemplate,3);
    else
        [~,classIds] = maxk(scores,3);
        classActivationMap = imageActivations(:,:,classIds(1));
    end
    scores = exp(scores)/sum(exp(scores));     
    maxScores = scores(classIds);
    labels = classes(classIds);        
    subplot(1,2,1)
    imshow(im)
    
    subplot(1,2,2)
    CAMshow(im,classActivationMap)
    title(string(labels) + ", " + string(maxScores));
    
%     drawnow
    
% end    
% clear camera

function CAMshow(im,CAM)
    imSize = size(im);
    CAM = imresize(CAM,imSize(1:2));
    CAM = normalizeImage(CAM);
    CAM(CAM<0.2) = 0;
    cmap = jet(255).*linspace(0,1,255)';
    CAM = ind2rgb(uint8(CAM*255),cmap)*255;

    combinedImage = double(rgb2gray(im))/2 + CAM;
    combinedImage = normalizeImage(combinedImage)*255;
    imshow(uint8(combinedImage));
end

function N = normalizeImage(I)
    minimum = min(I(:));
    maximum = max(I(:));
    N = (I-minimum)/(maximum-minimum);
end

function layerName = activationLayerName(netName)
    if netName == "squeezenet"
        layerName = 'relu_conv10';
    elseif netName == "googlenet"
        layerName = 'inception_5b-output';
    elseif netName == "resnet18"
        layerName = 'res5b_relu';
    elseif netName == "mobilenetv2"
        layerName = 'out_relu';
    end
end
