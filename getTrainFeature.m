%{ 
***************************************************************************
 * 
 * Copyright (c) wyr 2015. All Rights Reserved
 * 
 **************************************************************************
%}
% Note:
%       Outputs: trainFeature and trainLabel

clear;
%% Initialization 

% Set Related Para. (Known Information)
trainNumLesionPerBrain = 1000;
trainNumIntactPerBrain = 2000;
featureDim = 3 + 123 + 120;
brainSize = [256,256,176];
dataDir = '/home/joyce/HDData/brainData/';
brainList = dir([dataDir 'BU*']);
% Set Histogram Equalization minimum density and maximum density
minThreshold_HE = 10;
maxThreshold_HE = 400;
% Initialize Gabor Filter 
gaborArray = gaborFilterBank(5, 8, 15, 15);
% Initialize slice dim
dimSlice = 2;

trainNumPerBrain = trainNumLesionPerBrain + trainNumIntactPerBrain;
brainNum = length(brainList);
% Initialize Output trainFeature and trainLabel
trainFeature = zeros((trainNumPerBrain) * brainNum, featureDim);
trainLabel = zeros((trainNumPerBrain) * brainNum, 1);

for brainIdx = 1:brainNum
%% Read Brain Data

fname_T1 = [brainList(brainIdx).name '_T1.nii'];
Data2Read_T1 = fullfile([dataDir brainList(brainIdx).name],fname_T1);
HeaderInfo_T1 = spm_vol(Data2Read_T1);
brainOrig = spm_read_vols(HeaderInfo_T1);

fname_lesion = [brainList(brainIdx).name '_lesion.nii'];
Data2Read_lesion = fullfile([dataDir brainList(brainIdx).name '/'],fname_lesion);
HeaderInfo_lesion = spm_vol(Data2Read_lesion);
lesion = spm_read_vols(HeaderInfo_lesion);

fname_mask = [brainList(brainIdx).name '_mask.nii'];
Data2Read_mask = fullfile([dataDir brainList(brainIdx).name '/skullStrip'],fname_mask);
HeaderInfo_mask = spm_vol(Data2Read_mask);
maskSkullStrip = spm_read_vols(HeaderInfo_mask);

%% Get Skull Stripped Brain

% Modify the skull striped mask using Symmetry
maskSym = zeros(brainSize);
for i = 1 : brainSize(2)
    maskSym(:,i,:) = fliplr(squeeze(maskSkullStrip(:,i,:)));
%     brainStrippedSym(:,:,i) = flipud(squeeze(brainStripped(:,:,i)));
end
maskSkullStrip = maskSkullStrip | maskSym;
% Get Skull Stripped Brain
brainOrig = brainOrig .* maskSkullStrip;

%% MRI Signal Standardization

% Normalize Intensity and Enhance Contrast
brain = histogramEqual(brainOrig, maskSkullStrip, minThreshold_HE, maxThreshold_HE);

%% Extract Features

% Initialize
trainLesionPerBrain = zeros(trainNumLesionPerBrain, featureDim);
trainIntactPerBrain = zeros(trainNumIntactPerBrain, featureDim);
trainFeaturePerBrain = zeros(trainNumPerBrain, featureDim);
trainLabelPerBrain = zeros(trainNumPerBrain, 1);

% Generate trainFeaturePerBrain Lesion and Intact Voxels' Index (idxTrainLesion & idxTrainIntact)
% Lesion voxel Index
idxLesionVoxel = find(lesion == 1);
tmp = randperm(length(idxLesionVoxel));
idxTrainLesion = idxLesionVoxel(tmp(1 : trainNumLesionPerBrain));  

% Intact voxel Index
maskNoLesion = maskSkullStrip - lesion;
idxIntactVoxel = find(maskNoLesion == 1);
tmp = randperm(length(idxIntactVoxel));
idxTrainIntact = idxIntactVoxel(tmp(1 : trainNumIntactPerBrain));  

% Calculate centroid of the brain
[rows, cols, pag] = size(brain);
y = 1:rows;
x = 1:cols;
z = 1:pag;
[X,Y,Z] = meshgrid(x,y,z);
cY = mean(Y(brain ~= 0));
cX = mean(X(brain ~= 0));
cZ = mean(Z(brain ~= 0));

%% Feature1: space feature
[row,col,pag] = ind2sub(brainSize,idxTrainIntact);
trainIntactPerBrain(:,1) = row; 
trainIntactPerBrain(:,2) = col;
trainIntactPerBrain(:,3) = pag;

[row,col,pag] = ind2sub(brainSize,idxTrainLesion);
trainLesionPerBrain(:,1) = row; 
trainLesionPerBrain(:,2) = col;
trainLesionPerBrain(:,3) = pag;

trainFeaturePerBrain = [trainLesionPerBrain;trainIntactPerBrain];
trainLabelPerBrain = [ones(trainNumLesionPerBrain,1); zeros(trainNumIntactPerBrain,1)];

%% Feature2: Neighbor Intensity Feature
% % Generate neighbor voxel (cube: 5*5*5) coordinate mask 
% neighborNum = 125;
% maskNeighbor = zeros(125,3);
% count = 1;
% for i = -2:1:2
%     for j = -2:1:2
%         for k = -2:1:2
%             maskNeighbor(count,1) = i;
%             maskNeighbor(count,2) = j;
%             maskNeighbor(count,3) = k;
%             count = count + 1;
%         end
%     end
% end

% Generate neighbor voxel (Sphere Mask radius = 3) coordinate mask
[x,y,z] = ndgrid(-3: 3);
se = strel(sqrt(x.^2 + y.^2 + z.^2) <= 3);
maskNeighbor = getneighbors(se);
neighborNum = length(maskNeighbor);

% Add intensity of neighbor voxels to Train Feature
for i = 1:trainNumPerBrain
    for j = 1:neighborNum        
        coord = trainFeaturePerBrain(i, 1:3) + maskNeighbor(j, :);
        trainFeaturePerBrain(i, j+3) = brain(coord(1), coord(2), coord(3));
    end
end
%% Feature3: Gabor Filter Feature

gaborFeatureAllSlice = cell(brainSize(3), 1);
for i = 1:brainSize(dimSlice)
    slice = squeeze(brain(:,i,:));
    gaborFeatureAllSlice{i} = gaborFeaturesMatrix(slice, gaborArray, 1, 1);
end

for i = 1:trainNumIntactPerBrain
    idxSlice = trainIntactPerBrain(i,dimSlice);
    gaborFeaVoxel = zeros(120,1);
    count = 0;
    for h = -1:1
        gaborSlice = gaborFeatureAllSlice{idxSlice+h};
        for u = 1:5
            for v = 1:8
                gaborImg = gaborSlice{u,v};
                count = count+1;
                gaborFeaVoxel(count) = gaborImg(trainIntactPerBrain(i,3), trainIntactPerBrain(i,1));         
            end
         end
    end
    trainIntactPerBrain(i,127:246) = gaborFeaVoxel;
end

for i = 1:trainNumLesionPerBrain
    idxSlice = trainLesionPerBrain(i,dimSlice);
    gaborFeaVoxel = zeros(120,1);
    count = 0;
    for h = -1:1
        gaborSlice = gaborFeatureAllSlice{idxSlice+h};
        for u = 1:5
            for v = 1:8
                gaborImg = gaborSlice{u,v};
                count = count+1;
                gaborFeaVoxel(count) = gaborImg(trainLesionPerBrain(i,3), trainLesionPerBrain(i,1));         
            end
         end
    end
    trainLesionPerBrain(i,127:246) = gaborFeaVoxel;
end

%% Normalize Space Coord for Feature1
trainFeaturePerBrain(:,1) = trainFeaturePerBrain(:,1) - cX; 
trainFeaturePerBrain(:,2) = trainFeaturePerBrain(:,2) - cY;
trainFeaturePerBrain(:,3) = trainFeaturePerBrain(:,3) - cZ;

%% Add trainFeaturePerBrain to the overall trainFeature
startIdx = (brainIdx - 1) * trainNumPerBrain + 1;
endIdx = brainIdx * trainNumPerBrain;
tmp = [trainLesionPerBrain;trainIntactPerBrain];
trainFeaturePerBrain(:,127:246) = tmp(:,127:246);
trainFeature(startIdx:endIdx, :) = trainFeaturePerBrain;
trainLabel(startIdx:endIdx) = trainLabelPerBrain;
end

% % %% Train Model
% trainFeature = scale_norm(trainFeature);
% trainFeature = L2_norm(trainFeature);
% %g = rbf_g(trainFeature);
% 
% model = svmtrain(trainLabel, trainFeature, ['-c 10 -g ' num2str(g) ' -b 1 -t 2']);
% %save('model_HE','model');