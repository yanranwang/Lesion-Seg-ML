%{ 
***************************************************************************
 * 
 * Copyright (c) wyr 2015. All Rights Reserved
 * 
 **************************************************************************
%}
 
function brainNew = histogramEqual(brain, maskSkullStrip, minThreshold, maxThreshold)

% Transform Brain Matrix to Vector
brainVector = brain(find(maskSkullStrip == 1));
voxelNum = numel(brainVector);
minVoxelOrig = min(brainVector);
maxVoxelOrig = max(brainVector);
edges = minVoxelOrig:1:maxVoxelOrig;

% Get Histogram Count: the number of voxels with each intensity value
[n, bin] = histc(brainVector,edges);

% Initialize Cumulative Distribution Function (CDF) and Intensity Mapping Function (MAP)
MAP = zeros(length(edges),2);
CDF = zeros(length(edges),2);
MAP(:,1) = edges;
CDF(:,1) = edges;

% Generate Cumulative Distribution Function (CDF)
for i = 1:length(CDF)
    CDF(i,2) = sum(n(1:i));    
end

% Generate Mapping Intensity Function (MAP)
cdf_min = min(CDF(:,2));
for i = 1:length(MAP)
    MAP(i,2) = round((CDF(i,2) - cdf_min) / (voxelNum - cdf_min) * (maxThreshold - minThreshold)) + minThreshold;
end

% Generate brainNew using Intensity Mapping Function (MAP)
brainNew = zeros(size(brain));
for i = 1:length(MAP)
    brainNew(find(brain == MAP(i,1))) = MAP(i,2);
end
end
