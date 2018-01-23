function obDetectAndTrack()
options = 0;

obj = setupSystemObjects();
tracks = initializeTracks();
nextId = 1;

while ~isDone(obj.reader)
    frame = readFrame();
    [centroids, bboxes, mask, mask1] = detectObjects(frame);
    [cA1] = DiscreteWT(mask);
    [eigvector, eigvalue] = PCA2(cA1, options);
        FindLocations();
    [assignments, unassignedTracks, unassignedDetections] = ...
        detectionToTrackAssignment();
    
    updateAssignedTracks();
    updateUnassignedTracks();
    deleteLostTracks();
    createNewTracks();
    
    display();
end
function obj = setupSystemObjects()
obj.reader = vision.VideoFileReader('Demo1.avi');
obj.videoPlayer = vision.VideoPlayer('Position', [20, 400, 700, 400]);
obj.DWTPlayer = vision.VideoPlayer('Position', [720, -400, 700, 400]);
obj.maskPlayer = vision.VideoPlayer('Position', [40, -400, 700, 400]);
obj.mask1Player = vision.VideoPlayer('Position', [740, 400, 700, 400]);
obj.detector = vision.ForegroundDetector('NumGaussians', 5,...
            'NumTrainingFrames', 40, 'MinimumBackgroundRatio', 0.7);
       obj.blobAnalyser = vision.BlobAnalysis('BoundingBoxOutputPort', true, ...
            'AreaOutputPort', true, 'CentroidOutputPort', true, ...
            'MinimumBlobArea', 400);
        
        
end
 function tracks = initializeTracks()
        % empty array of tracks

        tracks = struct(...
            'id', {}, ...
            'bbox', {}, ...
            'position', {}, ...
            'whenOccluded', {}, ...
            'age', {}, ...
            'totalVisibleCount', {}, ...
            'consecutiveInvisibleCount', {});
 end

 function FindLocations()
        for i = 1:length(tracks)
            bbox = tracks(i).bbox;         
            FindPosition(eigvector);  
          
            Position = tracks(i).position;
            
            [r,c] = size(Position);
            if( r == c )
                 Position = int32(Position(1:c)) - bbox(3:4) / 2;
            else
                Position = int32(Position) - bbox(3:4) / 2;
            end
            tracks(i).bbox = [Position, bbox(3:4)];
            WhenOccluded = predict(tracks(i).whenOccluded);
           
        end
    end



function [centroids, bboxes, mask, mask1] = detectObjects(frame)
    mask = obj.detector.step(frame);
    mask = imopen(mask, strel('rectangle', [4,4])); 
    mask = imclose(mask, strel('rectangle', [16, 16])); 
    mask = imfill(mask, 'holes');
    mask1 = obj.detector.step(frame);
      [~, centroids, bboxes] = obj.blobAnalyser.step(mask);
end
function frame = readFrame()

    frame = obj.reader.step();
 
end
function [cA1] = DiscreteWT(mask)
    [LL,LH,HL,HH]=dwt2(mask,'haar');
    b1=[LL,LH;HL,HH];

    [cA1,cH1,cV1,cD1] =dwt2(LL,'haar'); 
   b2=[cA1,cH1;cV1,cD1]; 
end

function [assignments, unassignedTracks, unassignedDetections] = ...
            detectionToTrackAssignment()
      
        nTracks = length(tracks);
        nDetections = size(centroids, 1);
        
  
        cost = zeros(nTracks, nDetections);
        for i = 1:nTracks

            cost(i, :) = distance(tracks(i).whenOccluded, centroids);
        end
        
 
        costOfNonAssignment = 20;
        [assignments, unassignedTracks, unassignedDetections] = ...
            assignDetectionsToTracks(cost, costOfNonAssignment);
    end

 function updateAssignedTracks()
        numAssignedTracks = size(assignments, 1);
        for i = 1:numAssignedTracks
            trackIdx = assignments(i, 1);
            detectionIdx = assignments(i, 2);
            centroid = centroids(detectionIdx, :);
            bbox = bboxes(detectionIdx, :);
            
        
            correct(tracks(trackIdx).whenOccluded, centroid);
            
          
            tracks(trackIdx).bbox = bbox;
            
       
            tracks(trackIdx).age = tracks(trackIdx).age + 1;
            
          
            tracks(trackIdx).totalVisibleCount = ...
                tracks(trackIdx).totalVisibleCount + 1;
            tracks(trackIdx).consecutiveInvisibleCount = 0;
        end
 end

function updateUnassignedTracks()
        for i = 1:length(unassignedTracks)
            ind = unassignedTracks(i);
            tracks(ind).age = tracks(ind).age + 1;
            tracks(ind).consecutiveInvisibleCount = ...
                tracks(ind).consecutiveInvisibleCount + 1;
        end
end

function deleteLostTracks()
        if isempty(tracks)
            return;
        end
        
        invisibleForTooLong = 20;
        ageThreshold = 8;
        
    
        ages = [tracks(:).age];
        totalVisibleCounts = [tracks(:).totalVisibleCount];
        visibility = totalVisibleCounts ./ ages;
        

        lostInds = (ages < ageThreshold & visibility < 0.6) | ...
            [tracks(:).consecutiveInvisibleCount] >= invisibleForTooLong;
        
    
        tracks = tracks(~lostInds);
end

function createNewTracks()
        centroids = centroids(unassignedDetections, :);
        bboxes = bboxes(unassignedDetections, :);
        
        for i = 1:size(centroids, 1)
            
   
            centroid = centroids(i,:);
            bbox = bboxes(i, :);
            
     
            State = configureKalmanFilter('ConstantVelocity', ...
                centroid, [200, 50], [100, 25], 100);
            

            newTrack = struct(...
                'id', nextId, ...
                'bbox', bbox, ...
                'position', centroids, ... 
                'whenOccluded', State, ...
                'age', 1, ...
                'totalVisibleCount', 1, ...
                'consecutiveInvisibleCount', 0);
            
            % Add it to the array of tracks.
            tracks(end + 1) = newTrack;
            
            % Increment the next id.
            nextId = nextId + 1;
        end
    end


function [eigvector, eigvalue] = PCA2(data, options)
if (~exist('options','var'))
    options = [];
end

ReducedDim = 0;
if isfield(options,'ReducedDim')
    ReducedDim = options.ReducedDim;
end


[nSmp,nFea] = size(data);
if (ReducedDim > nFea) || (ReducedDim <=0)
    ReducedDim = nFea;
end


if issparse(data)
    data = full(data);
end
sampleMean = mean(data,1);
data = (data - repmat(sampleMean,nSmp,1));

[eigvector, eigvalue] = mySVD(data',ReducedDim);
eigvalue = full(diag(eigvalue)).^2;

if isfield(options,'PCARatio')
    sumEig = sum(eigvalue);
    sumEig = sumEig*options.PCARatio;
    sumNow = 0;
    for idx = 1:length(eigvalue)
        sumNow = sumNow + eigvalue(idx);
        if sumNow >= sumEig
            break;
        end
    end
    eigvector = eigvector(:,1:idx);
end
end

function [U, S, V] = mySVD(X,ReducedDim)
%mySVD    Accelerated singular value decomposition.
%   [U,S,V] = mySVD(X) produces a diagonal matrix S, of the  
%   dimension as the rank of X and with nonnegative diagonal elements in
%   decreasing order, and unitary matrices U and V so that
%   X = U*S*V'.
%
%   [U,S,V] = mySVD(X,ReducedDim) produces a diagonal matrix S, of the  
%   dimension as ReducedDim and with nonnegative diagonal elements in
%   decreasing order, and unitary matrices U and V so that
%   Xhat = U*S*V' is the best approximation (with respect to F norm) of X
%   among all the matrices with rank no larger than ReducedDim.
%
%   Based on the size of X, mySVD computes the eigvectors of X*X^T or X^T*X
%   first, and then convert them to the eigenvectors of the other.  

                                             

MAX_MATRIX_SIZE = 1600; % You can change this number according your machine computational power
EIGVECTOR_RATIO = 0.1; % You can change this number according your machine computational power


if ~exist('ReducedDim','var')
    ReducedDim = 0;
end

[nSmp, mFea] = size(X);
if mFea/nSmp > 1.0713
    ddata = X*X';
    ddata = max(ddata,ddata');
    
    dimMatrix = size(ddata,1);
    if (ReducedDim > 0) && (dimMatrix > MAX_MATRIX_SIZE) && (ReducedDim < dimMatrix*EIGVECTOR_RATIO)
        option = struct('disp',0);
        [U, eigvalue] = eigs(ddata,ReducedDim,'la',option);
        eigvalue = diag(eigvalue);
    else
        if issparse(ddata)
            ddata = full(ddata);
        end
        
        [U, eigvalue] = eig(ddata);
        eigvalue = diag(eigvalue);
        [~, index] = sort(-eigvalue);
        eigvalue = eigvalue(index);
        U = U(:, index);
    end
    clear ddata;
    
    maxEigValue = max(abs(eigvalue));
    eigIdx = find(abs(eigvalue)/maxEigValue < 1e-10);
    eigvalue(eigIdx) = [];
    U(:,eigIdx) = [];
    
    if (ReducedDim > 0) && (ReducedDim < length(eigvalue))
        eigvalue = eigvalue(1:ReducedDim);
        U = U(:,1:ReducedDim);
    end
    
    eigvalue_Half = eigvalue.^.5;
    S =  spdiags(eigvalue_Half,0,length(eigvalue_Half),length(eigvalue_Half));

    if nargout >= 3
        eigvalue_MinusHalf = eigvalue_Half.^-1;
        V = X'*(U.*repmat(eigvalue_MinusHalf',size(U,1),1));
    end
else
    ddata = X'*X;
    ddata = max(ddata,ddata');
    
    dimMatrix = size(ddata,1);
    if (ReducedDim > 0) && (dimMatrix > MAX_MATRIX_SIZE) && (ReducedDim < dimMatrix*EIGVECTOR_RATIO)
        option = struct('disp',0);
        [V, eigvalue] = eigs(ddata,ReducedDim,'la',option);
        eigvalue = diag(eigvalue);
    else
        if issparse(ddata)
            ddata = full(ddata);
        end
        
        [V, eigvalue] = eig(ddata);
        eigvalue = diag(eigvalue);
        
        [~, index] = sort(-eigvalue);
        eigvalue = eigvalue(index);
        V = V(:, index);
    end
    clear ddata;
    
    maxEigValue = max(abs(eigvalue));
    eigIdx = find(abs(eigvalue)/maxEigValue < 1e-10);
    eigvalue(eigIdx) = [];
    V(:,eigIdx) = [];
    
    if (ReducedDim > 0) && (ReducedDim < length(eigvalue))
        eigvalue = eigvalue(1:ReducedDim);
        V = V(:,1:ReducedDim);
    end
    
    eigvalue_Half = eigvalue.^.5;
    S =  spdiags(eigvalue_Half,0,length(eigvalue_Half),length(eigvalue_Half));
    
    eigvalue_MinusHalf = eigvalue_Half.^-1;
    U = X*(V.*repmat(eigvalue_MinusHalf',size(V,1),1));
end
end


function display()

    frame = im2uint8(frame);
        mask = uint8(repmat(mask, [1, 1, 3])) .* 255;
        plot(eigvector);
        
        minVisibleCount = 8;
        if ~isempty(tracks)
              
         
            reliableTrackInds = ...
                [tracks(:).totalVisibleCount] > minVisibleCount;
            reliableTracks = tracks(reliableTrackInds);
            

            if ~isempty(reliableTracks)
                % Get bounding boxes.
                bboxes = cat(1, reliableTracks.bbox);
                
                % Get ids.
                ids = int32([reliableTracks(:).id]);
        
                labels = cellstr(int2str(ids'));
  
       
                
                % Draw the objects on the frame.
                frame = insertObjectAnnotation(frame, 'rectangle', ...
                    bboxes, labels);
                
                % Draw the objects on the mask.
                mask = insertObjectAnnotation(mask, 'rectangle', ...
                    bboxes, labels);
            end
        end
        obj.videoPlayer.step(frame);
        obj.maskPlayer.step(mask);        
        obj.DWTPlayer.step(cA1);
        obj.mask1Player.step(mask1);    
end

function FindPosition(eigvector)
    if isempty(eigvector),  return; end
row1 = sum(eigvector,3);
if(row1 == -1), return; end
nonZero = true;
j = 1;
[r,c] = size(eigvector);
for i=1:r
for k=1:c
element = eigvector(i:k) ;
if(nonZero)
    if (element > 0.04) 
        tracks(j).position(1) = k;
        if (( i == r ) || ((i == r) && (k == c))) 
            tracks(j).position(2) = 75;
        end
        nonZero = false;
    end
elseif ( ~nonZero)
    if (element == 0)
      tracks(j).position(2) = k-1;
        if (( i == r ) || ((i == r) && (k == c))) 
            tracks(j).position(1) = 75;
        end
        
        j = j + 1;
        nonZero = true;
    end
end
end
end
end
end

