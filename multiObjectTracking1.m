function multiObjectTracking1()
options=0;
obj = setupSystemObjects();

tracks = initializeTracks(); 

nextId = 1; 


while ~isDone(obj.reader)
    frame = readFrame();
    [centroids, bboxes, mask] = detectObjects(frame);
    [cA1] = DiscreteWT(mask);
    [eigvector, eigvalue] = PCA2(cA1, options);

    predictNewLocationsOfTracks();
    [assignments, unassignedTracks, unassignedDetections] = ...
        detectionToTrackAssignment();

    updateAssignedTracks();
    updateUnassignedTracks();
    deleteLostTracks();
    createNewTracks();

    displayTrackingResults();
end
 function obj = setupSystemObjects()
        
        obj.reader = vision.VideoFileReader('Demo1.avi');

        
        obj.videoPlayer = vision.VideoPlayer('Position', [20, 200, 700, 400]);
        obj.maskPlayer = vision.VideoPlayer('Position', [740, 200, 700, 400]);

      

        obj.detector = vision.ForegroundDetector('NumGaussians', 3, ...
            'NumTrainingFrames', 40, 'MinimumBackgroundRatio', 0.7);

        

        obj.blobAnalyser = vision.BlobAnalysis('BoundingBoxOutputPort', true, ...
            'AreaOutputPort', true, 'CentroidOutputPort', true, ...
            'MinimumBlobArea', 400);
 end
function [cA1] = DiscreteWT(mask)
            %N= 60 * 160;
    [LL,LH,HL,HH]=dwt2(mask,'haar');
    b1=[LL,LH;HL,HH];
 %   figure(1)
 %   imshow(b1); 
    [cA1,cH1,cV1,cD1] =dwt2(LL,'haar'); 
   b2=[cA1,cH1;cV1,cD1]; 
end
function [eigvector, eigvalue] = PCA2(b2, options)
if (~exist('options','var'))
    options = [];
end

ReducedDim = 0;
if isfield(options,'ReducedDim')
    ReducedDim = options.ReducedDim;
end


[nSmp,nFea] = size(b2);
if (ReducedDim > nFea) || (ReducedDim <=0)
    ReducedDim = nFea;
end


if issparse(b2)
    b2 = full(b2);
end
sampleMean = mean(b2,1);
b2 = (b2 - repmat(sampleMean,nSmp,1));

[eigvector, eigvalue] = mySVD(b2',ReducedDim);
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
function tracks = initializeTracks()
        
        tracks = struct(...
            'id', {}, ...
            'bbox', {}, ...
            'kalmanFilter', {}, ...
            'age', {}, ...
            'totalVisibleCount', {}, ...
            'consecutiveInvisibleCount', {});
end
function frame = readFrame()
        frame = obj.reader.step();
    end
 function [centroids, bboxes, mask] = detectObjects(frame)

       
        mask = obj.detector.step(frame);

       
        mask = imopen(mask, strel('rectangle', [3,3]));
        mask = imclose(mask, strel('rectangle', [15, 15]));
        mask = imfill(mask, 'holes');

        
        [~, centroids, bboxes] = obj.blobAnalyser.step(mask);
 end


function predictNewLocationsOfTracks()
        for i = 1:length(tracks)
            bbox = tracks(i).bbox;

            
            predictedCentroid = predict(tracks(i).kalmanFilter);

           
            predictedCentroid = int32(predictedCentroid) - bbox(3:4) / 2;
            tracks(i).bbox = [predictedCentroid, bbox(3:4)];
        end
end
function [assignments, unassignedTracks, unassignedDetections] = ...
            detectionToTrackAssignment()

        nTracks = length(tracks);
        nDetections = size(centroids, 1);

        
        cost = zeros(nTracks, nDetections);
        for i = 1:nTracks
            cost(i, :) = distance(tracks(i).kalmanFilter, centroids);
        end

        % Solve the assignment problem.
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

            
            correct(tracks(trackIdx).kalmanFilter, centroid);

           
            tracks(trackIdx).bbox = bbox;

            % Update track's age.
            tracks(trackIdx).age = tracks(trackIdx).age + 1;

            % Update visibility.
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

        % Compute the fraction of the track's age for which it was visible.
        ages = [tracks(:).age];
        totalVisibleCounts = [tracks(:).totalVisibleCount];
        visibility = totalVisibleCounts ./ ages;

        % Find the indices of 'lost' tracks.
        lostInds = (ages < ageThreshold & visibility < 0.6) | ...
            [tracks(:).consecutiveInvisibleCount] >= invisibleForTooLong;

        % Delete lost tracks.
        tracks = tracks(~lostInds);
 end
function createNewTracks()
        centroids = centroids(unassignedDetections, :);
        bboxes = bboxes(unassignedDetections, :);

        for i = 1:size(centroids, 1)

            centroid = centroids(i,:);
            bbox = bboxes(i, :);

            % Create a Kalman filter object.
            kalmanFilter = configureKalmanFilter('ConstantVelocity', ...
                centroid, [200, 50], [100, 25], 100);

            % Create a new track.
            newTrack = struct(...
                'id', nextId, ...
                'bbox', bbox, ...
                'kalmanFilter', kalmanFilter, ...
                'age', 1, ...
                'totalVisibleCount', 1, ...
                'consecutiveInvisibleCount', 0);

            
            tracks(end + 1) = newTrack;

           
            nextId = nextId + 1;
        end
end
function displayTrackingResults()
        
        frame = im2uint8(frame);
        mask = uint8(repmat(mask, [1, 1, 3])) .* 255;

        minVisibleCount = 8;
        if ~isempty(tracks)

            
            reliableTrackInds = ...
                [tracks(:).totalVisibleCount] > minVisibleCount;
            reliableTracks = tracks(reliableTrackInds);

            
            if ~isempty(reliableTracks)
                
                bboxes = cat(1, reliableTracks.bbox);

                % Get ids.
                ids = int32([reliableTracks(:).id]);

               
                labels = cellstr(int2str(ids'));
                predictedTrackInds = ...
                    [reliableTracks(:).consecutiveInvisibleCount] > 0;
                isPredicted = cell(size(labels));
              %  isPredicted(predictedTrackInds) = {' predicted'};
                labels = strcat(labels, isPredicted);

                
                frame = insertObjectAnnotation(frame, 'rectangle', ...
                    bboxes, labels);

                
                mask = insertObjectAnnotation(mask, 'rectangle', ...
                    bboxes, labels);
            end
        end

        % Display the mask and the frame.
       obj.maskPlayer.step(mask);
        obj.videoPlayer.step(frame);
end
end
