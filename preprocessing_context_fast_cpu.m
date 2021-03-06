function output = preprocessing_context_fast_cpu(data,dim,frac_scale)
% PREPROCESSING_CONTEXT_FAST_CPU is a runtime optimized preprocessing script 
% to encode input data with a VSA running on CPU (without VSA_toolbox)
%
% This version can exploits that there are data duplicates
% e.g. in the UAH dataset. There, the end (second half 33:64) of each sequence is the beginning 
% (first halt 1:32) of the next sequence. This fundtion vesrion avoids duplicate computations.
%
% The function does a lot of initial allococation and some precomputations. The relevant 
% runtime is measured inside the function!
% 
% data ... single(!) array
% 
% example call:
%
% data = load('data/uah_dataset.mat');
% D = data.motorway_dataset(1:2000,:,:); DU = single(D);
% output=preprocessing_context_fast_duplicateData(DU,2048,6);
% 
%   INPUT: 
%       data            -   data array with size of n x t x m (n... number of samples,
%                           t... the number of time-steps and m... the number of
%                           sensortypes 
%       dim             -   number of dimensions of the resulting high-dimensional
%                           vectors
%       frac_scale      -   scaling of fractional binding 
%
%   OUTPUT:
%       output_complete -   output array with size of n x d (d... number of
%                           dimensions and n... the number of samples)
%
% nepe, Feb 2021
% Copyright (C) 2021 Chair of Automation Technology / TU Chemnitz
    seqL = size(data,2);

    % exploit double entries
    % the assumption is that there is a 50% overlap between sequences
    seqL = floor(seqL/2);
    data_orig = data;
    data = data(:,1:seqL,:);
    % add last half sequence 
    data(end+1,:,:) = data_orig(end,seqL+1:end,:);

    bs = size(data,1); 
    
    % used the same HD vector generated by the VSA_toolbox in script
    % 'preprocessing_context.m'
    Lo = load('data/oldInitSingle.mat');
    init_vector = frac_scale*reshape(Lo.init_vector(1:dim), 1, 1, dim);
    for i=1:9
        sensor_enc_vec(i,1,:) = reshape(Lo.sensor_enc_vec(1:dim,i), 1, 1, dim);
    end
    for i=1:64
        timestamps_vecs(1,i,:) = reshape(Lo.timestamps_vecs(1:dim,i), 1, 1, dim);
    end
    timestamps_mat = repmat( timestamps_vecs, bs, 1, 1 );
        
    % allocate matrices
    E = zeros(bs, seqL, dim, 'single');
    SE = zeros(bs, seqL, dim, 'single');
    
    init_vector = init_vector;
    sensor_enc_vec = sensor_enc_vec;
    timestamps_mat1 = timestamps_mat(1:end-1,1:seqL,:);
    timestamps_mat2 = timestamps_mat(1:end-1,seqL+1:end,:);
        
    BSE_cos_sum = zeros(bs, seqL, dim, 'single');    
    BSE_sin_sum = zeros(bs, seqL, dim, 'single'); 
    
    % start runtime measure
    % everything before this point could be done once per lifetime of the algorithm
    totalTime = tic();
    
    % Encode data for each sensor type and accumulate
    % This Matlab loop does not hurt since it has only few iterations (e.g. 9 for UAH dataset)        
    for sensorID = 1:size(data,3)
        X = data(:,:,sensorID);        
                    
        % encode using fractional binding
        % wrapping to +/- pi is not required since 
        E = X .* init_vector; % expands to a result of size bs x seqL x dim, each channel is one HDC dimension
        
        % bind to sensor ID
        SE = E + sensor_enc_vec(sensorID,1,:); % again, this automatically expands the vector sensor_enc_vec
        
        % accumulate                
        if i==1 % this is (IMHO) a fast way to reinitialize the large arrarys
            BSE_cos_sum = cos(SE);                
            BSE_sin_sum = sin(SE);
        else
            BSE_cos_sum = BSE_cos_sum + cos(SE);                
            BSE_sin_sum = BSE_sin_sum + sin(SE);
        end
    end
    
    % get angle
    BSE = atan2(BSE_sin_sum, BSE_cos_sum); % again of size bs x seqL x dim        
    clear BSE_sin_sum, BSE_cos_sum;
        
    %alternative approach: before binding, concatenate data
    %BSE = cat(2, BSE(1:end-1,:,:), BSE(2:end,:,:)); 
        
    % bind to time ID
    TBSE1 = BSE(1:end-1,:,:) + timestamps_mat1;
    TBSE2 = BSE(2:end,:,:) + timestamps_mat2;   
    clear BSE;
    
    % bundle over time
    TBSE_cos1 = cos(TBSE1);
    TBSE_sin1 = sin(TBSE1);
    TBSE_cos2 = cos(TBSE2);
    TBSE_sin2 = sin(TBSE2);
        
    TBSE_cos_sum1 = sum(TBSE_cos1, 2);
    TBSE_sin_sum1 = sum(TBSE_sin1, 2);
    TBSE_cos_sum2 = sum(TBSE_cos2, 2);
    TBSE_sin_sum2 = sum(TBSE_sin2, 2);
    
    TBSE_cos_sum = TBSE_cos_sum1 + TBSE_cos_sum2;
    TBSE_sin_sum = TBSE_sin_sum1 + TBSE_sin_sum2;
    
    STBSE = atan2(TBSE_sin_sum, TBSE_cos_sum);
    
    % finish
    output = reshape(STBSE, size(STBSE,1), size(STBSE,3));

    totalTime = toc(totalTime);
    
    fprintf('Runtime is %f  (%f ms per sequence)\n', totalTime, totalTime*1000/size(data,1)); 
end
   


