function output = preprocessing_context(data,dim, frac_scale, mode)
% PREPROCESSING_CONTEXT can exploits that there are data duplicates
% the function check if this assumtion is true and generate the context
% vectors as fast as possible
% 
% depending on the given mode, the function computes in CPU mode with or without
% the VSA_toolbox, or in GPU mode 
% 
%   INPUT: 
%       data        -   data array with size of n x t x m (n... number of samples,
%                       t... the number of time-steps and m... the number of
%                       sensortypes 
%       dim         -   number of dimensions of the resulting high-dimensional
%                       vectors
%       frac_scale  -   scaling of fractional binding 
%       mode        -   computing mode (0... CPU without VSA_toolbox, 1...
%                       GPU without VSA_toolbox, 2... CPU with VSA_toolbox
%
%   OUTPUT:
%       output      -   output array with size of d x n (d... number of
%                       dimensions and n... the number of samples)
%
% scken, 2021
% Copyright (C) 2021 Chair of Automation Technology / TU Chemnitz

if nargin < 4
    mode = 0; % default is using CPU without VSA_toolbox
end

vsa = 'FHRR'; % default VSA is FHRR

if mode==2 % if mode is VSA_toolbox
    output = preprocessing_context_VSA_toolbox(single(data),dim,frac_scale,vsa);
else
    % check where the duplicate assumtion is true 
    seq1 = data(:,1:32,:);
    seq2 = data(:,33:end,:);
    assum = seq1==circshift(seq2,1,1);
    assum = sum(sum(assum,2),3);
    assum = assum==288; % where is the duplicate sequence true
    
    % extract consecutive ones
    flag = diff([0 assum']);
    idx_start = find(flag==1);
    idx_end = find(flag==-1);
    if idx_end(end) ~= size(data,1)
        idx_end(end+1) = size(data,1)+1;
    end
    current_idx = 1;
    output = [];
        
    for i=1:numel(idx_start)
        % preprocess the data, which are not in block
        for d=current_idx:(idx_start(i)-1)
            if mode == 0; output(end+1,:) = preprocessing_context_fast_cpu(data(d,:,:),dim, frac_scale); end
            if mode == 1; output(end+1,:) = preprocessing_context_fast_gpu(data(d,:,:),dim, frac_scale); end
                
        end
        
        if mode == 0; output = cat(1,output, preprocessing_context_fast_cpu(data(idx_start(i):idx_end(i)-1,:,:),dim, frac_scale)); end
        if mode == 1; output = cat(1,output, preprocessing_context_fast_gpu(data(idx_start(i):idx_end(i)-1,:,:),dim, frac_scale)); end
        
        current_idx = idx_end(i);
    end
end