function output = preprocessing_context(data,dim,frac_scale,vsa,seed)
% PREPROCESSING_CONTEXT creates a HDC encoding based on the given input data 
% with the VSA_toolbox (https://github.com/TUC-ProAut/VSA_Toolbox) (is
% required)
% 
%   INPUT: 
%       data        -   data array with size of nxtxm (n... number of samples,
%                       t... the number of time-steps and m... the number of
%                       sensortypes 
%       dim         -   number of dimensions of the resulting high-dimensional
%                       vectors
%       frac_scale  -   scaling of fractional binding 
%       vsa         -   name of the used VSA implementation (from the VSA
%                       Toolbox) [string] 
%       seed        -   random seed
%   OUTPUT: 
%       output      -   output array with size of dxn (d... number of
%                       dimensions and m... the number of samples)
% scken, 2021
% Copyright (C) 2021 Chair of Automation Technology / TU Chemnitz
    
    if nargin < 5
        seed = 0;
    end
    
    try
        VSA = vsa_env('vsa',vsa,'dim',dim);
    catch
        error('Cannot find the VSA toolbox. Please download it from https://github.com/TUC-ProAut/VSA_Toolbox and add it to MATLAB path!')
        return
    end

    rand('seed',seed);
    init_vector = VSA.add_vector();
    sensor_enc_vec = VSA.add_vector('num',size(data,3));
    timestamps_vecs = VSA.add_vector('num',size(data,2));

    output = zeros([size(data,1) dim]);

    parfor i=1:size(data,1)
        % value encoding
        encoded_values = VSA.frac_binding(init_vector,data(i,:,:) * frac_scale);

        % bind to sensortype
        sensor_value_pairs = VSA.bind(encoded_values, permute(repmat(sensor_enc_vec,[1 1 size(data,2)]),[1 3 2]));

        % bundle to one timestep
        sensor_bundle = squeeze(VSA.bundle(permute(sensor_value_pairs,[1 3 2]),[]));

        % encode time context
        context_vecs = VSA.bind(sensor_bundle, timestamps_vecs);
        context_vecs = VSA.bundle(context_vecs,[]);

        output(i,:) = context_vecs;
    end

    output=single(output);
end