%% processing script to create the HDC encodings 
% scken, 2021
% Copyright (C) 2021 Chair of Automation Technology / TU Chemnitz

dim = 2048;
scale = 6;
vsa = 'FHRR';
use_gpu = 0; % 1... use GPU, 0... use CPU, other... use the VSA_toolbox  

% load dataset
data = load('data/uah_dataset.mat');

if use_gpu == 0
    output_motor = preprocessing_context_fast_cpu(single(data.motorway_dataset),dim,scale);
    output_second = preprocessing_context_fast_cpu(single(data.secondary_dataset),dim,scale);
elseif use_gpu == 1
    output_motor = preprocessing_context_fast_gpu(gpuArray(single(data.motorway_dataset)),dim,scale);
    output_second = preprocessing_context_fast_gpu(gpuArray(single(data.secondary_dataset)),dim,scale);
else
    % the VSA_toolbox will be used
    output_motor = preprocessing_context(single(data.motorway_dataset),dim,scale,vsa);
    output_second = preprocessing_context(single(data.secondary_dataset),dim,scale,vsa);
end

% save files 
motorway_labels = data.motorway_labels;
secondary_labels = data.secondary_labels;
save(['data/preproc_data_' num2str(dim) '_' num2str(scale)],'output_motor','output_second','motorway_labels','secondary_labels');

        