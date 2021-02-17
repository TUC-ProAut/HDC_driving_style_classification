%% processing script to create the HDC encodings for the complete hyper-parameter analysis 
% scken, 2021
% Copyright (C) 2021 Chair of Automation Technology / TU Chemnitz

vsa = 'FHRR';
dim = [2048]%[512 1024 2048];
scale = 2:2:10;
use_gpu = 1; % 1... use GPU, 0... use CPU, other... use the VSA_toolbox 

data = load('data/uah_dataset.mat');

for d=1:numel(dim)
    for s=1:numel(scale)
        disp(['Dimension = ' num2str(dim(d)) ' and scale = ' num2str(scale(s))]);
        if use_gpu == 0
            output_motor = preprocessing_context_fast_cpu(single(data.motorway_dataset),dim(d),scale(s));
            output_second = preprocessing_context_fast_cpu(single(data.secondary_dataset),dim(d),scale(s));
        elseif use_gpu == 1
            output_motor = preprocessing_context_fast_gpu(gpuArray(single(data.motorway_dataset)),dim(d),scale(s));
            output_second = preprocessing_context_fast_gpu(gpuArray(single(data.secondary_dataset)),dim(d),scale(s));
        else
            % the VSA_toolbox will be used
            output_motor = preprocessing_context(single(data.motorway_dataset),dim(d),vsa,scale(s));
            output_second = preprocessing_context(single(data.secondary_dataset),dim(d),vsa,scale(s));
        end
        % save files 
        motorway_labels = data.motorway_labels;
        secondary_labels = data.secondary_labels;
        save(['data/preproc_data_' num2str(dim(d)) '_' num2str(scale(s))],'output_motor','output_second','motorway_labels','secondary_labels');
    end
end