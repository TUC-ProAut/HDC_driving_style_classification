%% processing script to create the HDC encodings 
% scken, 2021
% Copyright (C) 2021 Chair of Automation Technology / TU Chemnitz

dim = 2048;
scale = 6;
vsa = 'FHRR';
mode = 0; % 0... use CPU, 1... use GPU, 2... use the VSA_toolbox  

% load dataset
data = load('data/uah_dataset.mat');

output_motor = preprocessing_context(single(data.motorway_dataset),dim,scale,mode);
output_second = preprocessing_context(single(data.secondary_dataset),dim,scale,mode);

% save files 
motorway_labels = data.motorway_labels;
secondary_labels = data.secondary_labels;
save(['data/preproc_data_' num2str(dim) '_' num2str(scale)],'output_motor','output_second','motorway_labels','secondary_labels');

        