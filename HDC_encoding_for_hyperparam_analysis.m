%% processing script to create the HDC encodings for the complete hyper-parameter analysis 
% scken, 2021
% Copyright (C) 2021 Chair of Automation Technology / TU Chemnitz

vsa = 'FHRR';
dim = [512 1024 2048];
scale = 2:2:10;
mode = 1; % 0... use CPU, 1... use GPU, 2... use the VSA_toolbox  

data = load('data/uah_dataset.mat');

for d=1:numel(dim)
    for s=1:numel(scale)
        disp(['Dimension = ' num2str(dim(d)) ' and scale = ' num2str(scale(s))]);
        output_motor = preprocessing_context(single(data.motorway_dataset),dim(d),scale(s),mode);
        output_second = preprocessing_context(single(data.secondary_dataset),dim(d),scale(s),mode);

        % save files 
        motorway_labels = data.motorway_labels;
        secondary_labels = data.secondary_labels;
        save(['data/preproc_data_' num2str(dim(d)) '_' num2str(scale(s))],'output_motor','output_second','motorway_labels','secondary_labels');
    end
end