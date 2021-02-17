%% create figures and tables 
% scken, 2021
% Copyright (C) 2021 Chair of Automation Technology / TU Chemnitz

clear all 
close all

n_dim = [512 1024 2048];
scale = [2 4 6 8];
encoding_dim = [20 40 60 80 100];
HDC_network = 'True';
dataset = 'full';

f1_tensor = zeros([3 3 3]);

%% iterate over all results of hyper-parameter analysis (table 3)

for d=1:numel(n_dim)
    for s=1:numel(scale)
        for e=1:numel(encoding_dim)
            % load file
            load(['results/' dataset '/results_VSA_' num2str(n_dim(d)) '_' num2str(scale(s)) '_' num2str(encoding_dim(e)) '_1.0.mat'])
            
            f1_tensor(d,s,e) = round(report.accuracy,2);
        end
    end
end

% create result table
subtable = {};

% create Rownames
rownames = {};
for s=1:numel(scale) 
    rownames{end +1} = ['scale = ' num2str(scale(s))];
end

% variable names 
varnames = {}; 
for e=1:numel(encoding_dim)
    varnames{end+1} = [num2str(encoding_dim(e))];
end

for d=1:numel(n_dim)
    subtable{d} = array2table(squeeze(f1_tensor(d,:,:)), ...
        'VariableNames',varnames);
end

tab = table(subtable{:},'VariableNames',{'\# Dimensions = 512', '\# Dimensions = 1024', '\# Dimensions = 2048'},'RowNames',rownames)
table2latex(tab,[3 2],0.5,'tables/results_vsa.tex')

%% evaluate model size (table 4)

param_array = zeros([numel(n_dim) numel(encoding_dim)]);

for d=1:numel(n_dim)
    for e=1:numel(encoding_dim)
        % load file
        load(['results/' dataset '/results_VSA_' num2str(n_dim(d)) '_8_' num2str(encoding_dim(e)) '_1.0.mat'])
        start_idx = findstr(model_summary,'Trainable params: ');
        start_idx = start_idx + 18;
        end_idx = findstr(model_summary,'Non-trainable');
        n_params = str2num(replace(model_summary(start_idx:end_idx-1),',',''));
        
        param_array(d,e) = n_params;
    end
end


%% plot different scales of fractional binding (fig. 2)

vsa = 'FHRR';
dim = 2048;

VSA = vsa_env('vsa',vsa,'dim',dim);
init_vector = VSA.add_vector();
values = -1.5:0.01:1.5;
line_type = {':';'-';'--';'-.'};

sim_values = zeros([numel(scale) numel(values)]);

leg = {};

for s=1:numel(scale)
    encoded_values = VSA.frac_binding(init_vector,values * scale(s));
    ref_value = VSA.frac_binding(init_vector,0);
    sim_values(s,:) = VSA.sim(ref_value, encoded_values);
    
    plot(values,sim_values(s,:),line_type{s})
    hold on 
    leg{end+1} = ['scaling = ' num2str(scale(s))];
end

grid on 
title('Similarity of encoded scalar value 0 to neighboring values')
xlabel('scalar value')
ylabel('similarity to encoded value 0')
legend(leg)
set(gcf,'color','w')
saveas(gcf,'images/similarity_plot_frac_binding.png')
export_fig('images/similarity_plot_frac_binding.pdf','-dpdf') 


%% plot data efficiency (fig. 6)

figure()
n_dim = 2048;
scale = 10;
encoding_dim = 40;
dataset = 'full';
training_volume = {'0.2'; '0.4'; '0.6'; '0.8'; '1.0'};
f1_array_orig = [];
f1_array_VSA = [];

% original network 
for t=1:numel(training_volume)
    % load file
    load(['results/' dataset '/results_origNet_' training_volume{t} '.mat'])

    f1_array_orig(end+1) = report.accuracy;
end

% VSA network 
for t=1:numel(training_volume)
    % load file
    load(['results/' dataset '/results_VSA_' num2str(n_dim) '_' num2str(scale) '_' num2str(encoding_dim) '_' training_volume{t} '.mat'])

    f1_array_VSA(end+1) = report.accuracy;
end

plot(str2num(cell2mat(training_volume)), f1_array_orig,'--','LineWidth',2)
hold on 
plot(str2num(cell2mat(training_volume)), f1_array_VSA,'-.','LineWidth',2)
grid on 
xlabel('training volume')
ylabel('F_1 Score')
title('Data efficiency')
set(gcf,'color','w')
set(gcf,'Position',[100 100 500 200])
legend({'LSTM-ANN'; 'HDC-ANN'},'Location','SouthEast')
saveas(gcf,'images/data_efficiency.png')
export_fig('images/data_efficiency.pdf','-dpdf') 




