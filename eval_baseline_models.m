%% classify all datasets with all baseline models 
% scken, 2021
% Copyright (C) 2021 Chair of Automation Technology / TU Chemnitz

clear all
close all

dim = 576;
frac_scale = 6;


%% secondary 
if 1
    dataset = 'secondary';
    Baseline_Models
end

%% motorway
if 1
    dataset = 'motorway';
    Baseline_Models
end

%% full
if 1
    dataset = 'full';
    Baseline_Models
end

%% full dataset cross validation 
if 1
    dataset = 'full_crossval';
    Baseline_Models
end

