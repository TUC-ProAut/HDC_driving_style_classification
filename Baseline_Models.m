%% simple baseline predicition models
% scken, 2021
% Copyright (C) 2021 Chair of Automation Technology / TU Chemnitz


% parameter setup
dim = 576;
frac_scale = 6;

disp('----------------------------')
disp(['Baseline ' dataset])
disp(['Dim: ' num2str(dim)])


%% if dataset = full_crossval, create the training and test splits

if contains(dataset,'full_crossval')
    %%% HDC encoding with kNN classifier 
    
    % load the data with the python script 
    ret = system(['python3 create_train_test_split_MATLAB.py --dataset=' dataset ' --preproc=1 --input_dim=' num2str(dim) ' --scale=' num2str(frac_scale)]);
    if ret==0
        load('temp_data.mat')
    else
        disp('Data could not converted')
        return
    end
    delete('temp_data.mat')
    
    for i=1:size(X_train,2)
        %%%
        % HDC with k-NN

        % load data into item memory
        VSA = vsa_env('vsa','FHRR','dim',dim);
        VSA.add_vector('vec',X_train{i}','name',num2cell(num2str(Y_train{i})));

        % find k nearest neigbors 
        tic
        [~, l, s] = VSA.find_k_nearest(X_test{i}',3);
        pred = [];

        for c=1:size(l,2)
            temp = str2num(cell2mat(l(:,c)));
            pred(end+1) = mode(temp);
        end
        disp('Time for testing k-NN:')
        toc
        disp('Accuracy of HDC k-NN method: ')
        f1 = getF1Score(Y_test{i},pred);
        disp(f1)

    end

    %%%
    %  spectral features (FFT) with kNN

    ret = system(['python3 create_train_test_split_MATLAB.py --dataset=' dataset ' --preproc=0 --input_dim=' num2str(dim) ' --scale=' num2str(frac_scale)]);

    if ret==0
        load('temp_data.mat')
    else
        disp('Data could not converted')
        return
    end
    delete('temp_data.mat')

    for i=1:size(X_train,2)
        % fourier transformation
        X_train{i} = abs(fft(X_train{i},size(X_train{i},2),2));
        X_test{i} = abs(fft(X_test{i},size(X_test{i},2),2));

        % concat input
        X_train{i} = reshape(X_train{i},size(X_train{i},1),[]);
        X_test{i} = reshape(X_test{i},size(X_test{i},1),[]);

        Mdl = fitcknn(X_train{i},Y_train{i},'NumNeighbors',1,'Distance','Cityblock');

        % testing
        pred = predict(Mdl, X_test{i});

        disp('Accuracy of Spectral Features kNN method: ')
        f1 = getF1Score(Y_test{i},pred);
        disp(f1)

    end
end

%% HDC with SVM

ret = system(['python3 create_train_test_split_MATLAB.py --dataset=' dataset ' --preproc=1 --input_dim=' num2str(dim) ' --scale=' num2str(frac_scale)]);

if ret==0
    load('temp_data.mat')
else
    disp('Data could not converted')
    return
end
delete('temp_data.mat')

tic
Mdl = fitcecoc(X_train,Y_train);
disp('Time for training HDC-SVM:')
toc

% testing
tic
pred = predict(Mdl, X_test);
disp('Time for testing HDC-SVM:')
toc
f1 = getF1Score(Y_test,pred);
disp('Accuracy of HDC SVM:')
disp(f1)

% add result to table
Result = table({'HDC-SVM'},f1,'VariableNames',{'Model','F1'});

%% HDC with k-NN

% load data into item memory
VSA = vsa_env('vsa','FHRR','dim',dim);
VSA.add_vector('vec',X_train','name',num2cell(num2str(Y_train)));

% find k nearest neigbors
tic
[~, l, s] = VSA.find_k_nearest(X_test',3);
pred = [];

for c=1:size(l,2)
    temp = str2num(cell2mat(l(:,c)));
    pred(end+1) = mode(temp);
end
disp('Time for testing k-NN:')
toc
disp('Accuracy of HDC k-NN method: ')
f1 = getF1Score(Y_test,pred);
disp(f1)

% add to table
% Result = table({'HDC-kNN'},acc,'VariableNames',{'Model','F1'});
Result.Model{end+1} = 'HDC-kNN';
Result.F1(end) = f1;

%% concat with SVM

ret = system(['python3 create_train_test_split_MATLAB.py --dataset=' dataset ' --preproc=0 --input_dim=' num2str(dim) ' --scale=' num2str(frac_scale)]);

if ret==0
    load('temp_data.mat')
else
    disp('Data could not converted')
    return
end
delete('temp_data.mat')

% concat input 
X_train = reshape(X_train,size(X_train,1),[]);
X_test = reshape(X_test,size(X_test,1),[]);


Mdl = fitcecoc(X_train,Y_train);

% testing 
pred = predict(Mdl, X_test);
f1 = getF1Score(Y_test,pred);
disp('Accuracy of Concat SVM method: ')
disp(f1)

% add to table
Result.Model{end+1} = 'Concat-SVM';
Result.F1(end) = f1;


%% concat with kNN 

% find optimal hyperparameter for concat model
% rng(0)
% Mdl_opt = fitcknn([X_train; X_test],[Y_train; Y_test],'OptimizeHyperparameters','auto',...
%     'HyperparameterOptimizationOptions',...
%     struct('AcquisitionFunctionName','expected-improvement-plus'))
% 
% Mdl = fitcknn(X_train,Y_train,'NumNeighbors',Mdl_opt.NumNeighbors,'Distance',Mdl_opt.Distance);

Mdl = fitcknn(X_train,Y_train,'NumNeighbors',3,'Distance','Cityblock');

% testing 
pred = predict(Mdl, X_test);
f1 = getF1Score(Y_test,pred);
disp('Accuracy of Concat k-NN method: ')
disp(f1)

% add to table
Result.Model{end+1} = 'Concat-kNN';
Result.F1(end) = f1;

%%  spectral features (FFT) with SVM 

ret = system(['python3 create_train_test_split_MATLAB.py --dataset=' dataset ' --preproc=0 --input_dim=' num2str(dim) ' --scale=' num2str(frac_scale)]);

if ret==0
    load('temp_data.mat')
else
    disp('Data could not converted')
    return
end
delete('temp_data.mat')

% fourier transformation
X_train = abs(fft(X_train,size(X_train,2),2));
X_test = abs(fft(X_test,size(X_test,2),2));

% concat input
X_train = reshape(X_train,size(X_train,1),[]);
X_test = reshape(X_test,size(X_test,1),[]);

tic
% Mdl = fitcecoc(X_train,Y_train,'Learners',svm_template);
Mdl = fitcecoc(X_train,Y_train);
disp('Time for training SVM-Stat:')
toc

% testing 
tic
pred = predict(Mdl, X_test);
disp('Time for testing SVM-Stat:')
toc 

disp('Accuracy of Spectral Features SVM method: ')
f1 = getF1Score(Y_test,pred);
disp(f1)

% add to table
Result.Model{end+1} = 'Spect-SVM';
Result.F1(end) = f1;

%%%
%  spectral features (FFT) with kNN

ret = system(['python3 create_train_test_split_MATLAB.py --dataset=' dataset ' --preproc=0 --input_dim=' num2str(dim) ' --scale=' num2str(frac_scale)]);

if ret==0
    load('temp_data.mat')
else
    disp('Data could not converted')
    return
end
delete('temp_data.mat')

% fourier transformation
X_train = abs(fft(X_train,size(X_train,2),2));
X_test = abs(fft(X_test,size(X_test,2),2));

% concat input
X_train = reshape(X_train,size(X_train,1),[]);
X_test = reshape(X_test,size(X_test,1),[]);


% find optimal hyperparameter for concat model
% rng(0)
% Mdl_opt = fitcknn([X_train; X_test],[Y_train; Y_test],'OptimizeHyperparameters','auto',...
%     'HyperparameterOptimizationOptions',...
%     struct('AcquisitionFunctionName','expected-improvement-plus'))
% 
% Mdl = fitcknn(X_train,Y_train,'NumNeighbors',Mdl_opt.NumNeighbors,'Distance',Mdl_opt.Distance);

Mdl = fitcknn(X_train,Y_train,'NumNeighbors',1,'Distance','Cityblock');

% testing 
pred = predict(Mdl, X_test);

disp('Accuracy of Spectral Features kNN method: ')
f1 = getF1Score(Y_test,pred);
disp(f1)

% add to table
Result.Model{end+1} = 'Spect-kNN';
Result.F1(end) = f1;

%% print results

disp([dataset ' Dataset:'])
disp(Result)

