# VSA for driving behaviour classification

This repository is mainly based on the code of https://github.com/KhaledSaleh/driving_behaviour_classification

It has 3 different models:
- a LSTM model (original model from [1])
- a feed-forward model (ANN) for HDC encodings
- a spiking neural model (SNN) for HDC encodings
 
[1] K. Saleh, M. Hossny, and S. Nahavandi, “Driving behavior classification based on sensor data fusion using LSTM recurrent neural networks,” in International Conference on Intelligent Transportation Systems (ITSC), 2017.

Tested with the Python packages listed in requirements.txt.

## Usage 
* first, clone the Repo `git clone https://github.com/TUC-ProAut/HDC_driving_style_classification.git`

### Encode the data set into high-dimensional vectors
1. run the `HDC_encoding.m` script:
    * default parameters: 2048 dimensions, scaling of 6 and using cpu computing
    * with the parameter *mode* you can choose between three different options:
        * 0... use the CPU (default, no VSA_toolbox is required)
        * 1... use the GPU
        * 2... use the original script with the VSA_toolbox from https://github.com/TUC-ProAut/VSA_Toolbox
            * using the VSA_toolbox can provide a better understanding of the code, but requires more computing time
            * to use the VSA_toolbox clone the repo `git clone https://github.com/TUC-ProAut/VSA_Toolbox` and add to MATLAB path `addpath('/path/to/VSA_toolbox')`
2. (Optional) run the `HDC_encoding_for_hyperparam_analysis.m` to create all encodings for the complete hyper-parameter analysis (different number of dimenions, different scaling)

### Train the networks (Python)
1. Run `python3 main.py --help` to check the available command line args.
2. Run ANN with HDC encodings:
    * `python3 main.py --HDC_ANN True` (use --dataset argument to select between full, motorway, secondary or full_crossval) 
3. Run ANN with concatenated input sequences:
    * `python3 main.py --Concat_ANN True`
4. Run the original LSTM model from https://github.com/KhaledSaleh/driving_behaviour_classification
    * `python3 main.py --LSTM True`
5. Run SNN with HDC encodings:
    * `python3 main.py --HDC_SNN True`

The results are written to the log file main_log.log

### Data efficiency experiment (Python)
1. Run `python3 main.py --data_efficiency True --HDC_ANN True` for the appropriate network as in section above 

### (Optional) Hyper-parameter analysis for HDC encodings (Python)
1. Run `python3 main.py --hyperparams_experiment True --HDC_ANN True` 

### Run Baseline models (MATLAB)
1. Run `eval_baseline_models.m`

