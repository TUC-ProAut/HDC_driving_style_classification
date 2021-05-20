# Data Description
The training/testing data has the shape (N x T x D) with labels of shape (N x C):
- N: The total number of the training/testing samples
- T: The times steps (64)
- D: The dimension of the data (9), which corresponds to [X_ACC, Y_ACC, Z_ACC, ROLL, PITCH, YAW, distance ahead, NoV, speed]
- C: The class of driving behaviour (1) [0-> normal, 1-> aggressive, 2-> drowsy]

## Files 

- .pkl files are the original preprocessed data sets of [1]
- .mat files are the same data as in .pkl but in Matlab file format  

[1] K. Saleh, M. Hossny, and S. Nahavandi, “Driving behavior classification based on sensor data fusion using LSTM recurrent neural networks,” in International Conference on Intelligent Transportation Systems (ITSC), 2017.
