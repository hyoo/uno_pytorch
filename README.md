# PyTorch implementation of UNO model
The original model was written in Keras, which can be found at https://github.com/ECP-CANDLE/Benchmarks/tree/master/Pilot1/Uno

## Regression Model
```
$ python uno_pytorch.py --device=cuda -z 512 -lr 4e-4 --mode reg
==========================================================================================
Layer (type:depth-idx)                   Output Shape              Param #
==========================================================================================
UnoModel                                 --                        --
├─FeatureModel: 1-1                      [512, 1000]               --
│    └─Sequential: 2-1                   [512, 1000]               --
│    │    └─Linear: 3-1                  [512, 1000]               943,000
│    │    └─ReLU: 3-2                    [512, 1000]               --
│    │    └─Dropout: 3-3                 [512, 1000]               --
│    │    └─Linear: 3-4                  [512, 1000]               1,001,000
│    │    └─ReLU: 3-5                    [512, 1000]               --
│    │    └─Dropout: 3-6                 [512, 1000]               --
│    │    └─Linear: 3-7                  [512, 1000]               1,001,000
│    │    └─ReLU: 3-8                    [512, 1000]               --
│    │    └─Dropout: 3-9                 [512, 1000]               --
├─FeatureModel: 1-2                      [512, 1000]               --
│    └─Sequential: 2-2                   [512, 1000]               --
│    │    └─Linear: 3-10                 [512, 1000]               5,271,000
│    │    └─ReLU: 3-11                   [512, 1000]               --
│    │    └─Dropout: 3-12                [512, 1000]               --
│    │    └─Linear: 3-13                 [512, 1000]               1,001,000
│    │    └─ReLU: 3-14                   [512, 1000]               --
│    │    └─Dropout: 3-15                [512, 1000]               --
│    │    └─Linear: 3-16                 [512, 1000]               1,001,000
│    │    └─ReLU: 3-17                   [512, 1000]               --
│    │    └─Dropout: 3-18                [512, 1000]               --
├─Sequential: 1-3                        [512, 1]                  --
│    └─Linear: 2-3                       [512, 1000]               2,001,000
│    └─ReLU: 2-4                         [512, 1000]               --
│    └─Dropout: 2-5                      [512, 1000]               --
│    └─Linear: 2-6                       [512, 1000]               1,001,000
│    └─ReLU: 2-7                         [512, 1000]               --
│    └─Dropout: 2-8                      [512, 1000]               --
│    └─Linear: 2-9                       [512, 1000]               1,001,000
│    └─ReLU: 2-10                        [512, 1000]               --
│    └─Dropout: 2-11                     [512, 1000]               --
│    └─Linear: 2-12                      [512, 1000]               1,001,000
│    └─ReLU: 2-13                        [512, 1000]               --
│    └─Dropout: 2-14                     [512, 1000]               --
│    └─Linear: 2-15                      [512, 1000]               1,001,000
│    └─ReLU: 2-16                        [512, 1000]               --
│    └─Dropout: 2-17                     [512, 1000]               --
│    └─Linear: 2-18                      [512, 1]                  1,001
==========================================================================================
Total params: 16,224,001
Trainable params: 16,224,001
Non-trainable params: 0
Total mult-adds (G): 8.31
==========================================================================================
Input size (MB): 12.72
Forward/backward pass size (MB): 45.06
Params size (MB): 64.90
Estimated Total Size (MB): 122.68
==========================================================================================
Loading script and data:  10.43
Epoch: 001, elasped: 8.13 sec(s), lr: 0.000400, loss: 0.02190, val_loss: 0.0139
Epoch: 002, elasped: 8.15 sec(s), lr: 0.000400, loss: 0.01960, val_loss: 0.0092
Epoch: 003, elasped: 8.02 sec(s), lr: 0.000400, loss: 0.01738, val_loss: 0.0085
Epoch: 004, elasped: 7.84 sec(s), lr: 0.000400, loss: 0.01842, val_loss: 0.0077
Epoch: 005, elasped: 8.14 sec(s), lr: 0.000400, loss: 0.01663, val_loss: 0.0070
Epoch: 006, elasped: 7.88 sec(s), lr: 0.000400, loss: 0.01319, val_loss: 0.0065
Epoch: 007, elasped: 7.99 sec(s), lr: 0.000400, loss: 0.01065, val_loss: 0.0072
Epoch: 008, elasped: 8.04 sec(s), lr: 0.000400, loss: 0.01285, val_loss: 0.0070
Epoch: 009, elasped: 8.30 sec(s), lr: 0.000400, loss: 0.01865, val_loss: 0.0069
Epoch: 010, elasped: 8.19 sec(s), lr: 0.000400, loss: 0.01024, val_loss: 0.0059
```


## Classification Model
```
$ python uno_pytorch.py --device=cuda -z 512 -lr 4e-4 --mode cls
==========================================================================================
Layer (type:depth-idx)                   Output Shape              Param #
==========================================================================================
UnoModel                                 --                        --
├─FeatureModel: 1-1                      [512, 1000]               --
│    └─Sequential: 2-1                   [512, 1000]               --
│    │    └─Linear: 3-1                  [512, 1000]               943,000
│    │    └─ReLU: 3-2                    [512, 1000]               --
│    │    └─Dropout: 3-3                 [512, 1000]               --
│    │    └─Linear: 3-4                  [512, 1000]               1,001,000
│    │    └─ReLU: 3-5                    [512, 1000]               --
│    │    └─Dropout: 3-6                 [512, 1000]               --
│    │    └─Linear: 3-7                  [512, 1000]               1,001,000
│    │    └─ReLU: 3-8                    [512, 1000]               --
│    │    └─Dropout: 3-9                 [512, 1000]               --
├─FeatureModel: 1-2                      [512, 1000]               --
│    └─Sequential: 2-2                   [512, 1000]               --
│    │    └─Linear: 3-10                 [512, 1000]               5,271,000
│    │    └─ReLU: 3-11                   [512, 1000]               --
│    │    └─Dropout: 3-12                [512, 1000]               --
│    │    └─Linear: 3-13                 [512, 1000]               1,001,000
│    │    └─ReLU: 3-14                   [512, 1000]               --
│    │    └─Dropout: 3-15                [512, 1000]               --
│    │    └─Linear: 3-16                 [512, 1000]               1,001,000
│    │    └─ReLU: 3-17                   [512, 1000]               --
│    │    └─Dropout: 3-18                [512, 1000]               --
├─Sequential: 1-3                        [512, 2]                  --
│    └─Linear: 2-3                       [512, 1000]               2,001,000
│    └─ReLU: 2-4                         [512, 1000]               --
│    └─Dropout: 2-5                      [512, 1000]               --
│    └─Linear: 2-6                       [512, 1000]               1,001,000
│    └─ReLU: 2-7                         [512, 1000]               --
│    └─Dropout: 2-8                      [512, 1000]               --
│    └─Linear: 2-9                       [512, 1000]               1,001,000
│    └─ReLU: 2-10                        [512, 1000]               --
│    └─Dropout: 2-11                     [512, 1000]               --
│    └─Linear: 2-12                      [512, 1000]               1,001,000
│    └─ReLU: 2-13                        [512, 1000]               --
│    └─Dropout: 2-14                     [512, 1000]               --
│    └─Linear: 2-15                      [512, 1000]               1,001,000
│    └─ReLU: 2-16                        [512, 1000]               --
│    └─Dropout: 2-17                     [512, 1000]               --
│    └─Linear: 2-18                      [512, 2]                  2,002
==========================================================================================
Total params: 16,225,002
Trainable params: 16,225,002
Non-trainable params: 0
Total mult-adds (G): 8.31
==========================================================================================
Input size (MB): 12.72
Forward/backward pass size (MB): 45.06
Params size (MB): 64.90
Estimated Total Size (MB): 122.69
==========================================================================================
Loading script and data:  12.50
Epoch: 001, elapsed: 9.10 sec(s), lr: 0.000400, loss: 0.23529, val_loss: 0.2261, accuracy: 0.8915, precision neg: 1.0000, pos: 0.1515, recall neg: 0.8893, pos: 1.0000
Epoch: 002, elapsed: 7.85 sec(s), lr: 0.000400, loss: 0.16545, val_loss: 0.1827, accuracy: 0.9109, precision neg: 1.0000, pos: 0.1786, recall neg: 0.9091, pos: 1.0000
Epoch: 003, elapsed: 7.83 sec(s), lr: 0.000400, loss: 0.08014, val_loss: 0.1941, accuracy: 0.9225, precision neg: 1.0000, pos: 0.2000, recall neg: 0.9209, pos: 1.0000
Epoch: 004, elapsed: 8.40 sec(s), lr: 0.000400, loss: 0.09752, val_loss: 0.1753, accuracy: 0.9147, precision neg: 1.0000, pos: 0.1852, recall neg: 0.9130, pos: 1.0000
Epoch: 005, elapsed: 7.90 sec(s), lr: 0.000400, loss: 0.11091, val_loss: 0.1704, accuracy: 0.9341, precision neg: 1.0000, pos: 0.2273, recall neg: 0.9328, pos: 1.0000
Epoch: 006, elapsed: 8.00 sec(s), lr: 0.000400, loss: 0.09788, val_loss: 0.1513, accuracy: 0.9419, precision neg: 1.0000, pos: 0.2500, recall neg: 0.9407, pos: 1.0000
Epoch: 007, elapsed: 8.03 sec(s), lr: 0.000400, loss: 0.21374, val_loss: 0.1466, accuracy: 0.9380, precision neg: 0.9958, pos: 0.2105, recall neg: 0.9407, pos: 0.8000
Epoch: 008, elapsed: 7.87 sec(s), lr: 0.000400, loss: 0.07795, val_loss: 0.1479, accuracy: 0.9380, precision neg: 1.0000, pos: 0.2381, recall neg: 0.9368, pos: 1.0000
Epoch: 009, elapsed: 7.83 sec(s), lr: 0.000400, loss: 0.07676, val_loss: 0.1605, accuracy: 0.9419, precision neg: 1.0000, pos: 0.2500, recall neg: 0.9407, pos: 1.0000
Epoch: 010, elapsed: 8.11 sec(s), lr: 0.000400, loss: 0.02915, val_loss: 0.1571, accuracy: 0.9496, precision neg: 1.0000, pos: 0.2778, recall neg: 0.9486, pos: 1.0000
```