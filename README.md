# Experiments with different optimizers and dropout on CNN

#### Starting params

- epochs = 10

### - Experiments with optimizers:

### SGD
Starting learning rate = 0.01
- Epoch [1/10], Loss: 0.7124
- Epoch [2/10], Loss: 0.1863
- Epoch [3/10], Loss: 0.1176
- Epoch [4/10], Loss: 0.0891
- Epoch [5/10], Loss: 0.0734
- Epoch [6/10], Loss: 0.0630
- Epoch [7/10], Loss: 0.0558
- Epoch [8/10], Loss: 0.0498
- Epoch [9/10], Loss: 0.0448
- Epoch [10/10], Loss: 0.0409
Accuracy: 98.36%

Starting learning rate = 0.5
- Epoch [1/10], Loss: 0.3137
- Epoch [2/10], Loss: 0.0760
- Epoch [3/10], Loss: 0.0528
- Epoch [4/10], Loss: 0.0437
- Epoch [5/10], Loss: 0.0374
- Epoch [6/10], Loss: 0.0347
- Epoch [7/10], Loss: 0.0302
- Epoch [8/10], Loss: 0.0354
- Epoch [9/10], Loss: 0.0275
- Epoch [10/10], Loss: 0.0313
Accuracy: 97.64%

### AdaGrad
Starting learning rate = 0.01
- Epoch [1/10], Loss: 0.1485
- Epoch [2/10], Loss: 0.0579
- Epoch [3/10], Loss: 0.0449
- Epoch [4/10], Loss: 0.0380
- Epoch [5/10], Loss: 0.0332
- Epoch [6/10], Loss: 0.0296
- Epoch [7/10], Loss: 0.0263
- Epoch [8/10], Loss: 0.0240
- Epoch [9/10], Loss: 0.0218
- Epoch [10/10], Loss: 0.0200
Accuracy: 98.88%

Starting learning rate = 0.5
- Epoch [1/10], Loss: 99.1315
- Epoch [2/10], Loss: 0.3188
- Epoch [3/10], Loss: 0.2202
- Epoch [4/10], Loss: 0.1834
- Epoch [5/10], Loss: 0.1585
- Epoch [6/10], Loss: 0.1419
- Epoch [7/10], Loss: 0.1281
- Epoch [8/10], Loss: 0.1164
- Epoch [9/10], Loss: 0.1077
- Epoch [10/10], Loss: 0.0996
Accuracy: 96.23%

### Adam
Starting learning rate = 0.01
- Epoch [1/10], Loss: 0.1826
- Epoch [2/10], Loss: 0.0976
- Epoch [3/10], Loss: 0.0888
- Epoch [4/10], Loss: 0.0845
- Epoch [5/10], Loss: 0.0781
- Epoch [6/10], Loss: 0.0786
- Epoch [7/10], Loss: 0.0784
- Epoch [8/10], Loss: 0.0750
- Epoch [9/10], Loss: 0.0742
- Epoch [10/10], Loss: 0.0705
Accuracy: 97.67%

Starting learning rate = 0.5
- Epoch [1/10], Loss: 105.1085
- Epoch [2/10], Loss: 2.3606
- Epoch [3/10], Loss: 2.3618
- Epoch [4/10], Loss: 2.3599
- Epoch [5/10], Loss: 2.3624
- Epoch [6/10], Loss: 2.3580
- Epoch [7/10], Loss: 2.3602
- Epoch [8/10], Loss: 2.3604
- Epoch [9/10], Loss: 2.3564
- Epoch [10/10], Loss: 2.3577
Accuracy: 11.35%

### - Experiments with Dropout:

### SGD
Starting learning rate = 0.01
- Epoch [1/10], Loss: 0.7665
- Epoch [2/10], Loss: 0.2385
- Epoch [3/10], Loss: 0.1552
- Epoch [4/10], Loss: 0.1175
- Epoch [5/10], Loss: 0.0961
- Epoch [6/10], Loss: 0.0828
- Epoch [7/10], Loss: 0.0701
- Epoch [8/10], Loss: 0.0638
- Epoch [9/10], Loss: 0.0580
- Epoch [10/10], Loss: 0.0530
Accuracy: 97.74%

### Adam
Starting learning rate = 0.01
- Epoch [1/10], Loss: 0.3420
- Epoch [2/10], Loss: 0.2086
- Epoch [3/10], Loss: 0.2006
- Epoch [4/10], Loss: 0.1893
- Epoch [5/10], Loss: 0.1880
- Epoch [6/10], Loss: 0.1878
- Epoch [7/10], Loss: 0.1827
- Epoch [8/10], Loss: 0.1841
- Epoch [9/10], Loss: 0.1867
- Epoch [10/10], Loss: 0.1820
Accuracy: 95.53%
