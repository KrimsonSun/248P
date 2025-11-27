# DNN 

    1 . update 

       l_rate = 0.05
        n_epoch = 200

    2. not usin random but:
    for n_hidden in hidden_layers:
        # 
        limit = 1.0 / (prev_neurons ** 0.5) # Xavier/Glorot 
        
        layer = [{'weights': [random.uniform(-limit, limit) for _ in range(prev_neurons + 1)]} 
                 for _ in range(n_hidden)]
        network.append(layer)
        prev_neurons = n_hidden

    # 2. Create the Output Layer
    #
    limit = 1.0 / (prev_neurons ** 0.5)
    output_layer = [{'weights': [random.uniform(-limit, limit) for _ in range(prev_neurons + 1)]} 
                    for _ in range(n_outputs)]

    3. with hyperparameter setting, result with different layer 

    (50, 25, 10, 5):Starting training with rate=0.02, epochs=400...
>Epoch=10, lrate=0.020, error=139.58523
>Epoch=20, lrate=0.020, error=139.59937
>Epoch=30, lrate=0.020, error=139.60297
>Epoch=40, lrate=0.020, error=139.60652
>Epoch=50, lrate=0.020, error=139.61007
>Epoch=60, lrate=0.020, error=139.61362
>Epoch=70, lrate=0.020, error=139.61717
>Epoch=80, lrate=0.020, error=139.62072
>Epoch=90, lrate=0.020, error=139.62427
>Epoch=100, lrate=0.020, error=139.62782
>Epoch=110, lrate=0.020, error=139.63136
>Epoch=120, lrate=0.020, error=139.63491
>Epoch=130, lrate=0.020, error=139.63844
>Epoch=140, lrate=0.020, error=139.64198
>Epoch=150, lrate=0.020, error=139.64550
>Epoch=160, lrate=0.020, error=139.64901
>Epoch=170, lrate=0.020, error=139.65250
>Epoch=180, lrate=0.020, error=139.65598
>Epoch=190, lrate=0.020, error=139.65943
>Epoch=200, lrate=0.020, error=139.66287
>Epoch=210, lrate=0.020, error=139.66628
>Epoch=220, lrate=0.020, error=139.66966
>Epoch=230, lrate=0.020, error=139.67301
>Epoch=240, lrate=0.020, error=139.67633
>Epoch=250, lrate=0.020, error=139.67962
>Epoch=260, lrate=0.020, error=139.68287
>Epoch=270, lrate=0.020, error=139.68608
>Epoch=280, lrate=0.020, error=139.68925
>Epoch=290, lrate=0.020, error=139.69238
>Epoch=300, lrate=0.020, error=139.69547
>Epoch=310, lrate=0.020, error=139.69852
>Epoch=320, lrate=0.020, error=139.70152
>Epoch=330, lrate=0.020, error=139.70449
>Epoch=340, lrate=0.020, error=139.70741
>Epoch=350, lrate=0.020, error=139.71029
>Epoch=360, lrate=0.020, error=139.71312
>Epoch=370, lrate=0.020, error=139.71592
>Epoch=380, lrate=0.020, error=139.71868
>Epoch=390, lrate=0.020, error=139.72140
>Epoch=400, lrate=0.020, error=139.72409

--- Testing Predictions ---

Classification Complete. Total Accuracy: 33.33%

    (50, 25, 10)
    Starting training with rate=0.02, epochs=400...
>Epoch=10, lrate=0.020, error=139.06289
>Epoch=20, lrate=0.020, error=139.07589
>Epoch=30, lrate=0.020, error=139.08290
>Epoch=40, lrate=0.020, error=139.08901
>Epoch=50, lrate=0.020, error=139.09430
>Epoch=60, lrate=0.020, error=139.09888
>Epoch=70, lrate=0.020, error=139.10287
>Epoch=80, lrate=0.020, error=139.10644
>Epoch=90, lrate=0.020, error=139.10977
>Epoch=100, lrate=0.020, error=139.11305
>Epoch=110, lrate=0.020, error=139.11651
>Epoch=120, lrate=0.020, error=139.12041
>Epoch=130, lrate=0.020, error=139.12502
>Epoch=140, lrate=0.020, error=139.13063
>Epoch=150, lrate=0.020, error=139.13755
>Epoch=160, lrate=0.020, error=139.14606
>Epoch=170, lrate=0.020, error=139.15644
>Epoch=180, lrate=0.020, error=139.16895
>Epoch=190, lrate=0.020, error=139.18376
>Epoch=200, lrate=0.020, error=139.20100
>Epoch=210, lrate=0.020, error=139.22071
>Epoch=220, lrate=0.020, error=139.24282
>Epoch=230, lrate=0.020, error=139.26718
>Epoch=240, lrate=0.020, error=139.29353
>Epoch=250, lrate=0.020, error=139.32149
>Epoch=260, lrate=0.020, error=139.35065
>Epoch=270, lrate=0.020, error=139.38050
>Epoch=280, lrate=0.020, error=139.41052
>Epoch=290, lrate=0.020, error=139.44018
>Epoch=300, lrate=0.020, error=139.46897
>Epoch=310, lrate=0.020, error=139.49642
>Epoch=320, lrate=0.020, error=139.52214
>Epoch=330, lrate=0.020, error=139.54582
>Epoch=340, lrate=0.020, error=139.56725
>Epoch=350, lrate=0.020, error=139.58631
>Epoch=360, lrate=0.020, error=139.60297
>Epoch=370, lrate=0.020, error=139.61727
>Epoch=380, lrate=0.020, error=139.62936
>Epoch=390, lrate=0.020, error=139.63941
>Epoch=400, lrate=0.020, error=139.64764

--- Testing Predictions ---

Classification Complete. Total Accuracy: 33.33%



    (5,) has the best performance , to improve its performance, 
    

    result came out:
    Network initialized with 2 layers (Input:7, Hidden:(5,), Output:3)
Starting training with rate=0.02, epochs=400...
>Epoch=10, lrate=0.020, error=137.35673
>Epoch=20, lrate=0.020, error=132.21564
>Epoch=30, lrate=0.020, error=122.20982
>Epoch=40, lrate=0.020, error=107.50195
>Epoch=50, lrate=0.020, error=94.45478
>Epoch=60, lrate=0.020, error=85.90013
>Epoch=70, lrate=0.020, error=80.43524
>Epoch=80, lrate=0.020, error=76.61684
>Epoch=90, lrate=0.020, error=73.61779
>Epoch=100, lrate=0.020, error=70.97242
>Epoch=110, lrate=0.020, error=68.39763
>Epoch=120, lrate=0.020, error=65.71720
>Epoch=130, lrate=0.020, error=62.83807
>Epoch=140, lrate=0.020, error=59.74805
>Epoch=150, lrate=0.020, error=56.51361
>Epoch=160, lrate=0.020, error=53.25897
>Epoch=170, lrate=0.020, error=50.12428
>Epoch=180, lrate=0.020, error=47.22231
>Epoch=190, lrate=0.020, error=44.61596
>Epoch=200, lrate=0.020, error=42.32059
>Epoch=210, lrate=0.020, error=40.31944
>Epoch=220, lrate=0.020, error=38.57987
>Epoch=230, lrate=0.020, error=37.06475
>Epoch=240, lrate=0.020, error=35.73863
>Epoch=250, lrate=0.020, error=34.57038
>Epoch=260, lrate=0.020, error=33.53376
>Epoch=270, lrate=0.020, error=32.60718
>Epoch=280, lrate=0.020, error=31.77301
>Epoch=290, lrate=0.020, error=31.01693
>Epoch=300, lrate=0.020, error=30.32729
>Epoch=310, lrate=0.020, error=29.69458
>Epoch=320, lrate=0.020, error=29.11101
>Epoch=330, lrate=0.020, error=28.57016
>Epoch=340, lrate=0.020, error=28.06669
>Epoch=350, lrate=0.020, error=27.59617
>Epoch=360, lrate=0.020, error=27.15487
>Epoch=370, lrate=0.020, error=26.73963
>Epoch=380, lrate=0.020, error=26.34777
>Epoch=390, lrate=0.020, error=25.97699
>Epoch=400, lrate=0.020, error=25.62534

--- Testing Predictions ---

Classification Complete. Total Accuracy: 93.33%