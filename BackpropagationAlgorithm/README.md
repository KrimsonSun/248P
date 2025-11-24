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

    (50, 25, 10, 5):>Epoch=10, lrate=0.010, error=139.89004
>   Epoch=20, lrate=0.010, error=139.99416
>   Epoch=30, lrate=0.010, error=140.00715
>   Epoch=40, lrate=0.010, error=140.00833
>   Epoch=50, lrate=0.010, error=140.00852
>   Epoch=60, lrate=0.010, error=140.00862

    (50, 25, 10)
    Network initialized with 4 layers (Input:7, Hidden:(50, 25, 10), Output:3)
    Starting training with rate=0.02, epochs=200...
    Epoch=10, lrate=0.020, error=138.78218
    Epoch=20, lrate=0.020, error=138.79871
    Epoch=30, lrate=0.020, error=138.81180
    Epoch=40, lrate=0.020, error=138.82419

    (5,)
    >Epoch=10, lrate=0.010, error=138.55466
>   Epoch=20, lrate=0.010, error=137.00640
>   Epoch=30, lrate=0.010, error=134.63245
>   Epoch=40, lrate=0.010, error=131.15649
>   Epoch=50, lrate=0.010, error=126.35278
>   Epoch=60, lrate=0.010, error=120.20582
>   Epoch=70, lrate=0.010, error=113.08263
>   Epoch=80, lrate=0.010, error=105.74218
>   Epoch=90, lrate=0.010, error=98.99337
>   Epoch=100, lrate=0.010, error=93.29678
>   Epoch=110, lrate=0.010, error=88.70422
    Epoch=120, lrate=0.010, error=85.04330
>   Epoch=130, lrate=0.010, error=82.08923
>   Epoch=140, lrate=0.010, error=79.64339
>   Epoch=150, lrate=0.010, error=77.55123
>   Epoch=160, lrate=0.010, error=75.69715
>   Epoch=170, lrate=0.010, error=73.99485
>   Epoch=180, lrate=0.010, error=72.37922
>   Epoch=190, lrate=0.010, error=70.80074
>   Epoch=200, lrate=0.010, error=69.22213

    (5,) has the best performance , to improve its performance, 
    update the Irate and increase the epoch:
    l_rate = 0.02
    n_epoch = 400

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