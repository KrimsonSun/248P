Result:

 --- Configuration Loaded ---
MAX_FEATURES: 10000, MAX_LEN: 100, RNN_UNITS: 32

Loading IMDB data...
Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/imdb.npz
17464789/17464789 ━━━━━━━━━━━━━━━━━━━━ 0s 0us/step
25000 train sequences
25000 test sequences
Pad sequences (samples x time) to maxlen=100...
x_train shape: (25000, 100)
x_test shape: (25000, 100)

======================================================================
Building and training SimpleRNN Classifier...

--- SimpleRNN Model Summary ---
/usr/local/lib/python3.12/dist-packages/keras/src/layers/core/embedding.py:97: UserWarning: Argument `input_length` is deprecated. Just remove it.
  warnings.warn(
Model: "sequential"

┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Layer (type)                    ┃ Output Shape           ┃       Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ embedding (Embedding)           │ ?                      │   0 (unbuilt) │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ simple_rnn (SimpleRNN)          │ ?                      │   0 (unbuilt) │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense (Dense)                   │ ?                      │   0 (unbuilt) │
└─────────────────────────────────┴────────────────────────┴───────────────┘

 Total params: 0 (0.00 B)

 Trainable params: 0 (0.00 B)

 Non-trainable params: 0 (0.00 B)

Epoch 1/10
157/157 ━━━━━━━━━━━━━━━━━━━━ 5s 25ms/step - acc: 0.5128 - loss: 0.6943 - val_acc: 0.5590 - val_loss: 0.6846
Epoch 2/10
157/157 ━━━━━━━━━━━━━━━━━━━━ 3s 18ms/step - acc: 0.6165 - loss: 0.6674 - val_acc: 0.6756 - val_loss: 0.6378
Epoch 3/10
157/157 ━━━━━━━━━━━━━━━━━━━━ 3s 19ms/step - acc: 0.7062 - loss: 0.6117 - val_acc: 0.7192 - val_loss: 0.5947
Epoch 4/10
157/157 ━━━━━━━━━━━━━━━━━━━━ 4s 23ms/step - acc: 0.7615 - loss: 0.5529 - val_acc: 0.7440 - val_loss: 0.5478
Epoch 5/10
157/157 ━━━━━━━━━━━━━━━━━━━━ 3s 19ms/step - acc: 0.8018 - loss: 0.4931 - val_acc: 0.7488 - val_loss: 0.5216
Epoch 6/10
157/157 ━━━━━━━━━━━━━━━━━━━━ 4s 22ms/step - acc: 0.8311 - loss: 0.4369 - val_acc: 0.7786 - val_loss: 0.4800
Epoch 7/10
157/157 ━━━━━━━━━━━━━━━━━━━━ 6s 25ms/step - acc: 0.8511 - loss: 0.3861 - val_acc: 0.7660 - val_loss: 0.4954
Epoch 8/10
157/157 ━━━━━━━━━━━━━━━━━━━━ 3s 19ms/step - acc: 0.8652 - loss: 0.3525 - val_acc: 0.7958 - val_loss: 0.4424
Epoch 9/10
157/157 ━━━━━━━━━━━━━━━━━━━━ 5s 21ms/step - acc: 0.8803 - loss: 0.3241 - val_acc: 0.7860 - val_loss: 0.4722
Epoch 10/10
157/157 ━━━━━━━━━━━━━━━━━━━━ 4s 25ms/step - acc: 0.8931 - loss: 0.2964 - val_acc: 0.7936 - val_loss: 0.4985

Performance charts for SimpleRNN Classifier saved to simplernn_classifier_performance.png

======================================================================
Building and training LSTM Classifier...

--- LSTM Model Summary ---
Model: "sequential_1"

┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Layer (type)                    ┃ Output Shape           ┃       Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ embedding_1 (Embedding)         │ ?                      │   0 (unbuilt) │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ lstm (LSTM)                     │ ?                      │   0 (unbuilt) │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_1 (Dense)                 │ ?                      │   0 (unbuilt) │
└─────────────────────────────────┴────────────────────────┴───────────────┘

 Total params: 0 (0.00 B)

 Trainable params: 0 (0.00 B)

 Non-trainable params: 0 (0.00 B)

Epoch 1/10
157/157 ━━━━━━━━━━━━━━━━━━━━ 8s 41ms/step - acc: 0.5378 - loss: 0.6926 - val_acc: 0.5768 - val_loss: 0.6916
Epoch 2/10
157/157 ━━━━━━━━━━━━━━━━━━━━ 6s 36ms/step - acc: 0.6113 - loss: 0.6904 - val_acc: 0.6200 - val_loss: 0.6878
Epoch 3/10
157/157 ━━━━━━━━━━━━━━━━━━━━ 7s 45ms/step - acc: 0.6448 - loss: 0.6831 - val_acc: 0.7038 - val_loss: 0.6166
Epoch 4/10
157/157 ━━━━━━━━━━━━━━━━━━━━ 6s 37ms/step - acc: 0.7106 - loss: 0.5872 - val_acc: 0.7316 - val_loss: 0.5407
Epoch 5/10
157/157 ━━━━━━━━━━━━━━━━━━━━ 10s 37ms/step - acc: 0.7586 - loss: 0.5139 - val_acc: 0.7586 - val_loss: 0.5061
Epoch 6/10
157/157 ━━━━━━━━━━━━━━━━━━━━ 10s 38ms/step - acc: 0.7935 - loss: 0.4653 - val_acc: 0.7624 - val_loss: 0.4847
Epoch 7/10
157/157 ━━━━━━━━━━━━━━━━━━━━ 7s 42ms/step - acc: 0.8165 - loss: 0.4256 - val_acc: 0.7884 - val_loss: 0.4484
Epoch 8/10
157/157 ━━━━━━━━━━━━━━━━━━━━ 6s 38ms/step - acc: 0.8407 - loss: 0.3885 - val_acc: 0.7976 - val_loss: 0.4327
Epoch 9/10
157/157 ━━━━━━━━━━━━━━━━━━━━ 11s 40ms/step - acc: 0.8572 - loss: 0.3612 - val_acc: 0.8102 - val_loss: 0.4139
Epoch 10/10
157/157 ━━━━━━━━━━━━━━━━━━━━ 7s 44ms/step - acc: 0.8704 - loss: 0.3346 - val_acc: 0.8028 - val_loss: 0.4146

Performance charts for LSTM Classifier saved to lstm_classifier_performance.png

--- Execution Complete ---
Two RNN-based classifiers (SimpleRNN and LSTM) have been trained and results plotted.
By comparing the validation accuracy charts of SimpleRNN and LSTM, you should observe that LSTM generally performs better in handling temporal dependencies.
