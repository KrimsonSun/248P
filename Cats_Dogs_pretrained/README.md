Usage:
```bash
python cats_dogs_pretrained.py /home/lopes/Datasets/dogs-vs-cats ./smaller 

```
Epoch 30/30
579/581 ━━━━━━━━━━━━━━━━━━━━ 0s 24ms/step - accuracy: 0.9776 - loss: 0.0596WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`. 
581/581 ━━━━━━━━━━━━━━━━━━━━ 16s 26ms/step - accuracy: 0.9776 - loss: 0.0596 - val_accuracy: 0.9722 - val_loss: 0.0767
WARNING:absl:Compiled the loaded model, but the compiled metrics have yet to be built. `model.compile_metrics` will be empty until you train or evaluate the model.

--- Evaluating Best Fine-Tuned Model on Test Set ---
72/72 ━━━━━━━━━━━━━━━━━━━━ 2s 13ms/step - accuracy: 0.9761 - loss: 0.0536

Final Test Accuracy (VGG16): 0.9761