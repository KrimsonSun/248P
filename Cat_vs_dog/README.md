usage :
```bash 
module load cuda/11.7
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDA_HOME/lib64
python catvsdog.py /home/lopes/Datasets/dogs-vs-cats ./smaller
```

Model is store by .keras file because the .h5 is not suggested format for TensorFlow above 2.5version ;
For opengpu , we can only load cuda 11.7 and that require the tensorflow at least 2.6 version.


Result
--- Displaying Training and Validation Curves ---

--- Evaluating Best Saved Model on Test Set ---
62/62 [==============================] - 13s 202ms/step - loss: 0.4745 - accuracy: 0.7873

Final Test Accuracy: 0.7873