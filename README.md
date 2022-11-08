# Efficientnet-Tensorflow
https://towardsdatascience.com/an-in-depth-efficientnet-tutorial-using-tensorflow-how-to-use-efficientnet-on-a-custom-dataset-1cab0997f65c <br/>
```
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/
python3 -m pip install tensorflow
```
## Verify install:
```
python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```
