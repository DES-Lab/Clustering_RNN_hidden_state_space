# On the Relationship Between RNN Hidden State Vectors and Semantic Ground Truth

Code supporting the experiments from paper "On the Relationship Between RNN Hidden State Vectors and Semantic Ground Truth".

## Install

All experiments were performed with Python 3.9. We suggest crating a [virtual environment](https://docs.python.org/3/library/venv.html) for installation of recquirements.
```
pip install -r recquirements.txt
```
To run experiments with CUDA do:
```
pip3 install torch==1.10.1+cu102 -f https://download.pytorch.org/whl/cu102/torch_stable.html
```

## Run Experiments
For training of all RNNs. This step can be skipped as we provide pretrained RNNs for all experiments.
If you want to check if the trained RNN loading works, simply change the `perform_training` 
variable found at line 44 in `automated_trainer.py` to False.
``
python automated_trainer.py
``
For clustering of hidden-state vectors and clustering evaluation of trained RNNs:
```
python clustering_comparison.py
```
For clustering of hidden-state vectors and clustering evaluation of noisy correct-by-constructions RNNs:
```
python retraining_noisy_constructed_RNNs.py
```

# Statistics Computations and Figure Creation
Results from our experiments can be found in `experiment_results` folder.
To compute statistics and print them to output, as well as to create figures found in paper use:

```
python statistics_computation.py -metric X -experiment_set Y -min_accuracy Z 
```
Argument options:
- metric: Measurement metric: either `amb` (ambiguity), `wamb` (weighted ambiguity), or `size` (cluster size)
- experiment_set: `all` for all normal RNNs, `retrained` for noisy constructed RNNs, or one of {`lstm`, `relu`, `tanh`, `gru`} for only that network type
- min_accuracy: value between `[0,1]` which defines minimum accuracy RNN needs to achieve to be considered in statistics computation

For example:
```
python statistics_computation.py -metric wamb -min_accuracy 0.9 -experiment_set lstm
```