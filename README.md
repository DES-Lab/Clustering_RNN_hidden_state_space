# Analysis of Clustering-Based Abstraction of RNN State Vectors

Code supporting the experiments from paper "Analysis of Clustering-Based Abstraction of RNN State Vectors".

## Install

All experiments were performed with Python 3.10. 

Open a terminal and run:
```
sudo apt-get update
pip install -r requirements.txt
```

## Minimal Working Example

Small example which shows how a RNN is trained, and ambiguity computed can be executed by running:

```
python3.10 driver.py
```

In this example:
- an RNN with relu activation function will be trained to recognize a finite state machine
- k-means function with k set to 8 * model_size will be computed
- ambiguity and mapping of states to clusters will be printed

## Reduced Version/Subset of All Experiments

Therefore, we provide `run_small_experiment` script which computes and prints ambiguity results a subset of overall network configurations, 
and for a subset of all clustering functions.

To run:
```
python3.10 run_small_experiment.py
```

If you want to train networks yourself, set the `perform_training` in line 71 to True.

Example output of this script can be seen in the file `run_small_experiments_output.txt`

## Run All Experiments
All experiments can be performed with `experiment_runner.py`.

Note on runtime: in our evaluation, we ran the experiments on the pretrained networks over several days. 

This script will train (optionally) and create clusters and compute ambiguity for all experiments found in the paper.
Results of these experiments can be visualized with `python3.10 statistic_computation.py` script.
(Note that you have to set the paper_result flag to False in the beginning of the main function in the statistic_computation.py)

## Statistics Computations and Figure Creation

Results from our experiments can be found in `experiment_results` folder.
To compute statistics, print them and shown figures, simply run `python3.10 statistics_computation.py`.

### Repo Structure

This repository contains code which is is used in the paper "Analysis of Clustering-Based Abstraction of RNN State Vectors",
as well as many unused/unpublished material, such as computation of correct-by-construction RNNs and their retraining after adding noise to the weights.

Repo structure related to the paper:
- driver.pt - minimal working example
- run_small_experiment.py - subset of experiment_runner, much quicker to run
- run_small_experiment_output.txt - saved output of a single execution of run_small_experiment.py
- experiment_runner.py - code recquired to reproduce all data found in the paper
