# temporal
how do humans learn temporal connectives? and do LMs learn in a similar way?

## directory
* `train.py` trains a bunch of models and dumps their outputs
* `data` stores these outputs for both uniform and childes ratios
* `test.py` evaluates these outputs by randomly generating a test set
* `results` stores the outputs of evaluation
* `analysis.R` loads these outputs in and plots them
* `plots` stores the outputs of analysis

## dependencies
all of the model training/testing requires the [LOTlib3](https://github.com/piantado/LOTlib3/) library. you also need numpy.