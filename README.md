# circularitytest

This project is part of the seminar "Empirical Methods for NLP and Data Science" (SoSe 2021) at the Institute of Computational Linguistics,
Heidelberg University.

## Goal

The goal of this package is to provide a python-based implementation of the circularity test defined in [[1](#references)]
using the GAM package pyGAM [[2](#references)]. 
An R-based implementation of the circularity test as well as the data and more can be found 
[here](https://www.cl.uni-heidelberg.de/statnlpgroup/empirical_methods/). 
A short introduction to the circularity test is given in the section [The Circularity Test](#the-circularity-test).



## Installation

This project requires Python >= 3.7.

From source: 

  1. Clone this repository:
  `git clone https://github.com/lisakuhn/circularitytest.git`
  2. Install circularitytest and it's requirements:\
  `cd circularitytest`\
  `pip3 install .` (add `--user` for a local installation if necessary).


## Usage

Basic usage:

`python3 -m circularitytest --config_path configs/ir_example.yaml`  \
(here: config `ir_example.yaml`)

Please also have a look at the notebooks in [`example_notebooks`](example_notebooks) to
see how the individual functions can be used.


## Configuration

Everything you need to run circularitytest from the command line should be specified in the simple [YAML](http://yaml.org/) format. 
An overview of all the options can be found [here](configs/README.md). \
Configurations to run all the examples given in [[1](#References)] can be found in the [`configs`](configs) directory.


## The Circularity Test

Following is a short introduction to the circularity test and its purpose.
Please refer to [[1](#references)] (chapter 2.4.3) for the full definitions, intuition and further explanations 
and examples.

The goal of the circularity test is to identify features in a ML dataset
that allow a reconstruction of the function that defines the gold standard labels. \
If a dataset contains such targets that are defined by a deterministic function and the features that define this function (also called *circular features*)
are included in model training, the ML model will only learn to reconstruct the target function and disregard any other features that might be present. \
This scenario causes problems: While the model will perform (almost) perfectly on data that includes the target-defining
features, it would fail completely on unseen data where the target-defining features are not available/known beforehand.

To avoid (accidentally) using a model trained on circular features, the circularity test defined below can 
be applied in two settings: \
    - On a dataset: Apply the circularity test on a dataset where you suspect features defining the target labeling function. \
    - On a (black-box) model: If you want to know whether a model used circular features in its training, knowledge
    distillation can be applied: Perform the circularity test using the predictions of the model as targets. You only need 
    the test set predictions for this.
    
The circularity test uses interpretable GAMs so that the contribution of each feature can be inspected (and plotted nicely).

The formal definition of the test is as follows: 

**Definition Circularity Test**: \
Given a dataset of feature-label relations 
![formula](https://render.githubusercontent.com/render/math?math=D=\{(x^{n},y^{n})\}^{N}_{n=1})
 where
![formula](https://render.githubusercontent.com/render/math?math=x^{n}=(x_1,x_2,...,x_p))
 is a p-dimensional feature vector, let 
![formula](https://render.githubusercontent.com/render/math?math=C%20\subseteq%20P(\{1,...,p\}))
 indicate the set of candidate circular features in dataset ![formula](https://render.githubusercontent.com/render/math?math=D)
 , and let
![formula](https://render.githubusercontent.com/render/math?math=M:=\{\mu_c:c\in%20C\})
be the set of models obtained by fitting a GAM based on feature set 
![formula](https://render.githubusercontent.com/render/math?math=c)
to the data 
![formula](https://render.githubusercontent.com/render/math?math=D).
 A set of circular features
![formula](https://render.githubusercontent.com/render/math?math=c*)
is detected by applying the following two-step test:
1. ![formula](https://render.githubusercontent.com/render/math?math=c*=argmax_{c\subseteq%20C}D^2(\mu_c)) 
 where 
 ![formula](https://render.githubusercontent.com/render/math?math=D^2(\mu_{c*})) 
 is close to 1, and in case
the maximizer is not unique, the maximizer is chosen whose
associated GAM 
![formula](https://render.githubusercontent.com/render/math?math=\mu_{c*}) 
has the smallest degrees of freedom.
2. The feature shapes of any other features added to the GAM
 ![formula](https://render.githubusercontent.com/render/math?math=D^2(\mu_{c*})) 
 are nullified in the model 
  ![formula](https://render.githubusercontent.com/render/math?math=\mu\{1,...,p\})) 
that is based on the full
feature set.




## Step-by-step visualizations

For examples with visualizations of how the circularity test works and what circularitytest does when called, 
example notebooks for each of the examples in [[1]](#references) can be found in the [`example_notebooks`](example_notebooks) directory. There, the main procedure 
of the circularity test as well as different plots are shown in small code chunks.



## References

1. Stefan Riezler and Michael Hagmann \
Validity, Reliability, and Significance: Empirical Methods for NLP and Data Science\
To be published in: Synthesis Lectures on Human Language Technologies, Morgan & Claypool Publishers, 2021

2. Daniel Servén, & Charlie Brummitt. (2018, March 27). \
pyGAM: Generalized Additive Models in Python. Zenodo. DOI: 10.5281/zenodo.1208723



