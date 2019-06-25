ai
==============================

solution for machines can see competition

### What is given:

1. run eval.py to get descriptors for all referenced video track sequences

### How to run eval
 `$ make run_eval` or `$ ./run.sh`
 
### Notes for horovod image:

1. Install via pip from requirements directly
2. Replace default python with py36 in horovod image by adding extra repository

```
apt-get update
apt-get install -y libsm6 libxext6 libxrender-dev
pip install opencv-python catalyst

```
### What to do

The competition is on overcoming accuracy of face verification problem by using short video tracks instead of single picture of a person.
Data is presented in 10-frame sequences sampled uniformly throughout the video along with a number of person pictures.
Evaluation is done automatically by sampling negative and positive pairs from the data, applying some algorithm to the sequences and calculating True Positive Rate at False Positive Rate equals 1e-10.

### What was done

1. I built this folder using my favourite template cookiecutter data science
2. I've learnt how to make custom NN submissions by creating a number of scripts and get expected baseline results using my own weights from dropbox
3. Then I tried to tune only the head layers using the triplet net model on triplets of images and did not get better TPR 
4. Then I decided to use information from the sequences by tuning 1-layer LSTM model applied on top of sequences of baseline embeddings
Here I did not use the original images but only baseline embeddings which rapidly increases the speed of training.



Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.testrun.org


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
