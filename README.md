ai
==============================

solution for machines can see competition
(see 'what was done' section)

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
Evaluation is done automatically by sampling negative and positive pairs from the data, applying some algorithm to the 
sequences and calculating True Positive Rate at False Positive Rate equals 1e-6.

### What was done

1. I built this folder using my favourite template cookiecutter data science
2. I decided to use catalyst framework for reproducible experiments. This shipped me a trouble which include a lot of code-typing in order to run my training procedure. 
I spend a lot of time on it and hope next time it will be much faster. 
But, at least I confirmed that my PyCharm is the best debugging tool ever. 
Hope, the main problem here that i've had not such a common task and the software 
just dont have so flexible abstractions to fit my issue. Or even i'm bad at coding :)
All hacks that i did here src/catalyst_hacks/triplet_runner.py and this works
Btw, project structure:
    - src is the whole source code
    - reqs.txt as usual
    - Many useful commands in Makefile
    - run.sh is the final evaluation script which compute scores locally on train data just 
to ensure that gitlab-ci pipeline will work. Here i have a parameter branch which is local or 
leaderboard which defines the usage
    - Dockerfile is from competition, for training I used my own one based on horovod blended with minimal notebook docker images
    - notebooks contains jupyter that explains the data organisation
    - src also contains data for datasets and models for several implemented models
    - data is preserved for storing the data
    - models is for saving all training logs and checkpoints, for example one may unzip https://www.dropbox.com/s/r0yeb76mwhsizyk/models.zip?dl=0 and check tensorboard training logs or even run a model 

2. I've learnt how to make custom NN submissions by creating a number of scripts and get expected baseline results using my own weights from dropbox
3. Then I tried to tune only the head layers using the triplet net model on triplets of images and did not get better TPR 
4. Then I decided to use information from the sequences by tuning 1-layer LSTM model applied on top of sequences of baseline embeddings
Here I did not use the original images but only baseline embeddings which rapidly increases the speed of training.
I used embedding form lstm applied on top of sequence of baseline embeddings and tried to optimize triplet loss function on 1.2 million triplets. 
This gives me perfect learning curves with around 100% TPA at required FPR on both train and dev. This was definitely 
a bug and i got the worst score at leaderboard. Here the competition deadline was and i did not fixed it yet.
I also did not implement a lot of ideas including smart sampling of triplets where loss is computed only on "difficult" cases.
As far as I know in this competition classical ML hacks won, not dl.
That is it.

Summarizing the following: I met catalyst and dont know if it might be useful for me because it's not customizable enough, but i will try to learn more about it.
I tried to use rnn on top of image descriptors but it was not enough for this task. Deeper dive is required.




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
