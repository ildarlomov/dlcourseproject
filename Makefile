.PHONY: clean data lint requirements sync_data_to_s3 sync_data_from_s3

#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
BUCKET = [OPTIONAL] your-bucket-for-syncing-data (do not include 's3://')
PROFILE = default
PROJECT_NAME = ai
PYTHON_INTERPRETER = python3

ifeq (,$(shell which conda))
HAS_CONDA=False
else
HAS_CONDA=True
endif

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Install Python Dependencies
requirements: test_environment
	$(PYTHON_INTERPRETER) -m pip install -U pip setuptools wheel
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt

## Make Dataset
data: requirements
	$(PYTHON_INTERPRETER) src/data/make_dataset.py data/raw data/processed

## Delete all compiled Python files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

## Lint using flake8
lint:
	flake8 src

## Upload Data to S3
sync_data_to_s3:
ifeq (default,$(PROFILE))
	aws s3 sync data/ s3://$(BUCKET)/data/
else
	aws s3 sync data/ s3://$(BUCKET)/data/ --profile $(PROFILE)
endif

## Download Data from S3
sync_data_from_s3:
ifeq (default,$(PROFILE))
	aws s3 sync s3://$(BUCKET)/data/ data/
else
	aws s3 sync s3://$(BUCKET)/data/ data/ --profile $(PROFILE)
endif

## Set up python interpreter environment
create_environment:
ifeq (True,$(HAS_CONDA))
		@echo ">>> Detected conda, creating conda environment."
ifeq (3,$(findstring 3,$(PYTHON_INTERPRETER)))
	conda create --name $(PROJECT_NAME) python=3
else
	conda create --name $(PROJECT_NAME) python=2.7
endif
		@echo ">>> New conda env created. Activate with:\nsource activate $(PROJECT_NAME)"
else
	$(PYTHON_INTERPRETER) -m pip install -q virtualenv virtualenvwrapper
	@echo ">>> Installing virtualenvwrapper if not already intalled.\nMake sure the following lines are in shell startup file\n\
	export WORKON_HOME=$$HOME/.virtualenvs\nexport PROJECT_HOME=$$HOME/Devel\nsource /usr/local/bin/virtualenvwrapper.sh\n"
	@bash -c "source `which virtualenvwrapper.sh`;mkvirtualenv $(PROJECT_NAME) --python=$(PYTHON_INTERPRETER)"
	@echo ">>> New virtualenv created. Activate with:\nworkon $(PROJECT_NAME)"
endif

## Test python environment is setup correctly
test_environment:
	$(PYTHON_INTERPRETER) test_environment.py

#################################################################################
# PROJECT RULES                                                                 #
#################################################################################

run_eval:
#	./run.sh data/raw/train_df.csv data/raw/train_df_track_order_df.csv data/raw/train_df_descriptors.npy models/baseline/agg_descriptors.npy
	./run.sh

run_get_scores:
	python get_scores.py \
	models/baseline/train_df_agg_descriptors.npy \
	data/raw/train_df_track_order_df.csv \
	data/raw/train_gt_df.csv \
	data/raw/train_gt_descriptors.npy

freeze:
	pip freeze > requirements.txt

push_reqs:
	git add requirements.txt
	git commit -m 'fix reqs.txt'
	git push

run_train:
	pip install -e .
	python run_train.py

#run_get_scores:
#	python get_scores.py \
#	--predicted_descr_path=models/baseline/train_df_agg_descriptors.npy \
#	--test_track_order_path=data/raw/train_df_track_order_df.csv \
#	--test_gt_df_path=data/raw/train_gt_df.csv \
#	--gt_descriptors_path=data/raw/train_gt_descriptors.npy

#################################################################################
# Jupyter notebook launch                                                       #
#################################################################################

new_research:
	cp prod Pipfile & pipenv install

update_prod_env:
	cp reseach_pipenv from $(taskid) to prod_pipenv


PROJECT_DIR=$(shell pwd)

#USER=$(shell whoami)
#USERG=$(id -g $USER)
NOTEBOOK_IMAGE=project_epsilon
HOROVOD_IMAGE=horovod36:latest
CONTAINER_NAME=ildar-ai
USER_UID=$(shell id -u)
USER_GID=$(shell id -g)
USER_NAME=$(shell echo $(USER))
VOLUME_DIR=/home/$(USER_NAME)/playground/ai

ifndef PORT
    PORT=5000
endif

ifndef NOTEBOOK_PORT
    NOTEBOOK_PORT=1713
endif

pyclean:
	find . | grep -E "(__pycache__|\.pyc|\.pyo)" | xargs rm -rf

clean-build:
	rm --force --recursive build/
	rm --force --recursive dist/
	rm --force --recursive *.egg-info

get_reqs_dev:
	pipenv install
	pipenv lock -r > requirements.txt
	pipenv lock -r --dev >> requirements.txt

notebook-gpu-it:
	echo "Launching notebook http://research.ostrovok.in:$(NOTEBOOK_PORT)"
	nvidia-docker run -it --privileged -e CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 --rm -p $(NOTEBOOK_PORT):8888 -v $(PROJECT_DIR):$(VOLUME_DIR) --name $(CONTAINER_NAME) --user=root \
    -e NB_USER=$(USER_NAME) \
    -e NB_UID=$(USER_UID) \
    -e NB_GID=$(USER_GID) \
    -e GRANT_SUDO=yes \
    -e CHOWN_HOME=yes \
    $(NOTEBOOK_IMAGE) bash

gpu-it:
	nvidia-docker run -it --shm-size 32G --workdir=$(VOLUME_DIR) --privileged -e CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 --rm  -v $(PROJECT_DIR):$(VOLUME_DIR) --name $(CONTAINER_NAME) --user=root \
    $(HOROVOD_IMAGE) bash


test: pyclean
	pytest tests/

run_server:
	FLASK_APP=fnb/api/app.py flask run --reload

.PHONY: pyclean clean-build notebook lab test run_server
#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

# Inspired by <http://marmelab.com/blog/2016/02/29/auto-documented-makefile.html>
# sed script explained:
# /^##/:
# 	* save line in hold space
# 	* purge line
# 	* Loop:
# 		* append newline + line to hold space
# 		* go to next line
# 		* if line starts with doc comment, strip comment character off and loop
# 	* remove target prerequisites
# 	* append hold space (+ newline) to line
# 	* replace newline plus comments by `---`
# 	* print line
# Separate expressions are necessary because labels cannot be delimited by
# semicolon; see <http://stackoverflow.com/a/11799865/1968>
.PHONY: help
help:
	@echo "$$(tput bold)Available rules:$$(tput sgr0)"
	@echo
	@sed -n -e "/^## / { \
		h; \
		s/.*//; \
		:doc" \
		-e "H; \
		n; \
		s/^## //; \
		t doc" \
		-e "s/:.*//; \
		G; \
		s/\\n## /---/; \
		s/\\n/ /g; \
		p; \
	}" ${MAKEFILE_LIST} \
	| LC_ALL='C' sort --ignore-case \
	| awk -F '---' \
		-v ncol=$$(tput cols) \
		-v indent=19 \
		-v col_on="$$(tput setaf 6)" \
		-v col_off="$$(tput sgr0)" \
	'{ \
		printf "%s%*s%s ", col_on, -indent, $$1, col_off; \
		n = split($$2, words, " "); \
		line_length = ncol - indent; \
		for (i = 1; i <= n; i++) { \
			line_length -= length(words[i]) + 1; \
			if (line_length <= 0) { \
				line_length = ncol - indent - length(words[i]) - 1; \
				printf "\n%*s ", -indent, " "; \
			} \
			printf "%s ", words[i]; \
		} \
		printf "\n"; \
	}' \
	| more $(shell test $(shell uname) = Darwin && echo '--no-init --raw-control-chars')
