# Recommender

Develop a very basic recommender system for suggesting TV shows based on some features of users.

## Requirement

All you need is a Docker Engine. For details concerning how install Docker visit its website by clicking [here](https://docs.docker.com/get-docker/). Next follow steps below to run the forcaster on a Docker container.

## How to run the code

1. Clone the current repository `git clone https://github.com/mmohaghegh/recommender.git`
2. Navigate to the main directory `cd recommender`
3. Copy the data file *adition_forecaster_application_data.snappy.parquet* to the *recommender* directory.
4. After making sure that your docker is running build the docker image from *Dockerfile* by running `docker build --tag <your-desired-tag> ./` in your terminal.
5. Next run a docker container in background from the image that you have just built `docker run -d -p 5000:5000 --name <your-desired-cont-name> <the-tag-you-set-in-4>`.
6. Now the forcaster is running on a Docker container. The desired query and the accompanying terminal commands are already in *request.sh*. Run it in the terminal `bash request.py`
7. That was it, enjoy suggestions! :wink:
8. Once done with suggestions stop container from running `docker stop <the-cont-name-you-set-in-5>`.

## Brief description of what each code does

`forcaster.py` is a class that contains two subclasses, one handling the data and the other handling the model.

`model_selection.py` is a python file that select the model with the best accuracy.

`main.py` trains the model and save the model in a pickle file.

`server.py` loads the model from disk and run a flask server on port 5000, which receives queries and returns the top three show ids.

`requirements.txt` contains the python packages required for this repository.

`parameters.json` contains some of the model hyperparameters that was achieved using `model_selection.py`.
