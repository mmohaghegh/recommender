# forcaster

Develop a very basic recommender system for suggesting TV shows based on some features of users.

## Requirement

All you need is a Docker Engine. For details concerning how to install Docker visit its website by clicking [here](https://docs.docker.com/get-docker/). Next follow steps below to run the forcaster on a Docker container.

## How to run the forcaster

1. Clone the current repository `git clone https://github.com/mmohaghegh/recommender.git`.
2. Navigate to the main directory `cd recommender`.
3. Copy the data file *adition_forecaster_application_data.snappy.parquet* to the *recommender* to the directory. Alternatively, you can open `parameters.json` and under key "data" set the path, where the data file is located ("path"). Feel free to change the "fl_name" when it is different in your system.
4. After making sure that your docker engine is running build the docker image, described in *Dockerfile*, by running `docker build --tag <your-desired-tag> ./` in your terminal.
5. Next run a docker container in background from the image that you have just built `docker run -d -p 5000:5000 --name <your-desired-cont-name> <the-tag-you-set-in-4>`.
6. Now the forcaster is running on a Docker container. The desired query and the accompanying terminal commands are already in *request.sh*. Run it in the terminal `bash request.py`
7. That was it, enjoy suggestions! :wink:
8. Once done with suggestions you can stop container `docker stop <the-cont-name-you-set-in-5>`.

## Brief description of what each code does

`forcaster.py` contains two subclasses, one handling the data and the other handling the model.

`model_selection.py` selects the model with the best accuracy.

`main.py` trains the model and save the model in a pickle file.

`server.py` loads the model from disk and run a flask server on port 5000, which receives queries and returns the top three show ids.

`requirements.txt` contains the python packages required for this project.

`parameters.json` contains where to look for the data and the good model hyperparameter set that was found using `model_selection.py`.

`model_selection.pdf` illustrates accuracies of models with different complexities.

`helper.py` contains some python functions used in the preprocessing of the data.
