# Recommender

Develop a very basic recommender system for suggesting TV shows based on some features of users.

## Requirement

All you need is a Docker Engine. For details concerning how install Docker visit its website by clicking [here](https://docs.docker.com/get-docker/). Next follow steps below to run the forcaster on a Docker container.

1. Clone the current repository `git clone https://github.com/mmohaghegh/recommender.git`
2. Navigate to the main directory `cd recommender`
3. Copy the data file *adition_forecaster_application_data.snappy.parquet* to the *recommender* directory.
4. After making sure that your docker is running build the docker image from *Dockerfile* by running `docker build --tag <your-desired-tag> ./` in your terminal.
5. Next run a docker container in background from the image that you have just built `docker run -d -p 5000:5000 --name <your-desired-cont-name> <the-tag-you-set-in-4>`.
6. Now the forcaster is running on a Docker container. The desired query and the accompanying terminal commands are already in *request.sh*. Run it in the terminal `bash request.py`
7. That was it, enjoy suggestions! :wink:
8. Once done with suggestions stop container from running `docker stop <the-cont-name-you-set-in-5>`.
