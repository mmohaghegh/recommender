FROM ubuntu:20.04
RUN apt-get update && apt-get -y dist-upgrade
RUN apt-get install -y curl python3 python3-pip
WORKDIR ./
COPY ./ ./
RUN pip3 install -r requirements.txt
EXPOSE 5000
RUN python3 main.py
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "server:app"]
