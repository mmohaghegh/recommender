FROM ubuntu:20.04
RUN apt-get update && apt-get -y dist-upgrade
RUN apt-get install -y curl python3 python3-pip
WORKDIR ./
COPY ./ ./
RUN pip3 install -r requirements.txt
#CMD ["/bin/echo", "\nTraining the model on the training data and measuring its accuracy with validation data ...\n"]
CMD ["python3", "main.py"]
#CMD ["echo", "\nRunning server for receiving queries ...\n"]
#CMD ["python3", "server.py", "&"]
#CMD ["sleep 20"]
#CMD ["echo", "\nSending the query to the server ...\n"]
#RUN chmod u+x request.sh
#CMD "./request.sh"