# this is an official Python runtime, used as the parent image
FROM python:3.7.4

# set the working directory in the container to /app
WORKDIR /app

# add the current directory to the container as /app
ADD . /app

# execute everyone's favorite pip command, pip install -r
RUN pip install --trusted-host pypi.python.org -r requirements.txt

COPY ./models /models
COPY ./test /test

# unblock port 5001 for the Flask app to run on
EXPOSE 5001

# run the test script before starting
RUN python test/app_test.py

# execute the Flask app
CMD ["python", "app/app.py"]