FROM python:3.9.16
EXPOSE 8080
cmd mkdir -p /app
WORKDIR /app
COPY requirements.txt ./requirements.txt
RUN apt-get update && apt-get install -y libsndfile1-dev
RUN pip3 install -r requirements.txt
COPY . ./
ENTRYPOINT [ "streamlit", "run", "app.py", "--server.port=8080", "--server.address=0.0.0.0" ]
CMD ["app.py"]