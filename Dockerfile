# app/Dockerfile

FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && apt-get install -y 

COPY . .

RUN pip3 install -r requirements.txt

EXPOSE 8506

HEALTHCHECK CMD curl --fail http://localhost:8506/_stcore/health

ENTRYPOINT ["streamlit", "run", "streamlit_app.py", "--server.port=8506", "--server.address=0.0.0.0"]