FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# build the time series, train the production models and check the test suite at build time
RUN python ingestion.py \
 && python ingestion.py data/cs-production \
 && python model.py \
 && python run-tests.py

EXPOSE 8080
CMD ["python", "app.py"]
