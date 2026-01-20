FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends coinor-cbc \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y gdal-bin libgdal-dev

COPY requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

WORKDIR /app


CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--allow-root", "--no-browser"]
