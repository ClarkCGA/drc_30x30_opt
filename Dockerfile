FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends coinor-cbc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

RUN useradd -m mapuser
USER mapuser
WORKDIR /home/mapuser/app


CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser"]
