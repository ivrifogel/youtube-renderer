FROM python:3.10-slim

ENV PIP_BREAK_SYSTEM_PACKAGES=1
ENV IMAGEIO_FFMPEG_EXE=/usr/bin/ffmpeg

RUN apt-get update && apt-get install -y \
    ffmpeg imagemagick libmagick++-dev ghostscript git \
    && sed -i 's/none/read,write/g' /etc/ImageMagick*/policy.xml \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download whisper tiny model (smaller, faster for Railway)
RUN python3 -c "from faster_whisper import download_model; download_model('tiny', output_dir='/app/model')"

COPY main.py .

CMD exec gunicorn --bind :${PORT:-8080} --workers 1 --threads 4 --timeout 3600 main:app
