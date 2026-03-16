FROM python:3.10-slim

ENV PIP_BREAK_SYSTEM_PACKAGES=1
ENV IMAGEIO_FFMPEG_EXE=/usr/bin/ffmpeg
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    ffmpeg imagemagick libmagick++-dev ghostscript \
    && sed -i 's/none/read,write/g' /etc/ImageMagick*/policy.xml \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download whisper tiny model at build time
RUN python3 -c "from huggingface_hub import snapshot_download; snapshot_download('Systran/faster-whisper-tiny', local_dir='/app/model', local_dir_use_symlinks=False)" || \
    python3 -c "import urllib.request, os; os.makedirs('/app/model', exist_ok=True); [urllib.request.urlretrieve(f'https://huggingface.co/Systran/faster-whisper-tiny/resolve/main/{f}', f'/app/model/{f}') for f in ['model.bin','config.json','vocabulary.txt']]"

COPY main.py .

CMD exec gunicorn --bind :${PORT:-8080} --workers 1 --threads 4 --timeout 3600 main:app
