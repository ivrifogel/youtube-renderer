FROM python:3.10-slim

ENV PIP_BREAK_SYSTEM_PACKAGES=1
ENV PYTHONUNBUFFERED=1

# Install FFmpeg and font dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    fonts-liberation \
    fontconfig \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download Whisper tiny model (eliminates cold start)
RUN python3 -c "from faster_whisper import download_model; download_model('tiny', output_dir='/app/model')"

# Pre-download Poppins Medium font
RUN mkdir -p /usr/share/fonts/truetype/poppins && \
    python3 -c "import requests; r=requests.get('https://github.com/google/fonts/raw/main/ofl/poppins/Poppins-Medium.ttf'); open('/usr/share/fonts/truetype/poppins/Poppins-Medium.ttf','wb').write(r.content); r=requests.get('https://github.com/google/fonts/raw/main/ofl/poppins/Poppins-Bold.ttf'); open('/usr/share/fonts/truetype/poppins/Poppins-Bold.ttf','wb').write(r.content)" && \
    cp /usr/share/fonts/truetype/poppins/Poppins-Medium.ttf /tmp/Poppins-Medium.ttf && \
    fc-cache -fv

COPY main.py .

CMD exec gunicorn --bind :${PORT:-8080} --workers 1 --threads 8 --timeout 3600 main:app
