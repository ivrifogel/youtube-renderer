import os
import datetime
import requests
import traceback
import threading
import time
import json
import gc
import shutil

# --- FIX FOR "ANTIALIAS" ERROR ---
import PIL.Image
if not hasattr(PIL.Image, 'ANTIALIAS'):
    PIL.Image.ANTIALIAS = PIL.Image.LANCZOS

from flask import Flask, request, jsonify
from google.cloud import storage
from moviepy.editor import *
from moviepy.video.tools.subtitles import SubtitlesClip
from faster_whisper import WhisperModel
import google.auth
import google.auth.transport.requests
import google.generativeai as genai

app = Flask(__name__)

# --- CONFIGURATION ---
RENDER_PRESET = "ultrafast" 
RENDER_FPS = 30 
RENDER_THREADS = 1
BUCKET_NAME = "youtube-videogeneration" 
SERVICE_ACCOUNT_EMAIL = "693305496226-compute@developer.gserviceaccount.com"
WHISPER_MODEL_SIZE = "tiny" 

# --- API KEY SETUP ---
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
    except Exception as e:
        print(f"API Key Config Error: {e}")

# ASSETS
TRANSITION_VIDEO_URL = "https://storage.googleapis.com/youtube-videogeneration/LightLeak.mp4"
FONT_URL = "https://raw.githubusercontent.com/JulietaUla/Montserrat/master/fonts/ttf/Montserrat-Black.ttf"

# --- HELPER FUNCTIONS ---
def download_file(url, local_filename):
    if os.path.exists(local_filename):
        # Check if file is not empty
        if os.path.getsize(local_filename) > 0:
            return local_filename
        else:
            os.remove(local_filename) # Remove empty file
            
    print(f"Downloading: {url} -> {local_filename}")
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        with requests.get(url, stream=True, timeout=120, headers=headers) as r:
            r.raise_for_status()
            with open(local_filename, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        return local_filename
    except Exception as e:
        print(f"Download Error for {url}: {e}")
        return None

def download_font():
    local_font_path = "/tmp/Montserrat-Black.ttf"
    download_file(FONT_URL, local_font_path)
    return local_font_path if os.path.exists(local_font_path) else None

# --- MANUAL MODEL DOWNLOADER ---
def download_model_manually():
    """Manually downloads the specific files needed for the 'tiny' model."""
    base_url = "https://huggingface.co/Systran/faster-whisper-tiny/resolve/main"
    files = ["model.bin", "config.json", "vocabulary.txt"]
    local_model_dir = "/tmp/manual_whisper_tiny"
    
    os.makedirs(local_model_dir, exist_ok=True)
    
    print("--- STARTING MANUAL MODEL DOWNLOAD ---")
    for filename in files:
        url = f"{base_url}/{filename}"
        local_path = f"{local_model_dir}/{filename}"
        if not download_file(url, local_path):
            raise Exception(f"Failed to manually download model file: {filename}")
    
    print("--- MANUAL MODEL DOWNLOAD COMPLETE ---")
    return local_model_dir

# --- BATCH UPLOADER (Gemini) ---
def process_batch_upload_gemini(data):
    # ... (Keep this code same as before, simplified for brevity in this paste) ...
    # This section isn't changing, so we focus on the Director Mode below.
    return {"status": "skipped_for_brevity"} 

@app.route('/batch_upload_gemini', methods=['POST'])
def handle_gemini_batch():
    # Placeholder to keep the file valid if you paste the whole thing
    return jsonify({"status": "error", "message": "Use Director Mode"}), 400

# --- DIRECTOR MODE ---
def process_timeline_job(data, webhook_url):
    print("Starting 'Director Mode' Job (Manual Download Fix)...")
    base_dir = f"/tmp/{int(datetime.datetime.now().timestamp())}"
    os.makedirs(base_dir, exist_ok=True)
    user_filename = data.get('filename', "director_cut.mp4").strip().replace(" ", "_")
    if not user_filename.lower().endswith('.mp4'): user_filename += ".mp4"
    output_path = f"{base_dir}/{user_filename}"

    try:
        audio_url = data.get('audio_url')
        edl = data.get('timeline', [])
        font_path = download_font()
        
        if not audio_url: raise Exception("Missing 'audio_url'")
        local_audio_path = f"{base_dir}/master_audio.mp3"
        if not download_file(audio_url, local_audio_path):
             raise Exception(f"Could not download audio from: {audio_url}")
        
        # --- Transcribe (Using Manual Download) ---
        print("Loading Whisper Model...")
        try:
            # 1. Download model files manually first
            model_path = download_model_manually()
            
            # 2. Load model from that local folder
            # Note: We pass the FOLDER path, not a model name
            model = WhisperModel(model_path, device="cpu", compute_type="int8")
            
            segments, _ = model.transcribe(local_audio_path, word_timestamps=True)
            subs = [((w.start, w.end), w.word.strip().upper()) for s in segments for w in s.words]
            
            del model
            gc.collect()
            print("Transcription Complete & RAM Cleared.")
            
        except Exception as e:
            print(f"TRANSCRIPTION CRASH: {e}")
            raise Exception(f"Transcription Failed: {str(e)}")

        master_audio_clip = AudioFileClip(local_audio_path)
        final_duration = master_audio_clip.duration
        
        local_trans_video_path = f"{base_dir}/transition_visual.mp4"
        download_file(TRANSITION_VIDEO_URL, local_trans_video_path)
        
        final_clips = []
        asset_map = {}
        
        # Process clips
        for index, cut in enumerate(edl):
            url = cut.get('url')
            t_start = float(cut.get('timeline_start', 0.0))
            t_end = float(cut.get('timeline_end', 0.0))
            s_start = float(cut.get('source_start', 0.0))
            duration = t_end - t_start
            if duration <= 0 or t_start >= final_duration: continue

            if url not in asset_map:
                l_path = f"{base_dir}/v_{len(asset_map)}.mp4"
                if download_file(url, l_path): asset_map[url] = l_path
            
            if url in asset_map:
                try:
                    clip = VideoFileClip(asset_map[url]).without_audio().subclip(s_start, s_start + duration).resize(height=1920)
                    clip = clip.crop(x_center=clip.w/2, y_center=clip.h/2, width=1080, height=1920).set_start(t_start).set_duration(duration)
                    final_clips.append(clip)
                    gc.collect()
                except Exception as e:
                    print(f"Error processing clip {url}: {e}")

        if not final_clips:
            raise Exception("No video clips were processed successfully!")

        video_layer = CompositeVideoClip(final_clips, size=(1080, 1920)).set_duration(final_duration)
        final_video = video_layer.set_audio(master_audio_clip)
        
        final_export = CompositeVideoClip([final_video, SubtitlesClip(subs, lambda t: TextClip(t, font=font_path if font_path else 'Arial', fontsize=100, color='#fcba03', stroke_color='black', stroke_width=4, method='caption', size=(1000,None))).set_position('center')])
        
        final_export.write_videofile(output_path, fps=RENDER_FPS, codec="libx264", audio_codec="aac", preset=RENDER_PRESET, threads=RENDER_THREADS)

        blob = storage.Client().bucket(BUCKET_NAME).blob(f"generated_videos/{user_filename}")
        blob.upload_from_filename(output_path)
        requests.post(webhook_url, json={"status": "success", "filename": user_filename})

    except Exception as e:
        print(f"RENDER ERROR: {e}")
        traceback.print_exc()
        if webhook_url: requests.post(webhook_url, json={"status": "error", "error": str(e)})
    finally:
        try:
            shutil.rmtree(base_dir)
        except:
            pass

@app.route('/', methods=['POST'])
def handle_request():
    data = request.json
    webhook_url = data.get('webhook_url')
    if not webhook_url: return jsonify({"error": "Missing webhook_url"}), 400
    threading.Thread(target=process_timeline_job, args=(data, webhook_url)).start()
    return jsonify({"message": "Job started"}), 202

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8080)cd video batch_upload_gemini