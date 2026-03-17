import os
import datetime
import requests
import traceback
import threading
import time
import json
import gc
import shutil
import uuid
import hashlib

# --- FIX FOR "ANTIALIAS" ERROR ---
import PIL.Image
if not hasattr(PIL.Image, 'ANTIALIAS'):
    PIL.Image.ANTIALIAS = PIL.Image.LANCZOS

from flask import Flask, request, jsonify, send_file
from moviepy.editor import *
from moviepy.video.tools.subtitles import SubtitlesClip
from faster_whisper import WhisperModel
import google.generativeai as genai

app = Flask(__name__)

# --- CONFIGURATION ---
RENDER_PRESET = "ultrafast"
RENDER_FPS = 30
RENDER_THREADS = 2
OUTPUT_DIR = "/tmp/rendered_videos"
os.makedirs(OUTPUT_DIR, exist_ok=True)

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

# Track rendered files for download
rendered_files = {}

# --- HELPER FUNCTIONS ---
def download_file(url, local_filename):
    if os.path.exists(local_filename):
        if os.path.getsize(local_filename) > 0:
            return local_filename
        else:
            os.remove(local_filename)

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

def get_whisper_model():
    """Load whisper model - use pre-downloaded or download manually."""
    if os.path.exists("/app/model"):
        return WhisperModel("/app/model", device="cpu", compute_type="int8")

    base_url = "https://huggingface.co/Systran/faster-whisper-tiny/resolve/main"
    files = ["model.bin", "config.json", "vocabulary.txt"]
    local_model_dir = "/tmp/manual_whisper_tiny"
    os.makedirs(local_model_dir, exist_ok=True)

    for filename in files:
        url = f"{base_url}/{filename}"
        local_path = f"{local_model_dir}/{filename}"
        if not download_file(url, local_path):
            raise Exception(f"Failed to download model file: {filename}")

    return WhisperModel(local_model_dir, device="cpu", compute_type="int8")

# --- BATCH UPLOAD TO GEMINI ---
@app.route('/batch_upload_gemini', methods=['POST'])
def handle_gemini_batch():
    data = request.json
    urls = data.get('urls', [])
    voiceover_url = data.get('voiceover_url')

    if not urls:
        return jsonify({"error": "Missing 'urls'"}), 400

    try:
        uploaded_files = []

        # Upload voiceover first
        if voiceover_url:
            vo_path = f"/tmp/vo_{uuid.uuid4().hex[:8]}.mp3"
            download_file(voiceover_url, vo_path)
            vo_file = genai.upload_file(vo_path, mime_type="audio/mpeg")
            voiceover_result = {"file_uri": vo_file.uri, "name": vo_file.name}
            os.remove(vo_path)
        else:
            voiceover_result = None

        # Upload videos with generic names
        use_generic = data.get('use_generic_names', False)
        for i, url in enumerate(urls):
            try:
                local_path = f"/tmp/video_{i+1}.mp4"
                if download_file(url, local_path):
                    display_name = f"video_{i+1}.mp4" if use_generic else None
                    uploaded = genai.upload_file(local_path, mime_type="video/mp4", display_name=display_name)
                    uploaded_files.append({"file_uri": uploaded.uri, "name": uploaded.name})
                    os.remove(local_path)
                    print(f"Uploaded {i+1}/{len(urls)}")
                else:
                    uploaded_files.append({"file_uri": "", "name": "", "error": f"Download failed: {url}"})
            except Exception as e:
                uploaded_files.append({"file_uri": "", "name": "", "error": str(e)})
                print(f"Upload error for {url}: {e}")

        # Wait for files to become ACTIVE
        print("Waiting for files to become ACTIVE...")
        time.sleep(30)

        # Verify file states
        active_files = []
        for f_info in uploaded_files:
            if f_info.get('file_uri'):
                try:
                    f = genai.get_file(f_info['name'])
                    if f.state.name == 'ACTIVE':
                        active_files.append(f_info)
                    else:
                        print(f"File {f_info['name']} state: {f.state.name}, waiting...")
                        time.sleep(10)
                        f = genai.get_file(f_info['name'])
                        if f.state.name == 'ACTIVE':
                            active_files.append(f_info)
                        else:
                            active_files.append({**f_info, "state": f.state.name})
                except:
                    active_files.append(f_info)
            else:
                active_files.append(f_info)

        return jsonify({
            "status": "success",
            "files": active_files,
            "voiceover": voiceover_result
        })

    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500

# --- DIRECTOR MODE (RENDER) ---
def process_timeline_job(data, job_id):
    print(f"Starting render job {job_id}...")
    base_dir = f"/tmp/render_{job_id}"
    os.makedirs(base_dir, exist_ok=True)
    user_filename = data.get('filename', f"video_{job_id}").strip().replace(" ", "_")
    if not user_filename.lower().endswith('.mp4'):
        user_filename += ".mp4"
    output_path = f"{OUTPUT_DIR}/{job_id}.mp4"

    try:
        audio_url = data.get('audio_url')
        music_url = data.get('music_url')
        edl = data.get('timeline', [])
        font_path = download_font()

        if not audio_url:
            raise Exception("Missing 'audio_url'")
        local_audio_path = f"{base_dir}/master_audio.mp3"
        if not download_file(audio_url, local_audio_path):
            raise Exception(f"Could not download audio from: {audio_url}")

        # --- Transcribe ---
        print("Loading Whisper Model...")
        model = get_whisper_model()
        segments, _ = model.transcribe(local_audio_path, word_timestamps=True)
        subs = []
        for s in segments:
            for w in s.words:
                clean = w.word.strip().upper().replace('"', '').replace('.', '').replace(',', '')
                if clean:
                    subs.append(((w.start, w.end), clean))
        del model
        gc.collect()
        print(f"Transcription complete: {len(subs)} words")

        # Fill subtitle gaps
        filled_subs = []
        for j in range(len(subs)):
            start, end = subs[j][0]
            text = subs[j][1]
            if j == 0:
                start = 0.0
            if j < len(subs) - 1:
                next_start = subs[j+1][0][0]
                if next_start > end:
                    end = next_start
            filled_subs.append(((start, end), text))

        master_audio_clip = AudioFileClip(local_audio_path)
        final_duration = master_audio_clip.duration

        # --- Process video clips ---
        final_clips = []
        asset_map = {}

        for index, cut in enumerate(edl):
            url = cut.get('url')
            t_start = float(cut.get('timeline_start', 0.0))
            t_end = float(cut.get('timeline_end', 0.0))
            s_start = float(cut.get('source_start', 0.0))
            duration = t_end - t_start
            if duration <= 0 or t_start >= final_duration:
                continue

            if url not in asset_map:
                l_path = f"{base_dir}/v_{len(asset_map)}.mp4"
                if download_file(url, l_path):
                    asset_map[url] = l_path

            if url in asset_map:
                try:
                    clip = VideoFileClip(asset_map[url]).without_audio()
                    # Ensure source_start doesn't exceed clip duration
                    if s_start + duration > clip.duration:
                        s_start = max(0, clip.duration - duration)
                    if s_start < 0:
                        s_start = 0
                    clip = clip.subclip(s_start, min(s_start + duration, clip.duration))
                    clip = clip.resize(height=1920)
                    if clip.w < 1080:
                        clip = clip.resize(width=1080)
                    clip = clip.crop(x_center=clip.w/2, y_center=clip.h/2, width=1080, height=1920)
                    clip = clip.set_start(t_start).set_duration(duration)
                    final_clips.append(clip)
                    gc.collect()
                except Exception as e:
                    print(f"Error processing clip {index} ({url}): {e}")

        if not final_clips:
            raise Exception("No video clips were processed!")

        # --- Composite ---
        print(f"Compositing {len(final_clips)} clips...")
        video_layer = CompositeVideoClip(final_clips, size=(1080, 1920)).set_duration(final_duration)

        # Audio: voiceover + optional music
        if music_url:
            local_music = f"{base_dir}/music.mp3"
            if download_file(music_url, local_music):
                music_track = AudioFileClip(local_music).volumex(0.08)
                if music_track.duration < final_duration:
                    music_track = afx.audio_loop(music_track, duration=final_duration)
                else:
                    music_track = music_track.set_duration(final_duration)
                final_audio = CompositeAudioClip([master_audio_clip, music_track])
            else:
                final_audio = master_audio_clip
        else:
            final_audio = master_audio_clip

        video_layer = video_layer.set_audio(final_audio)

        # --- Subtitles ---
        def sub_generator(txt):
            return TextClip(
                txt,
                font=font_path if font_path else 'Arial',
                fontsize=100,
                color='#fcba03',
                stroke_color='black',
                stroke_width=4,
                method='caption',
                size=(1000, None)
            )

        subtitles = SubtitlesClip(filled_subs, sub_generator).set_position('center')
        final_export = CompositeVideoClip([video_layer, subtitles])

        # --- Render ---
        print("Rendering...")
        final_export.write_videofile(
            output_path,
            fps=RENDER_FPS,
            codec="libx264",
            audio_codec="aac",
            preset=RENDER_PRESET,
            threads=RENDER_THREADS
        )

        file_size_mb = round(os.path.getsize(output_path) / (1024 * 1024), 2)

        rendered_files[job_id] = {
            "path": output_path,
            "filename": user_filename,
            "duration": round(final_duration, 2),
            "file_size_mb": file_size_mb,
            "created_at": time.time(),
            "status": "ready"
        }

        print(f"Render complete: {user_filename} ({file_size_mb}MB)")

        # Call webhook if provided
        webhook_url = data.get('webhook_url')
        if webhook_url:
            requests.post(webhook_url, json={
                "status": "success",
                "job_id": job_id,
                "filename": user_filename,
                "download_url": f"/download/{job_id}",
                "duration_seconds": round(final_duration, 2),
                "file_size_mb": file_size_mb
            })

    except Exception as e:
        print(f"RENDER ERROR: {e}")
        traceback.print_exc()
        rendered_files[job_id] = {"status": "error", "error": str(e)}
        webhook_url = data.get('webhook_url')
        if webhook_url:
            requests.post(webhook_url, json={"status": "error", "job_id": job_id, "error": str(e)})
    finally:
        try:
            shutil.rmtree(base_dir)
        except:
            pass

# --- ROUTES ---
@app.route('/', methods=['POST'])
def handle_request():
    data = request.json
    job_id = uuid.uuid4().hex[:12]
    rendered_files[job_id] = {"status": "rendering"}
    threading.Thread(target=process_timeline_job, args=(data, job_id)).start()
    return jsonify({"message": "Job started", "job_id": job_id}), 202

@app.route('/status/<job_id>', methods=['GET'])
def check_status(job_id):
    if job_id not in rendered_files:
        return jsonify({"error": "Job not found"}), 404
    info = rendered_files[job_id]
    return jsonify(info)

@app.route('/download/<job_id>', methods=['GET'])
def download_video(job_id):
    if job_id not in rendered_files:
        return jsonify({"error": "Job not found"}), 404
    info = rendered_files[job_id]
    if info.get('status') != 'ready':
        return jsonify({"error": "Not ready", "status": info.get('status')}), 400
    return send_file(info['path'], as_attachment=True, download_name=info['filename'])

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok", "rendered_count": len(rendered_files)})

# Cleanup old files (older than 1 hour)
def cleanup_old_files():
    while True:
        time.sleep(3600)
        now = time.time()
        to_delete = []
        for job_id, info in rendered_files.items():
            if info.get('created_at') and now - info['created_at'] > 3600:
                try:
                    os.remove(info.get('path', ''))
                except:
                    pass
                to_delete.append(job_id)
        for jid in to_delete:
            del rendered_files[jid]
        if to_delete:
            print(f"Cleaned up {len(to_delete)} old renders")

cleanup_thread = threading.Thread(target=cleanup_old_files, daemon=True)
cleanup_thread.start()

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
