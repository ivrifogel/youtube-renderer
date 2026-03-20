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
import base64
import subprocess
from concurrent.futures import ThreadPoolExecutor

from flask import Flask, request, jsonify, send_file
from google.cloud import storage as gcs_storage
from faster_whisper import WhisperModel
import google.generativeai as genai
import google.auth
import google.auth.transport.requests as gauth_requests

app = Flask(__name__)

# --- CONFIGURATION ---
RENDER_FPS = 24
GCS_BUCKET = "cashflow-shorts-videos"
OUTPUT_DIR = "/tmp/rendered_videos"
os.makedirs(OUTPUT_DIR, exist_ok=True)
SERVICE_ACCOUNT_EMAIL = "693305496226-compute@developer.gserviceaccount.com"

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
    except Exception as e:
        print(f"API Key Config Error: {e}")

# --- PRE-LOAD WHISPER MODEL ---
def _load_whisper_model():
    print("Pre-loading Whisper model...")
    start = time.time()
    if os.path.exists("/app/model"):
        model = WhisperModel("/app/model", device="cpu", compute_type="int8")
    else:
        local_model_dir = "/tmp/manual_whisper_tiny"
        os.makedirs(local_model_dir, exist_ok=True)
        base_url = "https://huggingface.co/Systran/faster-whisper-tiny/resolve/main"
        for f in ["model.bin", "config.json", "vocabulary.txt"]:
            path = f"{local_model_dir}/{f}"
            if not os.path.exists(path):
                r = requests.get(f"{base_url}/{f}", timeout=30)
                r.raise_for_status()
                with open(path, 'wb') as fh:
                    fh.write(r.content)
        model = WhisperModel(local_model_dir, device="cpu", compute_type="int8")
    print(f"Whisper model loaded in {time.time()-start:.1f}s")
    return model

WHISPER_MODEL = _load_whisper_model()


# --- HELPER FUNCTIONS ---
def download_file(url, local_filename):
    if os.path.exists(local_filename) and os.path.getsize(local_filename) > 0:
        return local_filename

    # GCS direct download (internal network)
    if 'storage.googleapis.com/' in url:
        try:
            parts = url.replace('https://storage.googleapis.com/', '').split('/', 1)
            client = gcs_storage.Client()
            blob = client.bucket(parts[0]).blob(parts[1])
            blob.download_to_filename(local_filename)
            return local_filename
        except Exception as e:
            print(f"  GCS failed: {e}")

    # HTTP download
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        with requests.get(url, stream=True, timeout=120, headers=headers, allow_redirects=True) as r:
            r.raise_for_status()
            if 'text/html' in r.headers.get('Content-Type', ''):
                print(f"  Got HTML: {url}")
                return None
            with open(local_filename, 'wb') as f:
                for chunk in r.iter_content(8192):
                    f.write(chunk)
        if os.path.exists(local_filename) and os.path.getsize(local_filename) > 0:
            return local_filename
        return None
    except Exception as e:
        print(f"  Download error: {e}")
        return None


def get_video_duration(path):
    """Get video duration using ffprobe."""
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", path],
            capture_output=True, text=True, timeout=10
        )
        return float(result.stdout.strip())
    except:
        return 0.0


def group_words_into_chunks(word_subs, max_words=3):
    if not word_subs:
        return []
    chunks = []
    i = 0
    while i < len(word_subs):
        chunk_words = []
        chunk_start = word_subs[i][0][0]
        chunk_end = word_subs[i][0][1]
        for j in range(max_words):
            if i + j >= len(word_subs):
                break
            chunk_words.append(word_subs[i + j][1])
            chunk_end = word_subs[i + j][0][1]
        chunks.append(((chunk_start, chunk_end), " ".join(chunk_words)))
        i += len(chunk_words)
    return chunks


# =============================================
# BATCH UPLOAD TO GEMINI
# =============================================
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
        voiceover_result = None
        if voiceover_url and voiceover_url.strip():
            vo_path = f"/tmp/vo_{uuid.uuid4().hex[:8]}.mp3"
            if download_file(voiceover_url, vo_path):
                try:
                    vo_file = genai.upload_file(vo_path, mime_type="audio/mpeg")
                    voiceover_result = {"file_uri": vo_file.uri, "name": vo_file.name}
                except Exception as ve:
                    print(f"Voiceover upload failed: {ve}")
                finally:
                    if os.path.exists(vo_path):
                        os.remove(vo_path)

        # Upload videos to GCS + Gemini
        gcs_client = gcs_storage.Client()
        gcs_bucket = gcs_client.bucket(GCS_BUCKET)

        for i, url in enumerate(urls):
            try:
                local_path = f"/tmp/video_{i+1:02d}.mp4"
                if download_file(url, local_path):
                    # Upload to GCS
                    try:
                        blob = gcs_bucket.blob(f"temp-stock-videos/video_{i+1:02d}.mp4")
                        blob.upload_from_filename(local_path, content_type="video/mp4")
                    except Exception as ge:
                        print(f"GCS upload failed: {ge}")

                    # Upload to Gemini
                    display_name = f"video_{i+1:02d}.mp4"
                    uploaded = genai.upload_file(local_path, mime_type="video/mp4", display_name=display_name)
                    uploaded_files.append({
                        "file_uri": uploaded.uri,
                        "name": uploaded.name,
                        "gcs_url": f"https://storage.googleapis.com/{GCS_BUCKET}/temp-stock-videos/video_{i+1:02d}.mp4"
                    })
                    os.remove(local_path)
                    print(f"Uploaded {i+1}/{len(urls)}")
                else:
                    uploaded_files.append({"file_uri": "", "name": "", "error": f"Download failed"})
            except Exception as e:
                uploaded_files.append({"file_uri": "", "name": "", "error": str(e)})

        # Poll for ACTIVE state (instead of hard sleep)
        print("Polling for ACTIVE state...")
        for poll in range(18):
            all_active = True
            for f_info in uploaded_files:
                if not f_info.get('file_uri'):
                    continue
                try:
                    f = genai.get_file(f_info['name'])
                    if f.state.name != 'ACTIVE':
                        all_active = False
                        break
                except:
                    pass
            if all_active:
                print(f"All files ACTIVE after {(poll+1)*10}s")
                break
            if poll < 17:
                time.sleep(10)

        return jsonify({
            "status": "success",
            "files": uploaded_files,
            "voiceover": voiceover_result
        })

    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500


# =============================================
# UPLOAD AUDIO (base64 -> hosted file)
# =============================================
@app.route('/upload_audio', methods=['POST'])
def handle_audio_upload():
    data = request.json
    audio_b64 = data.get('audio_base64', '')
    fmt = data.get('format', 'mp3')
    if not audio_b64:
        return jsonify({"error": "Missing audio_base64"}), 400
    try:
        audio_id = uuid.uuid4().hex[:12]
        audio_path = f"{OUTPUT_DIR}/audio_{audio_id}.{fmt}"
        with open(audio_path, 'wb') as f:
            f.write(base64.b64decode(audio_b64))
        return jsonify({
            "status": "success",
            "audio_id": audio_id,
            "url": f"{request.host_url.rstrip('/')}/audio/{audio_id}"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/audio/<audio_id>', methods=['GET'])
def serve_audio(audio_id):
    for ext in ['mp3', 'wav', 'aac']:
        path = f"{OUTPUT_DIR}/audio_{audio_id}.{ext}"
        if os.path.exists(path):
            return send_file(path, mimetype='audio/mpeg')
    return jsonify({"error": "Audio not found"}), 404


# =============================================
# PURE FFmpeg RENDER PIPELINE
# =============================================
def process_timeline_job(data, webhook_url):
    """
    Render pipeline:
    1. Download clips in parallel
    2. FFmpeg: extract subclips + scale/crop to 1080x1920 + concat
    3. Whisper: transcribe voiceover -> subtitles
    4. FFmpeg: burn subtitles using ASS format
    5. FFmpeg: mix audio (voiceover + music)
    6. FFmpeg: mux final video + audio
    7. Upload to GCS
    8. Webhook callback
    """
    job_start_time = time.time()
    print("=== RENDER JOB START ===")
    base_dir = f"/tmp/render_{uuid.uuid4().hex[:8]}"
    os.makedirs(base_dir, exist_ok=True)
    user_filename = data.get('filename', 'video').strip().replace(" ", "_")
    if not user_filename.lower().endswith('.mp4'):
        user_filename += ".mp4"
    output_path = f"{base_dir}/{user_filename}"

    try:
        audio_url = data.get('audio_url', '')
        music_url = data.get('music_url')
        edl = data.get('timeline', [])
        script_hint = data.get('script', '')

        # --- Step 1: Download voiceover + determine duration ---
        has_voiceover = False
        local_audio = f"{base_dir}/voiceover.mp3"
        final_duration = 0.0

        if audio_url and audio_url.strip():
            if download_file(audio_url, local_audio):
                has_voiceover = True
                final_duration = get_video_duration(local_audio)
                print(f"Voiceover: {final_duration:.1f}s")

        if not has_voiceover:
            final_duration = max([float(c.get('timeline_end', 0)) for c in edl]) if edl else 30.0

        # --- Step 2: Download all clips in parallel ---
        asset_map = {}
        unique_urls = []
        for cut in edl:
            url = cut.get('url')
            if url and url not in asset_map:
                l_path = f"{base_dir}/src_{len(asset_map)}.mp4"
                asset_map[url] = l_path
                unique_urls.append((url, l_path))

        def dl_task(args):
            return download_file(args[0], args[1])

        print(f"Downloading {len(unique_urls)} clips...")
        dl_start = time.time()
        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(dl_task, unique_urls))

        for (url, path), result in zip(unique_urls, results):
            if not result:
                del asset_map[url]
        print(f"Downloads done: {len(asset_map)} clips in {time.time()-dl_start:.1f}s")

        # --- Step 3: FFmpeg extract + scale each segment ---
        print(f"Extracting {len(edl)} segments...")
        extract_start = time.time()
        segment_paths = []

        for idx, cut in enumerate(edl):
            url = cut.get('url')
            t_start = float(cut.get('timeline_start', 0))
            t_end = float(cut.get('timeline_end', 0))
            s_start = float(cut.get('source_start', 0))
            duration = t_end - t_start
            if duration <= 0 or url not in asset_map:
                continue

            seg_path = f"{base_dir}/seg_{idx:03d}.mp4"

            # Clamp source_start to actual duration
            src_dur = get_video_duration(asset_map[url])
            if src_dur > 0 and s_start + duration > src_dur:
                s_start = max(0, src_dur - duration)

            # Single FFmpeg command: seek + scale + crop + re-encode (ultrafast)
            cmd = [
                "ffmpeg", "-y",
                "-ss", str(max(0, s_start)),
                "-i", asset_map[url],
                "-t", str(duration),
                "-vf", "scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920",
                "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23",
                "-r", str(RENDER_FPS),
                "-an", "-sn",
                "-pix_fmt", "yuv420p",
                seg_path
            ]
            try:
                result = subprocess.run(cmd, capture_output=True, timeout=120)
                if os.path.exists(seg_path) and os.path.getsize(seg_path) > 0:
                    segment_paths.append(seg_path)
                else:
                    print(f"  Segment {idx} failed (empty output)")
            except subprocess.TimeoutExpired:
                print(f"  Segment {idx} timed out, skipping")
            except Exception as seg_err:
                print(f"  Segment {idx} error: {seg_err}")

        if not segment_paths:
            raise Exception("No segments extracted!")
        print(f"Extraction done: {len(segment_paths)} segments in {time.time()-extract_start:.1f}s")

        # --- Step 4: FFmpeg concat all segments ---
        print("Concatenating segments...")
        concat_start = time.time()
        concat_list = f"{base_dir}/concat.txt"
        with open(concat_list, 'w') as f:
            for seg in segment_paths:
                f.write(f"file '{seg}'\n")

        video_only = f"{base_dir}/video_only.mp4"
        result = subprocess.run([
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0",
            "-i", concat_list,
            "-c", "copy",
            "-movflags", "+faststart",
            video_only
        ], capture_output=True, text=True, timeout=120)

        if result.returncode != 0:
            # Fallback: re-encode concat
            result = subprocess.run([
                "ffmpeg", "-y",
                "-f", "concat", "-safe", "0",
                "-i", concat_list,
                "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23",
                "-r", str(RENDER_FPS),
                "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
                video_only
            ], capture_output=True, text=True, timeout=120)

        print(f"Concat done in {time.time()-concat_start:.1f}s")

        # --- Step 5: Whisper transcription -> ASS subtitles ---
        ass_path = None
        if has_voiceover:
            print("Transcribing voiceover...")
            whisper_start = time.time()
            segments_iter, _ = WHISPER_MODEL.transcribe(
                local_audio, word_timestamps=True,
                initial_prompt=script_hint, beam_size=5
            )
            word_subs = []
            for seg in segments_iter:
                for w in seg.words:
                    clean = w.word.strip().upper().replace('"', '').replace('.', '').replace(',', '')
                    if clean:
                        word_subs.append(((w.start, w.end), clean))
            print(f"Transcription: {len(word_subs)} words in {time.time()-whisper_start:.1f}s")

            # Group into 3-word chunks
            chunks = group_words_into_chunks(word_subs, max_words=3)
            # Fill gaps
            filled = []
            for j in range(len(chunks)):
                start, end = chunks[j][0]
                text = chunks[j][1]
                if j < len(chunks) - 1:
                    next_start = chunks[j+1][0][0]
                    if next_start > end:
                        end = next_start
                filled.append((start, end, text))

            # Generate ASS subtitle file
            ass_path = f"{base_dir}/subs.ass"
            _generate_ass_file(ass_path, filled)

        # --- Step 6: Burn subtitles into video ---
        if ass_path and os.path.exists(ass_path):
            print("Burning subtitles...")
            sub_start = time.time()
            video_with_subs = f"{base_dir}/video_subs.mp4"
            result = subprocess.run([
                "ffmpeg", "-y",
                "-i", video_only,
                "-vf", f"ass={ass_path}",
                "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23",
                "-r", str(RENDER_FPS),
                "-pix_fmt", "yuv420p",
                "-an",
                video_with_subs
            ], capture_output=True, text=True, timeout=180)

            if result.returncode == 0:
                video_only = video_with_subs
                print(f"Subtitles burned in {time.time()-sub_start:.1f}s")
            else:
                print(f"Subtitle burn failed: {result.stderr[:200]}")

        # --- Step 7: Mix audio ---
        local_music = f"{base_dir}/music.mp3"
        has_music = False
        if music_url:
            if download_file(music_url, local_music):
                has_music = True

        mixed_audio = f"{base_dir}/mixed.aac"
        has_mixed_audio = False

        if has_voiceover and has_music:
            print("Mixing voiceover + music...")
            result = subprocess.run([
                "ffmpeg", "-y",
                "-i", local_audio,
                "-stream_loop", "-1", "-i", local_music,
                "-filter_complex",
                "[1:a]volume=0.05[music];[0:a][music]amix=inputs=2:duration=first:dropout_transition=2:normalize=0[out]",
                "-map", "[out]",
                "-c:a", "aac", "-b:a", "192k",
                "-t", str(final_duration),
                mixed_audio
            ], capture_output=True, text=True, timeout=60)
            has_mixed_audio = result.returncode == 0
        elif has_voiceover:
            result = subprocess.run([
                "ffmpeg", "-y", "-i", local_audio,
                "-c:a", "aac", "-b:a", "192k", mixed_audio
            ], capture_output=True, timeout=30)
            has_mixed_audio = result.returncode == 0
        elif has_music:
            result = subprocess.run([
                "ffmpeg", "-y", "-i", local_music,
                "-af", "volume=0.3",
                "-c:a", "aac", "-b:a", "192k",
                "-t", str(final_duration),
                mixed_audio
            ], capture_output=True, timeout=30)
            has_mixed_audio = result.returncode == 0

        # --- Step 8: Mux video + audio ---
        if has_mixed_audio:
            print("Muxing video + audio...")
            result = subprocess.run([
                "ffmpeg", "-y",
                "-i", video_only,
                "-i", mixed_audio,
                "-c:v", "copy", "-c:a", "copy",
                "-shortest",
                "-movflags", "+faststart",
                output_path
            ], capture_output=True, text=True, timeout=60)
            if result.returncode != 0:
                shutil.copy(video_only, output_path)
        else:
            shutil.copy(video_only, output_path)

        # --- Step 9: Upload to GCS ---
        print("Uploading to GCS...")
        gcs_client = gcs_storage.Client()
        bucket = gcs_client.bucket(GCS_BUCKET)
        blob = bucket.blob(f"generated_videos/{user_filename}")
        blob.upload_from_filename(output_path)

        credentials, _ = google.auth.default()
        auth_req = gauth_requests.Request()
        credentials.refresh(auth_req)
        sa_email = getattr(credentials, 'service_account_email', SERVICE_ACCOUNT_EMAIL)

        download_url = blob.generate_signed_url(
            version="v4", expiration=datetime.timedelta(hours=1), method="GET",
            service_account_email=sa_email, access_token=credentials.token
        )

        # --- Done ---
        processing_time = time.time() - job_start_time
        file_size_mb = round(os.path.getsize(output_path) / (1024 * 1024), 2)

        webhook_payload = {
            "status": "success",
            "video_url": download_url,
            "gcs_url": f"https://storage.googleapis.com/{GCS_BUCKET}/generated_videos/{user_filename}",
            "filename": user_filename,
            "duration_seconds": round(final_duration, 2),
            "processing_time_seconds": round(processing_time, 2),
            "file_size_mb": file_size_mb,
            "resolution": "1080x1920",
            "fps": RENDER_FPS,
            "has_audio": has_mixed_audio
        }

        print(f"=== RENDER DONE: {user_filename} ({file_size_mb}MB) in {round(processing_time)}s ===")
        if webhook_url:
            requests.post(webhook_url, json=webhook_payload, timeout=10)

    except Exception as e:
        print(f"RENDER ERROR: {e}")
        traceback.print_exc()
        if webhook_url:
            try:
                requests.post(webhook_url, json={"status": "error", "error": str(e)}, timeout=10)
            except:
                pass
    finally:
        try:
            shutil.rmtree(base_dir)
        except:
            pass


def _generate_ass_file(ass_path, subtitle_entries):
    """Generate ASS subtitle file with Poppins styling.
    subtitle_entries: list of (start_sec, end_sec, text)
    """
    def _format_ass_time(seconds):
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = seconds % 60
        return f"{h}:{m:02d}:{s:05.2f}"

    header = """[Script Info]
Title: CashFlow Shorts Subtitles
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
WrapStyle: 0

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Poppins,90,&H00FFFFFF,&H000000FF,&H00603400,&H00000000,-1,0,0,0,100,100,0,0,1,5,0,2,40,40,350,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    with open(ass_path, 'w') as f:
        f.write(header)
        for start, end, text in subtitle_entries:
            start_str = _format_ass_time(start)
            end_str = _format_ass_time(end)
            f.write(f"Dialogue: 0,{start_str},{end_str},Default,,0,0,0,,{text}\n")


# =============================================
# ROUTES
# =============================================
@app.route('/', methods=['POST'])
def handle_request():
    data = request.json
    if not data:
        return jsonify({"error": "Invalid or missing JSON body"}), 400
    webhook_url = data.get('webhook_url')
    if not webhook_url:
        return jsonify({"error": "Missing 'webhook_url'"}), 400
    threading.Thread(target=process_timeline_job, args=(data, webhook_url)).start()
    return jsonify({"message": "Job started."}), 202


@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
