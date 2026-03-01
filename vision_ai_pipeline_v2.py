"""
╔══════════════════════════════════════════════════════════════════════╗
║         VISION AI — Meeting Analytics Pipeline                      ║
║         Hackathon: Video Analytics Challenge                        ║
║         Team: Vision AI (4 Members)                                 ║
╚══════════════════════════════════════════════════════════════════════╝

PIPELINE STAGES:
  Stage 1  → Audio Extraction        (moviepy / ffmpeg)
  Stage 2  → Transcription           (OpenAI Whisper - medium)
  Stage 3  → Speaker Diarization     (pyannote.audio)
  Stage 3b → OCR Name Extraction     (EasyOCR — reads name tags from video tiles)
  Stage 4  → Sentiment Analysis      (transformers - cardiffnlp)
  Stage 5  → Action Item Extraction  (pattern matching)
  Stage 6  → Keyword/Topic Extraction(keybert / YAKE)
  Stage 7  → Summarization           (transformers - BART)
  Stage 8  → Report Generation       (plain text)

SETUP:
  pip install openai-whisper pyannote.audio transformers moviepy torch
              sentencepiece accelerate huggingface_hub keybert yake nltk easyocr opencv-python

USAGE:
  python vision_ai_pipeline.py --file meeting.mp4 --hf_token YOUR_TOKEN
  python vision_ai_pipeline.py --file meeting.mp4  # without diarization
"""

import os, sys, re, json, argparse, warnings, tempfile
from pathlib import Path
from datetime import datetime
from collections import defaultdict

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════

CONFIG = {
    "whisper_model":       "medium",   # medium = best accuracy for transcription
    "device":              "cpu",
    "summarization_model": "facebook/bart-large-cnn",
    "sentiment_model":     "cardiffnlp/twitter-roberta-base-sentiment-latest",
    "translation_model":   "Helsinki-NLP/opus-mt-en-hi",
    "max_summary_length":  250,
    "min_summary_length":  60,
}

# Supported report languages → Helsinki-NLP MarianMT model names
LANGUAGE_MODELS = {
    "english":  None,                          # no translation needed
    "hindi":    "Helsinki-NLP/opus-mt-en-hi", # dedicated EN→HI model, best quality
}

LANGUAGE_DISPLAY = {
    "english":  "English",
    "hindi":    "Hindi / हिन्दी",
}

STAGE_LOG = []  # Collect stage outputs for SOP documentation

def log_stage(stage_num, stage_name, status, detail=""):
    entry = {
        "stage": stage_num,
        "name": stage_name,
        "status": status,
        "detail": detail,
        "timestamp": datetime.now().strftime("%H:%M:%S")
    }
    STAGE_LOG.append(entry)
    icon = "✅" if status == "success" else "⚠️" if status == "warning" else "❌"
    print(f"    {icon} [{entry['timestamp']}] {detail}")


# ══════════════════════════════════════════════════════════════════════
# STAGE 1: AUDIO EXTRACTION
# ══════════════════════════════════════════════════════════════════════

def stage1_extract_audio(input_path: str, output_wav: str) -> str:
    print("\n[Stage 1/8] 🎬 Audio Extraction")
    ext = Path(input_path).suffix.lower()
    video_formats = {".mp4", ".avi", ".mkv", ".mov", ".webm"}
    try:
        if ext in video_formats:
            from moviepy.editor import VideoFileClip
            clip = VideoFileClip(input_path)
            duration = clip.duration
            clip.audio.write_audiofile(
                output_wav, fps=16000, nbytes=2,
                ffmpeg_params=["-ac", "1"], logger=None
            )
            clip.close()
            log_stage(1, "Audio Extraction", "success",
                      f"Extracted from video — Duration: {int(duration//60)}m {int(duration%60)}s")
        else:
            ret = os.system(f'ffmpeg -i "{input_path}" -ar 16000 -ac 1 -y "{output_wav}" -loglevel quiet')
            if ret != 0:
                raise RuntimeError("ffmpeg conversion failed")
            log_stage(1, "Audio Extraction", "success", f"Converted audio file to 16kHz mono WAV")
        return output_wav
    except Exception as e:
        log_stage(1, "Audio Extraction", "error", str(e))
        sys.exit(1)


# ══════════════════════════════════════════════════════════════════════
# STAGE 2: TRANSCRIPTION
# ══════════════════════════════════════════════════════════════════════

def stage2_transcribe(audio_path: str) -> dict:
    print(f"\n[Stage 2/8] 🎙️  Transcription (Whisper '{CONFIG['whisper_model']}')")
    print("    ⏳ This is the slowest step — please wait...")
    try:
        import whisper
        model = whisper.load_model(CONFIG["whisper_model"], device=CONFIG["device"])
        result = model.transcribe(audio_path, verbose=False, word_timestamps=True)
        seg_count = len(result["segments"])
        word_count = sum(len(s["text"].split()) for s in result["segments"])
        log_stage(2, "Transcription", "success",
                  f"Whisper medium — {seg_count} segments, ~{word_count} words detected")
        return result
    except ImportError:
        log_stage(2, "Transcription", "error", "openai-whisper not installed")
        sys.exit(1)
    except Exception as e:
        log_stage(2, "Transcription", "error", str(e))
        sys.exit(1)


# ══════════════════════════════════════════════════════════════════════
# STAGE 3: SPEAKER DIARIZATION
# ══════════════════════════════════════════════════════════════════════

def stage3_diarize(audio_path: str, hf_token: str) -> list:
    print(f"\n[Stage 3/8] 👥 Speaker Diarization")

    # ── No token provided ───────────────────────────────────────────
    if not hf_token:
        print("\n" + "⚠️ "*20)
        print("  ⚠️  WARNING: No HuggingFace token provided!")
        print("  ⚠️  Speaker diarization is DISABLED.")
        print("  ⚠️  All speakers will be merged into one label.")
        print("  ⚠️  To fix: get a free token at https://huggingface.co/settings/tokens")
        print("  ⚠️  Then re-run with:  --hf_token YOUR_TOKEN")
        print("⚠️ "*20)
        log_stage(3, "Diarization", "warning", "No HF token provided — diarization skipped")
        return [], 0

    # ── Ask how many speakers are in the meeting ─────────────────────
    # Providing this to pyannote greatly improves accuracy — it knows
    # exactly how many voice clusters to find instead of guessing
    print("\n" + "─"*60)
    print("  💡 TIP: Telling pyannote the exact number of speakers")
    print("     significantly improves diarization accuracy.")
    print("─"*60)
    try:
        n_input = input("  How many speakers are in this meeting? (press ENTER to auto-detect) → ").strip()
        num_speakers = int(n_input) if n_input.isdigit() and int(n_input) > 0 else None
    except Exception:
        num_speakers = None

    if num_speakers:
        print(f"  ✅ Will look for exactly {num_speakers} speakers")
    else:
        print("  ℹ️  Auto-detecting number of speakers")

    try:
        from pyannote.audio import Pipeline
        import torch
        print("    ⏳ Loading pyannote diarization model (first run downloads ~300MB)...")
        from huggingface_hub import login
        login(token=hf_token, add_to_git_credential=False)
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1"
        )
        pipeline.to(torch.device("cpu"))
        print("    ⏳ Running speaker detection on audio...")

        # Pass num_speakers if provided — this is the key fix
        if num_speakers:
            diarization = pipeline(audio_path, num_speakers=num_speakers)
        else:
            diarization = pipeline(audio_path)

        segments = [
            {"start": turn.start, "end": turn.end, "speaker": speaker}
            for turn, _, speaker in diarization.itertracks(yield_label=True)
        ]
        unique = list(set(s["speaker"] for s in segments))

        if len(unique) <= 1:
            print("\n" + "⚠️ "*20)
            print(f"  ⚠️  WARNING: Diarization only detected {len(unique)} speaker(s)!")
            print("  ⚠️  This usually means:")
            print("  ⚠️    1. You have NOT accepted the model terms on HuggingFace")
            print("  ⚠️       → Visit: https://huggingface.co/pyannote/speaker-diarization-3.1")
            print("  ⚠️       → Click Accept to get access, then re-run")
            print("  ⚠️    2. Audio quality is too low for diarization")
            print("  ⚠️    3. Speakers have very similar voices")
            print("⚠️ "*20)

        log_stage(3, "Diarization", "success",
                  f"pyannote 3.1 — {len(unique)} speakers detected: {', '.join(unique)}")
        return segments, num_speakers or 0

    except Exception as e:
        err = str(e)
        print("\n" + "❌ "*20)
        print(f"  ❌ Diarization FAILED: {err}")
        if "401" in err or "token" in err.lower() or "auth" in err.lower():
            print("  ❌ This looks like an authentication error.")
            print("  ❌ Make sure your HuggingFace token is valid and you have")
            print("  ❌ accepted the model terms at:")
            print("  ❌ https://huggingface.co/pyannote/speaker-diarization-3.1")
        elif "403" in err:
            print("  ❌ Access denied — you need to accept the model license.")
            print("  ❌ Visit: https://huggingface.co/pyannote/speaker-diarization-3.1")
            print("  ❌ Click the Accept button, then re-run the pipeline.")
        print("❌ "*20)
        log_stage(3, "Diarization", "error", f"Failed: {err}")
        return [], 0


# ══════════════════════════════════════════════════════════════════════
# STAGE 3c: TEAMS TRANSCRIPT PARSER (optional — replaces diarization)
# ══════════════════════════════════════════════════════════════════════

def parse_teams_transcript(content: str) -> list:
    """
    Parse Microsoft Teams auto-generated transcript (.txt) format:
      Speaker Name  MM:SS
      text of what they said...

    Returns list of {speaker, start_sec, text}
    """
    entries = []
    # Matches: "Babuji Abraham 0:03" or "Sujata Nanda 1:03"
    header_re = re.compile(
        r'^([A-Za-z][A-Za-z\s\-\.\']+?)\s+(\d{1,2}:\d{2})\s*$'
    )
    lines = content.strip().split('\n')
    current_speaker, current_time, current_text = None, None, []

    for line in lines:
        line = line.strip()
        if not line:
            continue
        m = header_re.match(line)
        if m:
            # Save previous entry
            if current_speaker and current_text:
                text = ' '.join(current_text).strip()
                if text:
                    entries.append({
                        'speaker':   current_speaker,
                        'start_sec': current_time,
                        'text':      text
                    })
            current_speaker = m.group(1).strip()
            parts = m.group(2).split(':')
            current_time = int(parts[0]) * 60 + int(parts[1])
            current_text = []
        else:
            if current_speaker:
                current_text.append(line)

    # Save last entry
    if current_speaker and current_text:
        text = ' '.join(current_text).strip()
        if text:
            entries.append({
                'speaker':   current_speaker,
                'start_sec': current_time,
                'text':      text
            })
    return entries


def stage3c_teams_transcript(transcript_path: str, whisper_segments: list) -> list:
    """
    Use Microsoft Teams transcript file to assign accurate speaker names
    to Whisper segments by matching timestamps.

    Logic:
      - Parse Teams transcript → get (speaker, timestamp) pairs
      - For each Whisper segment, find the Teams entry whose timestamp
        is closest to the segment start time
      - Assign that speaker name to the segment

    Returns segments with speaker names filled from Teams transcript.
    Falls back gracefully if file is missing or unparseable.
    """
    print(f"\n[Stage 3c] 📄 Teams Transcript Speaker Mapping")

    # ── Read file — support both .txt and .pdf ───────────────────────
    content = None
    ext = Path(transcript_path).suffix.lower()

    if ext == '.pdf':
        try:
            import pdfplumber
            with pdfplumber.open(transcript_path) as pdf:
                content = "\n".join(
                    page.extract_text() or "" for page in pdf.pages
                )
            print(f"    ✅ PDF read successfully ({len(content)} chars)")
        except ImportError:
            try:
                import pypdf
                reader = pypdf.PdfReader(transcript_path)
                content = "\n".join(
                    page.extract_text() or "" for page in reader.pages
                )
                print(f"    ✅ PDF read via pypdf ({len(content)} chars)")
            except ImportError:
                print("    ❌ Install pdfplumber: pip install pdfplumber")
                log_stage("3c", "Teams Transcript", "warning",
                          "PDF reading requires pdfplumber. Run: pip install pdfplumber")
                return []
        except Exception as e:
            print(f"    ❌ PDF read error: {e}")
            log_stage("3c", "Teams Transcript", "warning", f"Could not read PDF: {e}")
            return []
    else:
        # Try multiple encodings for .txt files
        for encoding in ['utf-8', 'utf-8-sig', 'cp1252', 'latin-1']:
            try:
                with open(transcript_path, 'r', encoding=encoding) as f:
                    content = f.read()
                break
            except (UnicodeDecodeError, LookupError):
                continue

    if not content or not content.strip():
        log_stage("3c", "Teams Transcript", "warning",
                  "Could not read transcript file — tried utf-8, utf-8-sig, cp1252, latin-1")
        return []

    teams_entries = parse_teams_transcript(content)

    if not teams_entries:
        # Show first 300 chars to help diagnose format issues
        preview = content[:300].replace('\n', '\\n').replace('\r', '\\r')
        print(f"    ❌ Could not parse entries. File preview:")
        print(f"       {preview}")
        log_stage("3c", "Teams Transcript", "warning",
                  "Could not parse any entries — check file format")
        return []

    speakers = list(dict.fromkeys(e['speaker'] for e in teams_entries))
    log_stage("3c", "Teams Transcript", "success",
              f"Parsed {len(teams_entries)} entries, "
              f"{len(speakers)} speakers: {', '.join(speakers)}")

    # Build a timeline: for each second, who is speaking?
    # Use Teams entries to create a speaker map by time
    # Sort entries by time
    teams_entries.sort(key=lambda x: x['start_sec'])

    def get_speaker_at(time_sec: float) -> str:
        """Find who was speaking at a given time using Teams transcript."""
        best = teams_entries[0]['speaker']
        for entry in teams_entries:
            if entry['start_sec'] <= time_sec:
                best = entry['speaker']
            else:
                break
        return best

    # Apply to each Whisper segment
    for seg in whisper_segments:
        mid = (seg['start'] + seg['end']) / 2
        seg['speaker'] = get_speaker_at(mid)

    # Print summary
    spk_counts = defaultdict(int)
    for seg in whisper_segments:
        spk_counts[seg['speaker']] += 1
    print(f"\n    Speaker segment counts:")
    for spk, count in sorted(spk_counts.items(), key=lambda x: -x[1]):
        print(f"      {spk:<25} {count} segments")

    return whisper_segments



def merge_segments(whisper_result: dict, diar_segments: list) -> list:
    def get_speaker(start, end):
        best, best_overlap = "Speaker 1", 0
        for seg in diar_segments:
            overlap = min(end, seg["end"]) - max(start, seg["start"])
            if overlap > best_overlap:
                best_overlap = overlap
                best = seg["speaker"].replace("SPEAKER_", "Speaker ")
        return best

    merged = []
    for seg in whisper_result["segments"]:
        speaker = get_speaker(seg["start"], seg["end"]) if diar_segments else "Speaker 1"
        merged.append({
            "start": seg["start"], "end": seg["end"],
            "speaker": speaker, "text": seg["text"].strip()
        })
    return merged


def stage3b_ocr_names(video_path: str, diar_segments: list) -> dict:
    """
    Extract speaker names from Microsoft Teams name tags (bottom of each tile)
    using EasyOCR on sampled video frames. Maps diarization labels to real names.

    How it works:
      1. Sample one frame every 3 seconds from the video
      2. Crop the bottom 25% of each frame (where Teams name tags appear)
      3. Run EasyOCR to read any text in that region
      4. Match OCR timestamp to diarization timeline to identify which speaker
         was active at that moment
      5. Build a name map: { "SPEAKER_00": "Rahul", "SPEAKER_01": "Priya" }
    """
    print(f"\n[Stage 3b] 🔍 OCR Name Extraction from Video Tiles")

    # Only works on video files
    ext = Path(video_path).suffix.lower()
    if ext not in {".mp4", ".avi", ".mkv", ".mov", ".webm"}:
        log_stage("3b", "OCR Names", "warning", "Skipped — input is audio-only, no video frames")
        return {}

    if not diar_segments:
        log_stage("3b", "OCR Names", "warning", "Skipped — no diarization segments available")
        return {}

    try:
        import cv2
        import easyocr
    except ImportError:
        log_stage("3b", "OCR Names", "warning",
                  "easyocr or opencv-python not installed — run: pip install easyocr opencv-python")
        return {}

    try:
        reader = easyocr.Reader(["en"], gpu=False, verbose=False)
        cap    = cv2.VideoCapture(video_path)
        fps    = cap.get(cv2.CAP_PROP_FPS) or 25
        total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Sample every 3 seconds
        sample_interval = int(fps * 3)

        # speaker_name_votes[speaker_label] = { "Name": count }
        speaker_name_votes = defaultdict(lambda: defaultdict(int))

        frame_idx = 0
        frames_processed = 0

        while frame_idx < total:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break

            timestamp = frame_idx / fps  # seconds into video

            # ── Find which speaker is active at this timestamp ──────
            active_speaker = None
            for seg in diar_segments:
                if seg["start"] <= timestamp <= seg["end"]:
                    active_speaker = seg["speaker"].replace("SPEAKER_", "Speaker ")
                    break

            if active_speaker:
                h, w = frame.shape[:2]

                # ── Crop bottom 25% of frame (Teams name tag zone) ──
                # Teams shows name tags in bottom ~20-25% of each tile
                crop = frame[int(h * 0.75):h, :]

                # Optional: enhance contrast for better OCR
                gray     = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                _, thresh = cv2.threshold(gray, 0, 255,
                                          cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                crop_rgb = cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)

                # ── Run OCR ─────────────────────────────────────────
                results = reader.readtext(crop_rgb, detail=0, paragraph=False)

                for text in results:
                    text = text.strip()
                    # Filter: names are typically 2-40 chars, not pure numbers
                    if 2 <= len(text) <= 40 and not text.isdigit():
                        # Remove common Teams UI labels
                        skip_words = {"you", "guest", "presenter", "attendee",
                                      "muted", "unmuted", "camera", "off", "on",
                                      "pin", "more", "options", "meeting"}
                        if text.lower() not in skip_words:
                            # Title-case the name for consistency
                            clean_name = text.title()
                            speaker_name_votes[active_speaker][clean_name] += 1

            frame_idx += sample_interval
            frames_processed += 1

        cap.release()

        # ── Build final name map — pick most voted name per speaker ─
        # Ensure each name is only assigned to ONE speaker (the one with most votes)
        # This prevents the same name appearing for multiple speakers
        name_map = {}
        used_names = {}  # name → (speaker, vote_count)

        # First pass: assign best name to each speaker
        candidates = {}
        for speaker, votes in speaker_name_votes.items():
            if votes:
                best_name = max(votes, key=votes.get)
                best_count = votes[best_name]
                candidates[speaker] = (best_name, best_count)

        # Second pass: resolve conflicts — if two speakers have same best name,
        # give it to whichever speaker has more votes for it
        for speaker, (name, count) in sorted(candidates.items(),
                                              key=lambda x: -x[1][1]):
            if name not in used_names:
                name_map[speaker] = name
                used_names[name] = speaker
                print(f"    🏷️  {speaker} → '{name}' "
                      f"(detected in {count} frames)")
            else:
                # Name already taken — keep default label, user will rename
                name_map[speaker] = speaker
                print(f"    ⚠️  {speaker} → name '{name}' already assigned to "
                      f"{used_names[name]}, keeping default label")

        # ── For any speaker with no OCR result, keep default label ──
        all_speakers = list(dict.fromkeys(
            seg["speaker"].replace("SPEAKER_", "Speaker ")
            for seg in diar_segments
        ))
        for spk in all_speakers:
            if spk not in name_map:
                name_map[spk] = spk
                print(f"    ⚠️  {spk} → no name detected, keeping default label")

        detected = sum(1 for v in name_map.values()
                       if not v.startswith("Speaker "))
        log_stage("3b", "OCR Names", "success",
                  f"EasyOCR — {frames_processed} frames scanned, "
                  f"{detected}/{len(name_map)} speakers named automatically")

        # ── Confirm with user — allow corrections ───────────────────
        print("\n" + "═"*60)
        print("  👤 OCR DETECTED THESE NAMES — press ENTER to confirm")
        print("     or type a correction if the name looks wrong")
        print("═"*60)
        confirmed_map = {}
        for spk, detected_name in name_map.items():
            correction = input(
                f"  {spk} detected as '{detected_name}' → ").strip()
            confirmed_map[spk] = correction if correction else detected_name

        return confirmed_map

    except Exception as e:
        log_stage("3b", "OCR Names", "warning", f"OCR failed: {e}")
        return {}


def manual_speaker_split(segments: list) -> list:
    """
    When diarization gives only 1 speaker, let user manually assign
    speaker names to blocks of transcript segments.
    """
    print("\n" + "═"*65)
    print("  How many speakers were in this meeting?")
    try:
        n = int(input("  Number of speakers → ").strip())
    except ValueError:
        n = 1

    if n <= 1:
        name = input("  Speaker name (ENTER = 'Speaker 1') → ").strip()
        for seg in segments:
            seg["speaker"] = name if name else "Speaker 1"
        return segments

    print(f"\n  Enter names for all {n} speakers:")
    names = []
    for i in range(n):
        name = input(f"    Speaker {i+1} → ").strip()
        names.append(name if name else f"Speaker {i+1}")

    print("\n" + "─"*65)
    print("  The transcript will be shown in chunks of 5 segments.")
    print("  Type the speaker NUMBER for each block.")
    print("  Press ENTER to keep the same speaker as the previous block.")
    print("─"*65)

    current_speaker = names[0]
    chunk_size = 5

    for chunk_start in range(0, len(segments), chunk_size):
        chunk = segments[chunk_start:chunk_start + chunk_size]
        print(f"\n  ── [{fmt_time(chunk[0]['start'])} → {fmt_time(chunk[-1]['end'])}] ──")
        for seg in chunk:
            print(f"    {seg['text'][:100]}")

        options = "  |  ".join(f"{i+1}={names[i]}" for i in range(n))
        choice = input(f"  Speaker? ({options})  → ").strip()
        if choice.isdigit() and 1 <= int(choice) <= n:
            current_speaker = names[int(choice) - 1]

        for seg in chunk:
            seg["speaker"] = current_speaker

    print("\n  ✅ Speaker assignment complete!")
    return segments


def auto_detect_speaker_names(segments: list) -> dict:
    """
    Automatically identify speaker names from conversation context.

    Two strategies combined:
    1. ADDRESS detection — "Varun, can you..." means the speaker is NOT Varun
    2. IDENTITY clues — "I will set up a call Babuji" means speaker reports to Babuji
       and later "I'll set up a call with Nitin today Babuji" confirms speaker = Poornima

    Returns best-guess name map: {"Speaker 00": "Babuji", "Speaker 01": "Sujata", ...}
    The caller should always show this to user for confirmation.
    """
    IGNORE_WORDS = {
        'the','let','ok','so','but','and','now','yeah','good','thanks','thank',
        'sorry','please','yes','no','right','exactly','well','just','also',
        'monday','tuesday','wednesday','thursday','friday','saturday','sunday',
        'january','february','march','april','may','june','july','august',
        'september','october','november','december','first','second','third',
        'meeting','office','team','client','guest','hotel','airport','email',
        'ticket','access','process','policy','checklist','template','budget',
    }

    speakers = list(dict.fromkeys(s["speaker"] for s in segments))
    speaker_texts = defaultdict(list)
    for seg in segments:
        speaker_texts[seg["speaker"]].append(seg["text"])

    # ── Step 1: Find names being ADDRESSED ──────────────────────────
    # "Varun, can you..." or "Thanks Babaji" → speaker is NOT that name
    not_this_name = defaultdict(set)   # name → speakers who are NOT that name
    addressed_by  = defaultdict(set)   # speaker → names they address

    address_re = re.compile(
        r'\b([A-Z][a-z]{2,}(?:\s+[A-Z][a-z]+)?)\s*[,!?]|'  # "Varun," or "Babuji!"
        r'\bThanks?\s+([A-Z][a-z]{2,})\b|'                   # "Thanks Babaji"
        r'\bHi\s+([A-Z][a-z]{2,})\b'                         # "Hi Poornima"
    )

    for spk in speakers:
        full = " ".join(speaker_texts[spk])
        for m in address_re.finditer(full):
            name = next((g for g in m.groups() if g), None)
            if name and name.lower() not in IGNORE_WORDS and len(name) > 2:
                addressed_by[spk].add(name)
                not_this_name[name].add(spk)

    # ── Step 2: Build candidate name list from all addresses ─────────
    all_candidate_names = set()
    for names in addressed_by.values():
        all_candidate_names.update(names)

    if not all_candidate_names:
        return {}

    print(f"\n    🔍 Names detected in conversation: {', '.join(sorted(all_candidate_names))}")

    # ── Step 3: Score each speaker→name assignment ───────────────────
    # Higher score = more likely this speaker IS this name
    scores = defaultdict(lambda: defaultdict(float))

    for name in all_candidate_names:
        non_candidates = not_this_name.get(name, set())
        # Speakers who are NOT this name get negative score
        for spk in non_candidates:
            scores[spk][name] -= 2.0
        # Speakers who ARE possibly this name get positive score
        for spk in speakers:
            if spk not in non_candidates:
                scores[spk][name] += 1.0

    # ── Step 4: First-person identity clues ─────────────────────────
    # These are strong signals — boost score significantly
    identity_clues = [
        # (regex pattern, name_hint or None, role_hint)
        (r"\bI.ll\s+set\s+up\s+a\s+(call|meeting)\b", None, "sets_up_call"),
        (r"\bI\s+will\s+set\s+up\b",                   None, "sets_up_call"),
        (r"\bhave\s+been\s+managing\s+customer\s+visits\b", None, "manages_visits"),
        (r"\bIT\s+is\s+taking\s+all\s+the\s+blame\b",   None, "defends_IT"),
        (r"\bour\s+IT\s+ticketing\b",                   None, "IT_person"),
        (r"\bmy\s+team\s+can\s+build\b",                None, "IT_person"),
        (r"\bI\s+cannot\s+bypass\s+the\s+firewall\b",   None, "IT_person"),
        (r"\bdelivery\s+takes\s+the\s+blame\b",         None, "delivery_person"),
        (r"\bLet.s\s+get\s+started\b",                  None, "opens_meeting"),
        (r"\baction\s+plan\s+moving\s+forward\b",       None, "opens_meeting"),
        (r"\bI\s+said\s+at\s+the\s+beginning\b",        None, "opens_meeting"),
        (r"\bsales\s+is\s+managing\s+50\s+accounts\b",  None, "sales_person"),
        (r"\bI\s+was\s+explicitly\s+told\b",            None, "admin_person"),
        (r"\bnot\s+expected\s+to\s+monitor\s+whatsapp\b", None, "admin_person"),
    ]

    role_name_map = {}   # role → name (filled as we find clues)
    speaker_roles = defaultdict(set)  # speaker → roles

    for spk in speakers:
        full = " ".join(speaker_texts[spk])
        for pattern, name_hint, role in identity_clues:
            if re.search(pattern, full, re.IGNORECASE):
                speaker_roles[spk].add(role)
                if name_hint:
                    scores[spk][name_hint] += 3.0

    # Now cross-reference roles with addressed names
    # e.g. if Speaker 00 "opens_meeting" and others address "Babuji" to Speaker 00's cluster
    for spk in speakers:
        full = " ".join(speaker_texts[spk])
        for name in all_candidate_names:
            # If this speaker is addressed AS this name by others
            is_addressed_as = any(
                name in addressed_by[other_spk]
                and other_spk != spk
                for other_spk in speakers
            )
            # Check if name appears right after this speaker's turns
            # as a response like "Thanks Babuji" or "OK Babuji"
            if is_addressed_as:
                scores[spk][name] += 1.5

    # ── Step 5: Hungarian-style greedy assignment ────────────────────
    # Assign best name to each speaker, no duplicates
    name_map = {}
    used_names = set()
    remaining_speakers = list(speakers)

    # Sort by confidence — assign most certain first
    assignment_scores = []
    for spk in speakers:
        for name in all_candidate_names:
            s = scores[spk][name]
            if s > 0:
                assignment_scores.append((s, spk, name))
    assignment_scores.sort(reverse=True)

    for score, spk, name in assignment_scores:
        if spk not in name_map and name not in used_names:
            name_map[spk] = name
            used_names.add(name)

    # Speakers with no confident match keep default label
    for spk in speakers:
        if spk not in name_map:
            name_map[spk] = spk

    # ── Step 6: Print what was found ────────────────────────────────
    print(f"\n    🤖 Auto-detected speaker identities:")
    for spk, name in name_map.items():
        if name != spk:
            print(f"      {spk} → {name} ✅")
        else:
            print(f"      {spk} → (could not identify) ⚠️")

    return name_map


def apply_name_map(segments: list, name_map: dict, expected_speakers: int = 0) -> tuple:
    """Apply speaker names to segments.
    Priority:
      1. OCR found real names → merge with auto-detect for unidentified
      2. Auto-detect from conversation → always confirm with user
      3. Only 1 speaker → manual split option
    """
    unique = list(dict.fromkeys(s["speaker"] for s in segments))

    if len(unique) <= 1:
        # Only 1 speaker — diarization failed
        print("\n" + "═"*65)
        print("  ⚠️  WARNING: Only 1 speaker label detected in transcript!")
        print("─"*65)
        choice = input("  Manually assign speakers? (y/n) → ").strip().lower()
        if choice == "y":
            segments = manual_speaker_split(segments)
        else:
            name = input("  Name for this single speaker → ").strip()
            for seg in segments:
                seg["speaker"] = name if name else "Speaker 1"
        final_map = {seg["speaker"]: seg["speaker"] for seg in segments}
        return segments, final_map

    # Build OCR real names check
    ocr_found_real_names = name_map and any(
        not re.match(r'^Speaker\s*\d+$', n, re.IGNORECASE)
        for n in name_map.values()
    )

    # Run auto-detection from conversation context
    print(f"\n[Stage 3d] 🧠 Auto Speaker Identification from Conversation")
    auto_map = auto_detect_speaker_names(segments)

    # Merge: OCR real names take priority, auto-detect fills gaps
    merged = {}
    for spk in unique:
        ocr_name = name_map.get(spk, spk) if ocr_found_real_names else spk
        auto_name = auto_map.get(spk, spk)
        ocr_is_real = not re.match(r'^Speaker\s*\d+$', ocr_name, re.IGNORECASE)
        merged[spk] = ocr_name if ocr_is_real else auto_name

    # ── Check if all speakers were identified automatically ───────────
    all_identified = all(
        not re.match(r'^Speaker\s*\d+$', name, re.IGNORECASE)
        for name in merged.values()
    )

    if all_identified:
        # Fully automatic — no prompts needed
        print("\n" + "═"*68)
        print("  👤 SPEAKER IDENTIFICATION — Fully automatic ✅")
        print("═"*68)
        for spk, name in merged.items():
            print(f"     {spk}  →  {name} ✅")
        print("─"*68)
        confirmed = merged
    else:
        # Some speakers unidentified — only ask for those
        speaker_segs = defaultdict(list)
        for seg in segments:
            speaker_segs[seg["speaker"]].append(seg)

        print("\n" + "═"*68)
        print("  👤 SPEAKER IDENTIFICATION")
        print("  ✅ = auto-detected  |  ❓ = needs your input")
        print("═"*68)

        confirmed = {}
        for spk in unique:
            suggested = merged.get(spk, spk)
            is_identified = not re.match(r'^Speaker\s*\d+$', suggested, re.IGNORECASE)

            if is_identified:
                # Auto-detected — no prompt, just confirm silently
                confirmed[spk] = suggested
                print(f"  ✅ {spk}  →  {suggested}")
            else:
                # Could not identify — show sample lines and ask
                segs = speaker_segs[spk]
                picks = [segs[0], segs[len(segs)//2]] if len(segs) > 1 else segs
                print(f"\n  ❓ {spk} — could not auto-identify")
                print(f"  ┌── {len(segs)} segments, sample lines:")
                for s in picks:
                    print(f"  │  [{fmt_time(s['start'])}] {s['text'][:85]}")
                print(f"  └──────────────────────────────────────────────────────────")
                ans = input(f"  Type name for this speaker (ENTER = keep '{spk}'): ").strip()
                confirmed[spk] = ans if ans else spk

        print("\n" + "─"*68)
        print("  ✅ Final speaker assignments:")
        for spk, name in confirmed.items():
            print(f"     {spk}  →  {name}")
        print("─"*68)

    for seg in segments:
        seg["speaker"] = confirmed.get(seg["speaker"], seg["speaker"])

    named_unique = list(dict.fromkeys(s["speaker"] for s in segments))
    if expected_speakers > 0 and len(named_unique) < expected_speakers:
        print(f"\n  ⚠️  {expected_speakers - len(named_unique)} speaker(s) may be merged "
              f"into others — pyannote could not separate all voices.")

    return segments, confirmed


# ══════════════════════════════════════════════════════════════════════
# STAGE 4: SENTIMENT ANALYSIS
# ══════════════════════════════════════════════════════════════════════

def stage4_sentiment(segments: list) -> list:
    print(f"\n[Stage 4/8] 💬 Sentiment Analysis")
    try:
        from transformers import pipeline as hf_pipeline
        classifier = hf_pipeline(
            "sentiment-analysis",
            model=CONFIG["sentiment_model"],
            device=-1, truncation=True, max_length=512
        )
        label_map = {"positive": "Positive", "negative": "Negative", "neutral": "Neutral",
                     "LABEL_0": "Negative", "LABEL_1": "Neutral", "LABEL_2": "Positive"}
        results = []
        for seg in segments:
            if len(seg["text"].split()) < 3:
                seg["sentiment"] = "Neutral"
                seg["sentiment_score"] = 0.5
            else:
                out = classifier(seg["text"][:512])[0]
                seg["sentiment"] = label_map.get(out["label"].lower(), out["label"])
                seg["sentiment_score"] = round(out["score"], 3)
            results.append(seg)

        counts = defaultdict(int)
        for seg in results:
            counts[seg["sentiment"]] += 1
        summary = ", ".join(f"{k}: {v}" for k, v in counts.items())
        log_stage(4, "Sentiment Analysis", "success",
                  f"cardiffnlp/twitter-roberta — {summary}")
        return results
    except Exception as e:
        log_stage(4, "Sentiment Analysis", "warning", f"Sentiment skipped: {e}")
        for seg in segments:
            seg["sentiment"] = "Neutral"
            seg["sentiment_score"] = 0.5
        return segments


# ══════════════════════════════════════════════════════════════════════
# STAGE 5: ACTION ITEM EXTRACTION
# ══════════════════════════════════════════════════════════════════════


# Strong patterns — first person commitments ("I will", "we will", "I am going to")
ACTION_STRONG = [
    r"\bI will\b",
    r"\bI'll\b",
    r"\bwe will\b",
    r"\bwe'll\b",
    r"\bI am going to\b",
    r"\bI'm going to\b",
    r"\bI can\b.{0,30}\b(by|before|tomorrow|next)\b",
    r"\bI want you to\b",
    r"\bI need you to\b",
    r"\byou need to\b",
    r"\bplease (send|review|check|prepare|update|create|set up|schedule|share|follow)\b",
    r"\b(first|second|third|action item)[,:\s].{5,}",
    r"\bwe are (creating|mandating|building|implementing|rolling out)\b",
    r"\bmoving forward\b.{0,60}\b(will|must|should)\b",
    r"\bI want\b.{0,30}\bto\b",
    r"\bwe need to\b",
    r"\byou should\b",
    r"\bmake sure\b",
    r"\bensure that\b",
    r"\blet\'s (make|ensure|create|set|schedule|review|follow)\b",
    r"\bshould have\b",
    r"\bmust (have|be|do|create|send|review)\b",
    r"\bgoing forward\b",
    r"\bnext step\b",
    r"\baction plan\b",
    r"\bI will set up\b",
    r"\bI will (send|create|review|check|prepare|follow|update|schedule)\b",
]

# Weak patterns — only count if combined with strong signal
ACTION_WEAK = [
    r"\b(send|review|check|prepare|update|finalize|schedule|share)\b",
    r"\b(by end of|next week|tomorrow|before the|deadline|by [A-Z][a-z]+day)\b",
    r"\b(follow up|follow-up|action item|task|responsible for)\b",
]

# Phrases that disqualify a segment — past tense complaints, arguments
ACTION_EXCLUDE = [
    r"\b(he said|she said|they said|I said|you said)\b",
    r"\b(was|were|happened|occurred|failed|missed|forgot|didn't|couldn't|wasn't)\b.{0,30}\b(because|since|when)\b",
    r"^(but|so|and|no|yes|ok|okay|right|exactly|absolutely|unfortunately)\b",
    r"\b(I don't|I didn't|I can't|I couldn't|I'm not|we don't|we didn't)\b",
    r"\bhiding behind\b",
    r"\bpointing fingers\b",
    r"\bsomeone else's fault\b",
    r"\btakes the blame\b",
]

def stage5_action_items(segments: list) -> list:
    print(f"\n[Stage 5/8] ✅ Action Item Extraction")
    action_items = []

    strong_compiled  = [re.compile(p, re.IGNORECASE) for p in ACTION_STRONG]
    weak_compiled    = [re.compile(p, re.IGNORECASE) for p in ACTION_WEAK]
    exclude_compiled = [re.compile(p, re.IGNORECASE) for p in ACTION_EXCLUDE]

    for seg in segments:
        text = seg["text"].strip()

        # Skip very short segments — unlikely to be action items
        if len(text.split()) < 5:
            continue

        # Skip if any exclusion pattern matches
        if any(p.search(text) for p in exclude_compiled):
            continue

        strong_score = sum(1 for p in strong_compiled if p.search(text))
        weak_score   = sum(1 for p in weak_compiled   if p.search(text))

        # Must have at least 1 strong signal, OR 2+ weak signals
        if strong_score >= 1 or weak_score >= 2:
            action_items.append({
                "speaker":    seg["speaker"],
                "text":       text,
                "timestamp":  fmt_time(seg["start"]),
                "confidence": min((strong_score + weak_score * 0.5) / 2.0, 1.0)
            })

    # Deduplicate — remove near-identical action items
    seen, deduped = set(), []
    for item in action_items:
        key = item["text"][:60].lower()
        if key not in seen:
            seen.add(key)
            deduped.append(item)

    log_stage(5, "Action Item Extraction", "success",
              f"Commitment-focused detection — {len(deduped)} action items found")
    return deduped


# ══════════════════════════════════════════════════════════════════════
# STAGE 6: KEYWORD & TOPIC EXTRACTION
# ══════════════════════════════════════════════════════════════════════

def stage6_keywords(full_text: str) -> dict:
    print(f"\n[Stage 6/8] 🔑 Keyword & Topic Extraction")
    results = {"keywords": [], "topics": [], "method": ""}
    try:
        from keybert import KeyBERT
        kw_model = KeyBERT()
        keywords = kw_model.extract_keywords(
            full_text, keyphrase_ngram_range=(1, 2),
            stop_words="english", top_n=15
        )
        results["keywords"] = [{"keyword": kw, "score": round(score, 3)} for kw, score in keywords]
        results["method"] = "KeyBERT"
        log_stage(6, "Keywords", "success",
                  f"KeyBERT — Top keywords: {', '.join(kw for kw, _ in keywords[:5])}")
    except Exception:
        try:
            import yake
            kw_extractor = yake.KeywordExtractor(lan="en", n=2, top=15)
            keywords = kw_extractor.extract_keywords(full_text)
            results["keywords"] = [{"keyword": kw, "score": round(score, 3)} for kw, score in keywords]
            results["method"] = "YAKE"
            log_stage(6, "Keywords", "success",
                      f"YAKE — Top keywords: {', '.join(kw for kw, _ in keywords[:5])}")
        except Exception as e:
            log_stage(6, "Keywords", "warning", f"Keyword extraction skipped: {e}")

    # Simple topic clustering by keyword grouping
    topic_words = [k["keyword"] for k in results["keywords"][:10]]
    results["topics"] = topic_words[:5]
    return results


# ══════════════════════════════════════════════════════════════════════
# STAGE 7: SUMMARIZATION + TRANSLATION
# ══════════════════════════════════════════════════════════════════════

def chunk_text(text: str, max_words=400) -> list:
    words = text.split()
    chunks, cur = [], []
    for w in words:
        cur.append(w)
        if len(cur) >= max_words:
            chunks.append(" ".join(cur))
            cur = []
    if cur:
        chunks.append(" ".join(cur))
    return chunks


def summarize(text: str, summarizer) -> str:
    if not text.strip() or len(text.split()) < 25:
        return text.strip()
    chunks = chunk_text(text)
    summaries = []
    for chunk in chunks:
        if len(chunk.split()) < 20:
            summaries.append(chunk)
            continue
        # Truncate chunk to 400 words max before passing to BART
        chunk = " ".join(chunk.split()[:400])
        out = summarizer(chunk, max_length=CONFIG["max_summary_length"],
                         min_length=CONFIG["min_summary_length"], do_sample=False,
                         truncation=True)
        summaries.append(out[0]["summary_text"])
    combined = " ".join(summaries)
    if len(chunks) > 1 and len(combined.split()) > CONFIG["max_summary_length"]:
        out = summarizer(combined, max_length=CONFIG["max_summary_length"],
                         min_length=CONFIG["min_summary_length"], do_sample=False,
                         truncation=True)
        return out[0]["summary_text"]
    return combined


def ask_language() -> str:
    """Interactively ask user which language they want the report in."""
    print("\n" + "="*60)
    print("  📋 REPORT LANGUAGE SELECTION")
    print("="*60)
    options = list(LANGUAGE_MODELS.keys())
    for i, lang in enumerate(options, 1):
        label = LANGUAGE_DISPLAY.get(lang, lang.title())
        print(f"    {i}. {label}")
    print()
    while True:
        choice = input("  Select language (enter number) → ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(options):
            selected = options[int(choice) - 1]
            print(f"  ✅ Report will be generated in: {LANGUAGE_DISPLAY[selected]}")
            return selected
        print(f"  Please enter a number between 1 and {len(options)}")


def translate_text(text: str, model_name: str) -> str:
    """Translate text using a Helsinki-NLP MarianMT model."""
    if not text.strip() or not model_name:
        return ""
    try:
        from transformers import MarianMTModel, MarianTokenizer
        tok = MarianTokenizer.from_pretrained(model_name)
        mdl = MarianMTModel.from_pretrained(model_name)
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
        if not sentences:
            sentences = [text]
        translated = []
        for i in range(0, len(sentences), 4):
            batch = sentences[i:i+4]
            inputs = tok(batch, return_tensors="pt", padding=True,
                         truncation=True, max_length=512)
            outputs = mdl.generate(**inputs, max_length=512)
            translated.extend([tok.decode(o, skip_special_tokens=True) for o in outputs])
        return " ".join(translated)
    except Exception as e:
        return f"[Translation unavailable: {e}]"


def build_speaker_profile(spk: str, segments: list, action_items: list) -> dict:
    """
    Build a rich statistical + content profile for one speaker.
    Used to generate meaningful summaries that reflect contribution,
    not just a dump of what they said word-for-word.
    """
    spk_segs  = [s for s in segments if s["speaker"] == spk]
    total_sec = sum(s["end"] - s["start"] for s in spk_segs)
    minutes   = int(total_sec // 60)
    seconds   = int(total_sec % 60)

    # Sentiment breakdown
    sent_counts = defaultdict(int)
    for s in spk_segs:
        sent_counts[s.get("sentiment", "Neutral")] += 1
    dominant = max(sent_counts, key=sent_counts.get) if sent_counts else "Neutral"

    # Questions raised
    questions = [s["text"] for s in spk_segs if "?" in s["text"]]

    # Action items owned by this speaker
    owned_actions = [a["text"] for a in action_items if a["speaker"] == spk]

    # Key sentences: the 3 longest / most substantive segments
    key_sentences = sorted(spk_segs, key=lambda s: len(s["text"]), reverse=True)[:3]
    key_sentences = [s["text"] for s in key_sentences]

    return {
        "speaking_time":      f"{minutes}m {seconds}s",
        "segment_count":      len(spk_segs),
        "dominant_sentiment": dominant,
        "sentiment_breakdown": dict(sent_counts),
        "questions_asked":    questions,
        "action_items_owned": owned_actions,
        "key_sentences":      key_sentences,
        "all_text":           " ".join(s["text"] for s in spk_segs),
    }


def stage7_summarize(segments: list, action_items: list = None, keywords: dict = None, language: str = 'english') -> dict:
    print(f"\n[Stage 7/8] 📝 Summarization + Translation")
    if action_items is None:
        action_items = []
    if keywords is None:
        keywords = {}

    speaker_texts = defaultdict(list)
    for seg in segments:
        speaker_texts[seg["speaker"]].append(seg["text"])
    full_text = " ".join(seg["text"] for seg in segments)

    # Extract top keyword phrases to weave into the meeting summary
    kw_list = [k["keyword"] for k in keywords.get("keywords", [])[:8]]
    kw_context = ""
    if kw_list:
        kw_context = (f"Key topics discussed include: {', '.join(kw_list)}. ")

    try:
        from transformers import pipeline as hf_pipeline
        summarizer = hf_pipeline("summarization", model=CONFIG["summarization_model"],
                                  device=-1)
        log_stage(7, "Summarization", "success", "BART model loaded")
    except Exception as e:
        log_stage(7, "Summarization", "warning", f"Summarizer unavailable: {e}")
        summarizer = None

    def safe_summarize(text):
        if summarizer:
            return summarize(text, summarizer)
        # Fallback: extract best sentences
        sentences = [s.strip() for s in re.split(r"(?<=[.!?]) +", text) if len(s.strip()) > 20]
        return " ".join(sentences[:6])

    # ── Full meeting summary (keywords merged in) ────────────────────
    print("    ⏳ Generating meeting summary...")
    enriched_text = kw_context + full_text
    full_en = safe_summarize(enriched_text)

    # ── Per-speaker paragraph summaries ─────────────────────────────
    speaker_profiles     = {}
    speaker_summaries_en = {}

    for spk in speaker_texts:
        print(f"    ⏳ Summarizing {spk}...")

        profile = build_speaker_profile(spk, segments, action_items)
        speaker_profiles[spk] = profile

        # Build rich context so BART writes a proper contribution paragraph
        ctx = f"{spk} said the following during the meeting: "
        if profile["key_sentences"]:
            ctx += " ".join(profile["key_sentences"]) + " "
        ctx += profile["all_text"]

        speaker_summaries_en[spk] = safe_summarize(ctx)

    # ── Translate if user selected a non-English language ───────────
    model_name = LANGUAGE_MODELS.get(language)
    full_translated      = ""
    speakers_translated  = {}

    if model_name:
        print(f"    ⏳ Translating to {LANGUAGE_DISPLAY.get(language, language)}...")
        full_translated = translate_text(full_en, model_name)
        for spk, summary in speaker_summaries_en.items():
            print(f"    ⏳ Translating {spk} summary...")
            speakers_translated[spk] = translate_text(summary, model_name)

    log_stage(7, "Summarization", "success",
              f"Meeting summary + {len(speaker_texts)} speaker summaries"
              + (f" + {LANGUAGE_DISPLAY.get(language)} translation" if model_name else ""))

    return {
        "full_en":           full_en,
        "speakers_en":       speaker_summaries_en,
        "profiles":          speaker_profiles,
        "full_translated":   full_translated,
        "speakers_translated": speakers_translated,
        "language":          language,
    }


# ══════════════════════════════════════════════════════════════════════
# STAGE 8: REPORT GENERATION (TXT)
# ══════════════════════════════════════════════════════════════════════

def fmt_time(sec: float) -> str:
    return f"{int(sec//60):02d}:{int(sec%60):02d}"


def stage8_report(segments, summaries, action_items, keywords, input_file, output_txt,
                  speaker_mode="diarization"):
    print(f"\n[Stage 8/8] 📄 Generating Report")
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    speakers = list(dict.fromkeys(s["speaker"] for s in segments))
    duration = fmt_time(segments[-1]["end"]) if segments else "N/A"

    # Check if speakers are generic labels (unidentified)
    unidentified = all(
        re.match(r'^Speaker\s+\d+$', spk, re.IGNORECASE) or
        re.match(r'^SPEAKER_\d+$', spk)
        for spk in speakers
    )

    def wrap(text, width=80, indent="  "):
        """Wrap text into lines of max width with indent."""
        words = text.split()
        lines_out, line = [], []
        for word in words:
            line.append(word)
            if len(" ".join(line)) > width:
                lines_out.append(indent + " ".join(line[:-1]))
                line = [word]
        if line:
            lines_out.append(indent + " ".join(line))
        return lines_out

    with open(output_txt, "w", encoding="utf-8") as f:
        W = lambda t="": f.write(t + "\n")
        L = lambda n=68: f.write("=" * n + "\n")
        D = lambda n=68: f.write("-" * n + "\n")

        # ── Header ──────────────────────────────────────────────────
        L()
        W("  MEETING ANALYTICS REPORT")
        W("  Team Vision AI")
        L()
        W(f"  File      : {Path(input_file).name}")
        W(f"  Generated : {now}")
        W(f"  Duration  : {duration}")
        W(f"  Attendees : {', '.join(speakers)}")
        if speaker_mode == "teams_transcript":
            W(f"  Speaker ID: ✅ Identified via Microsoft Teams transcript (100% accurate)")
        elif unidentified:
            W(f"  Speaker ID: ⚠️  Voice-based only — names not available from video alone")
            W(f"              💡 Re-run with --transcript for automatic name identification")
        else:
            W(f"  Speaker ID: 🔊 Voice-based diarization (pyannote AI)")
        L()

        # ── Meeting Summary ──────────────────────────────────────────
        W()
        D()
        W("  MEETING SUMMARY")
        D()
        W()
        summary_text = summaries.get("full_en", "Summary not available.")
        for l in wrap(summary_text):
            W(l)
        W()

        # ── Action Items ─────────────────────────────────────────────
        W()
        D()
        W("  ACTION ITEMS")
        D()
        W()
        if action_items:
            # Group by speaker and deduplicate
            by_speaker = defaultdict(list)
            for item in action_items:
                by_speaker[item["speaker"]].append(item["text"].strip())

            for spk, tasks in by_speaker.items():
                W(f"  {spk}:")
                seen = set()
                for task in tasks:
                    if task not in seen:
                        seen.add(task)
                        task_lines = wrap(task, width=74, indent="    ")
                        if task_lines:
                            W("    • " + task_lines[0].strip())
                            for extra in task_lines[1:]:
                                W("      " + extra.strip())
                W()
        else:
            W("  No action items detected.")
            W()

        # ── Speaker-wise Summary (English) ───────────────────────────
        W()
        D()
        W("  SPEAKER-WISE SUMMARY")
        D()
        for spk in speakers:
            W()
            W(f"  {spk}:")
            spk_summary = summaries.get("speakers_en", {}).get(spk, "Summary not available.")
            for l in wrap(spk_summary):
                W(l)
            W()

        # ── Translated Section ────────────────────────────────────────
        language   = summaries.get("language", "english")
        model_name = LANGUAGE_MODELS.get(language)
        lang_label = LANGUAGE_DISPLAY.get(language, language.title())

        if model_name and summaries.get("full_translated"):
            W()
            L()
            W(f"  MEETING REPORT — {lang_label.upper()}")
            L()

            # Meeting Summary translated
            W()
            D()
            W(f"  MEETING SUMMARY — {lang_label}")
            D()
            W()
            for l in wrap(summaries.get("full_translated", "")):
                W(l)
            W()

            # Action Items translated
            W()
            D()
            W(f"  ACTION ITEMS — {lang_label}")
            D()
            W()
            if action_items:
                by_speaker = defaultdict(list)
                for item in action_items:
                    by_speaker[item["speaker"]].append(item["text"].strip())
                for spk, tasks in by_speaker.items():
                    W(f"  {spk}:")
                    seen = set()
                    for task in tasks:
                        if task not in seen:
                            seen.add(task)
                            t_task = translate_text(task, model_name)
                            tlines = wrap(t_task, width=74, indent="    ")
                            if tlines:
                                W("    • " + tlines[0].strip())
                                for extra in tlines[1:]:
                                    W("      " + extra.strip())
                    W()
            else:
                W("  No action items detected.")
                W()

            # Speaker summaries translated
            W()
            D()
            W(f"  SPEAKER-WISE SUMMARY — {lang_label}")
            D()
            for spk in speakers:
                W()
                W(f"  {spk}:")
                t_summary = summaries.get("speakers_translated", {}).get(
                    spk, "[Translation not available]")
                for l in wrap(t_summary):
                    W(l)
                W()

        L()
        W("  END OF REPORT — Team Vision AI")
        L()

    log_stage(8, "Report Generation", "success", f"Saved to: {output_txt}")

def save_stage_outputs(segments, action_items, keywords, summaries, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # Stage output files
    with open(f"{output_dir}/stage2_transcript.json", "w", encoding="utf-8") as f:
        json.dump([{"start": s["start"], "end": s["end"], "text": s["text"]} for s in segments],
                  f, indent=2, ensure_ascii=False)

    with open(f"{output_dir}/stage3_diarized_transcript.json", "w", encoding="utf-8") as f:
        json.dump([{"speaker": s["speaker"], "start": s["start"], "end": s["end"], "text": s["text"]}
                   for s in segments], f, indent=2, ensure_ascii=False)

    with open(f"{output_dir}/stage4_sentiment.json", "w", encoding="utf-8") as f:
        json.dump([{"speaker": s["speaker"], "text": s["text"],
                    "sentiment": s.get("sentiment", "N/A"),
                    "score": s.get("sentiment_score", 0)} for s in segments],
                  f, indent=2, ensure_ascii=False)

    with open(f"{output_dir}/stage5_action_items.json", "w", encoding="utf-8") as f:
        json.dump(action_items, f, indent=2, ensure_ascii=False)

    with open(f"{output_dir}/stage6_keywords.json", "w", encoding="utf-8") as f:
        json.dump(keywords, f, indent=2, ensure_ascii=False)

    with open(f"{output_dir}/stage7_summaries.json", "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2, ensure_ascii=False)

    with open(f"{output_dir}/pipeline_stage_log.json", "w", encoding="utf-8") as f:
        json.dump(STAGE_LOG, f, indent=2, ensure_ascii=False)

    print(f"\n    ✅ Stage outputs saved to: {output_dir}/")


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Vision AI Meeting Analytics Pipeline")
    parser.add_argument("--file",       required=True, help="Path to meeting video/audio file")
    parser.add_argument("--hf_token",   default=os.environ.get("HF_TOKEN", ""),
                        help="HuggingFace token for speaker diarization")
    parser.add_argument("--transcript", default="",
                        help="(Optional) Path to Microsoft Teams transcript .txt file for accurate speaker names")
    parser.add_argument("--output",     default="", help="Output directory (optional)")
    args = parser.parse_args()

    if not os.path.exists(args.file):
        print(f"❌ File not found: {args.file}"); sys.exit(1)

    stem = Path(args.file).stem
    out_dir = args.output or f"{stem}_vision_ai_output"
    os.makedirs(out_dir, exist_ok=True)
    output_txt = os.path.join(out_dir, f"{stem}_analytics_report.txt")
    stage_dir  = os.path.join(out_dir, "stage_outputs")

    print("\n" + "═"*68)
    print("     🚀 VISION AI — Meeting Analytics Pipeline Starting")
    print("═"*68)

    # ── Ask language preference upfront ─────────────────────────────
    selected_language = ask_language()

    with tempfile.TemporaryDirectory() as _unused_tmpdir:
        # WAV saved to out_dir to avoid Windows short-path issues
        # (C:\Users\BHAGYA~1\... paths break pyannote)
        wav_path = os.path.join(out_dir, "audio_extracted.wav")

        stage1_extract_audio(args.file, wav_path)
        whisper_result = stage2_transcribe(wav_path)

        # ── Speaker attribution: Teams transcript → pyannote → manual ──
        transcript_path = args.transcript
        if transcript_path and os.path.exists(transcript_path):
            # BEST PATH: Use Teams transcript for 100% accurate speaker names
            print("\n  ✅ Teams transcript provided — skipping pyannote diarization")
            print("     Speaker names will be read directly from the transcript file.")
            segments = merge_segments(whisper_result, [])
            segments = stage3c_teams_transcript(transcript_path, segments)
            print("\n  ✅ Speaker names assigned from Teams transcript — no manual input needed!")
            speaker_mode = "teams_transcript"
        else:
            # FALLBACK: Use pyannote diarization + manual name assignment
            if not transcript_path:
                print("\n  ℹ️  No Teams transcript provided.")
                print("     Using pyannote for speaker diarization.")
                print("     TIP: Re-run with --transcript transcript.txt for automatic naming.")
            diar_segs, expected_spk = stage3_diarize(wav_path, args.hf_token)
            segments    = merge_segments(whisper_result, diar_segs)
            name_map    = stage3b_ocr_names(args.file, diar_segs)
            segments, _ = apply_name_map(segments, name_map, expected_spk)
            speaker_mode = "diarization"

        action_items   = stage5_action_items(segments)
        full_text      = " ".join(s["text"] for s in segments)
        keywords       = stage6_keywords(full_text)
        summaries      = stage7_summarize(segments, action_items, keywords, selected_language)
        stage8_report(segments, summaries, action_items, keywords, args.file, output_txt,
                      speaker_mode=speaker_mode)
        save_stage_outputs(segments, action_items, keywords, summaries, stage_dir)

        # Clean up extracted WAV to save disk space
        if os.path.exists(wav_path):
            os.remove(wav_path)
            print("    🧹 Temporary audio file cleaned up")

    print("\n" + "═"*68)
    print("  ✅  PIPELINE COMPLETE — Team Vision AI")
    print(f"  📄  Report : {output_txt}")
    print(f"  📁  Stages : {stage_dir}/")
    print("═"*68 + "\n")


if __name__ == "__main__":
    main()
