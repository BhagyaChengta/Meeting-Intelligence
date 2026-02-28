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
    "whisper_model":       "medium",   # medium = best accuracy for competition
    "device":              "cpu",
    "summarization_model": "facebook/bart-large-cnn",
    "sentiment_model":     "cardiffnlp/twitter-roberta-base-sentiment-latest",
    "translation_model":   "Helsinki-NLP/opus-mt-en-hi",
    "max_summary_length":  250,
    "min_summary_length":  60,
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
        return []

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
        return segments

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
        return []


# ══════════════════════════════════════════════════════════════════════
# MERGE TRANSCRIPT + SPEAKERS
# ══════════════════════════════════════════════════════════════════════

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
        name_map = {}
        for speaker, votes in speaker_name_votes.items():
            if votes:
                best_name = max(votes, key=votes.get)
                name_map[speaker] = best_name
                print(f"    🏷️  {speaker} → '{best_name}' "
                      f"(detected in {votes[best_name]} frames)")

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


def apply_name_map(segments: list, name_map: dict) -> tuple:
    """Apply speaker name map to all segments.
    Handles 3 cases:
      1. OCR gave names → confirm/correct them
      2. Diarization worked (multiple speakers) → just rename labels
      3. Only 1 speaker detected → warn + offer manual split
    """
    unique = list(dict.fromkeys(s["speaker"] for s in segments))

    # Case 1: OCR detected names — confirm them
    if name_map:
        print("\n" + "═"*60)
        print("  👤 CONFIRMING OCR-DETECTED NAMES")
        print("     Press ENTER to confirm, or type correction")
        print("═"*60)
        confirmed = {}
        for spk, detected in name_map.items():
            fix = input(f"  {spk} → '{detected}'  (ENTER to confirm) → ").strip()
            confirmed[spk] = fix if fix else detected
        for seg in segments:
            seg["speaker"] = confirmed.get(seg["speaker"], seg["speaker"])
        return segments, confirmed

    # Case 2: Diarization detected multiple speakers — rename labels
    if len(unique) > 1:
        print("\n" + "═"*65)
        print(f"  👤 {len(unique)} SPEAKERS DETECTED BY DIARIZATION")
        print("  ─"*32)
        print("  NOTE: Speaker numbers are based on VOICE PATTERNS,")
        print("  not speaking order. Read the sample lines below")
        print("  to identify who each speaker is, then type their name.")
        print("═"*65)

        # Show 2-3 sample lines per speaker so user can identify them
        speaker_samples = {}
        for seg in segments:
            spk = seg["speaker"]
            if spk not in speaker_samples:
                speaker_samples[spk] = []
            if len(speaker_samples[spk]) < 3:
                speaker_samples[spk].append(
                    f"  [{fmt_time(seg['start'])}] {seg['text'][:80]}"
                )

        confirmed = {}
        while True:
            # Show all speakers with samples
            print()
            for i, spk in enumerate(unique):
                print(f"  ┌── {spk} ─────────────────────────────────────")
                for sample in speaker_samples.get(spk, []):
                    print(f"  │  {sample}")
                print(f"  └─────────────────────────────────────────────")
                print()

            # Ask for names
            print("  Now enter the real name for each speaker:")
            print("  (Press ENTER to keep the auto label)")
            print("─"*65)
            temp_map = {}
            for spk in unique:
                name = input(f"  {spk} → ").strip()
                temp_map[spk] = name if name else spk

            # Show confirmation summary
            print("\n" + "─"*65)
            print("  ✅ CONFIRM SPEAKER NAMES:")
            for spk, name in temp_map.items():
                print(f"     {spk}  →  {name}")
            print("─"*65)
            ok = input("  Is this correct? (y = confirm / n = re-enter) → ").strip().lower()
            if ok == "y" or ok == "":
                confirmed = temp_map
                break
            else:
                print("\n  Let\'s try again...\n")

        for seg in segments:
            seg["speaker"] = confirmed.get(seg["speaker"], seg["speaker"])
        return segments, confirmed

    # Case 3: Only 1 speaker detected — diarization failed
    print("\n" + "═"*65)
    print("  ⚠️  WARNING: Only 1 speaker label detected in transcript!")
    print("  ⚠️  Diarization likely failed or was skipped.")
    print("  ⚠️  Without a valid HuggingFace token, all speakers")
    print("  ⚠️  are merged into one. Re-run with --hf_token for")
    print("  ⚠️  automatic multi-speaker detection.")
    print("─"*65)
    print("  You can manually assign speakers to transcript blocks now.")
    choice = input("  Manually assign speakers? (y/n) → ").strip().lower()
    if choice == "y":
        segments = manual_speaker_split(segments)
    else:
        name = input("  Name for this single speaker → ").strip()
        for seg in segments:
            seg["speaker"] = name if name else "Speaker 1"

    final_map = {}
    for seg in segments:
        final_map[seg["speaker"]] = seg["speaker"]
    return segments, final_map


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

ACTION_PATTERNS = [
    r"\b(will|shall|need to|should|must|going to|plan to|have to|assigned to|action[:\s])\b",
    r"\b(follow up|follow-up|send|review|check|schedule|prepare|update|complete|finalize|discuss)\b",
    r"\b(by [A-Z][a-z]+day|by end of|next week|tomorrow|before the|deadline)\b",
    r"\b(todo|to-do|action item|task|responsible)\b",
]

def stage5_action_items(segments: list) -> list:
    print(f"\n[Stage 5/8] ✅ Action Item Extraction")
    action_items = []
    compiled = [re.compile(p, re.IGNORECASE) for p in ACTION_PATTERNS]
    for seg in segments:
        text = seg["text"]
        score = sum(1 for p in compiled if p.search(text))
        if score >= 1:
            action_items.append({
                "speaker": seg["speaker"],
                "text": text.strip(),
                "timestamp": fmt_time(seg["start"]),
                "confidence": min(score / 2.0, 1.0)
            })
    log_stage(5, "Action Item Extraction", "success",
              f"Pattern-matching + heuristics — {len(action_items)} action items found")
    return action_items


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


def translate_to_hindi(text: str) -> str:
    if not text.strip():
        return ""
    try:
        from transformers import MarianMTModel, MarianTokenizer
        tok = MarianTokenizer.from_pretrained(CONFIG["translation_model"])
        mdl = MarianMTModel.from_pretrained(CONFIG["translation_model"])
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
        translated = []
        for i in range(0, len(sentences), 4):
            batch = sentences[i:i+4]
            inputs = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=512)
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


def stage7_summarize(segments: list, action_items: list = None, keywords: dict = None) -> dict:
    print(f"\n[Stage 7/8] 📝 Summarization")
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

    log_stage(7, "Summarization", "success",
              f"Meeting summary + {len(speaker_texts)} speaker summaries generated")

    return {
        "full_en":     full_en,
        "speakers_en": speaker_summaries_en,
        "profiles":    speaker_profiles,
    }


# ══════════════════════════════════════════════════════════════════════
# STAGE 8: REPORT GENERATION (TXT)
# ══════════════════════════════════════════════════════════════════════

def fmt_time(sec: float) -> str:
    return f"{int(sec//60):02d}:{int(sec%60):02d}"


def stage8_report(segments, summaries, action_items, keywords, input_file, output_txt):
    print(f"\n[Stage 8/8] 📄 Generating Report")
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    speakers = list(dict.fromkeys(s["speaker"] for s in segments))
    duration = fmt_time(segments[-1]["end"]) if segments else "N/A"

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

        # ── Speaker-wise Summary ─────────────────────────────────────
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
    parser.add_argument("--file",      required=True, help="Path to meeting video/audio file")
    parser.add_argument("--hf_token",  default=os.environ.get("HF_TOKEN", ""),
                        help="HuggingFace token for speaker diarization")
    parser.add_argument("--output",    default="", help="Output directory (optional)")
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

    with tempfile.TemporaryDirectory() as tmpdir:
        wav_path = os.path.join(tmpdir, "audio.wav")

        stage1_extract_audio(args.file, wav_path)
        whisper_result = stage2_transcribe(wav_path)
        diar_segs      = stage3_diarize(wav_path, args.hf_token)
        segments       = merge_segments(whisper_result, diar_segs)
        name_map       = stage3b_ocr_names(args.file, diar_segs)   # ← NEW: auto-read names from video
        segments, _    = apply_name_map(segments, name_map)         # ← replaces manual rename
        action_items   = stage5_action_items(segments)
        full_text      = " ".join(s["text"] for s in segments)
        keywords       = stage6_keywords(full_text)
        summaries      = stage7_summarize(segments, action_items, keywords)
        stage8_report(segments, summaries, action_items, keywords, args.file, output_txt)
        save_stage_outputs(segments, action_items, keywords, summaries, stage_dir)

    print("\n" + "═"*68)
    print("  ✅  PIPELINE COMPLETE — Team Vision AI")
    print(f"  📄  Report : {output_txt}")
    print(f"  📁  Stages : {stage_dir}/")
    print("═"*68 + "\n")


if __name__ == "__main__":
    main()
