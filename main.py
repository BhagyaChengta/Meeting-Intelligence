from services.speech_to_text import transcribe_audio

# audio_file = "audio/sample.wav"
audio_file = "audio/committee.mp3"

result = transcribe_audio(audio_file)

print("\nTranscription:\n")
print(result["text"])

with open("output/transcript.txt", "w", encoding="utf-8") as f:
    f.write(result["text"])
