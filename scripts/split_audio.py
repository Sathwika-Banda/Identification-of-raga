import os
from pydub import AudioSegment

INPUT_FOLDER = "data/raw"
OUTPUT_FOLDER = "data/raw"
CLIP_DURATION_MS = 5000  # 5 seconds

# Process all files matching *_full*.wav
for filename in os.listdir(INPUT_FOLDER):
    if "_full" in filename and filename.endswith(".wav"):
        input_path = os.path.join(INPUT_FOLDER, filename)
        raga_name = filename.replace(".wav", "").split("_full")[0]

        print(f"\n🎧 Processing {filename}...")
        audio = AudioSegment.from_wav(input_path)
        duration = len(audio)
        clip_count = duration // CLIP_DURATION_MS

        for i in range(clip_count):
            start = i * CLIP_DURATION_MS
            end = start + CLIP_DURATION_MS
            clip = audio[start:end]
            out_name = f"{raga_name}_{i+1}.wav"
            out_path = os.path.join(OUTPUT_FOLDER, out_name)
            clip.export(out_path, format="wav")
            print(f"✅ Saved: {out_name}")

        print(f"✅ Done! {clip_count} clips created for {raga_name}")
