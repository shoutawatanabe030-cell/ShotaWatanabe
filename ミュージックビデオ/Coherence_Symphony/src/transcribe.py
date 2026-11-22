import sys, json
from faster_whisper import WhisperModel
import soundfile as sf, numpy as np
model_dir, wav, out = sys.argv[1], sys.argv[2], sys.argv[3]
m = WhisperModel(model_dir, device="cpu", compute_type="int8", cpu_threads=4)
audio, sr = sf.read(wav, dtype="float32"); assert sr == 16000
segs, info = m.transcribe(audio, language="ja", word_timestamps=True, beam_size=5,
    condition_on_previous_text=False, vad_filter=False,
    initial_prompt="Coherence Symphony 手を取り合っていこう We are one ひとつになっていこう 同じ波動で We are love もう ひとりじゃない")
res = []
for s in segs:
    d = {"start": s.start, "end": s.end, "text": s.text, "words": [(w.start, w.end, w.word, w.probability) for w in (s.words or [])]}
    res.append(d)
    print(f"[{s.start:7.2f} - {s.end:7.2f}] {s.text}", flush=True)
json.dump(res, open(out, "w"), ensure_ascii=False, indent=1)
