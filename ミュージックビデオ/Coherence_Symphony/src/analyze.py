import numpy as np, librosa, sys, json
y, sr = librosa.load(sys.argv[1], sr=22050)
dur = len(y)/sr
FPS = 30
hop = int(sr/FPS)  # 735
tempo, beats = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop)
bt = librosa.frames_to_time(beats, sr=sr, hop_length=hop)
print("dur", dur, "tempo", tempo, "nbeats", len(bt))
print("first beats", np.round(bt[:12],2))
S = np.abs(librosa.stft(y, n_fft=2048, hop_length=hop))
freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
def band(lo,hi):
    m=(freqs>=lo)&(freqs<hi); return S[m].mean(0)
bass=band(20,150); mid=band(150,2000); high=band(2000,11000)
rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=hop)[0]
onset = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
# 64-band log spectrum for circular visualizer
mel = librosa.feature.melspectrogram(S=S**2, sr=sr, n_mels=64, fmax=8000)
meldb = librosa.power_to_db(mel, ref=np.max)
n = int(np.ceil(dur*FPS))
def fit(a):
    a=np.asarray(a,dtype=np.float32)
    if a.shape[-1]<n: a=np.pad(a,[(0,0)]*(a.ndim-1)+[(0,n-a.shape[-1])],mode='edge')
    return a[...,:n]
np.savez(sys.argv[2], bass=fit(bass), mid=fit(mid), high=fit(high), rms=fit(rms), onset=fit(onset), mel=fit(meldb), beats=bt, tempo=np.array(tempo), dur=np.array(dur))
# print a coarse energy profile per 2s
r = fit(rms); 
for i in range(0, n, 60):
    seg=r[i:i+60]; b=fit(bass)[i:i+60]
    print(f"{i/FPS:6.1f}s rms {seg.mean():.3f} bass {b.mean():.2f} " + '#'*int(seg.mean()*150))
