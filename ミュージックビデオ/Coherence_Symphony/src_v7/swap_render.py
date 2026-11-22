import os, sys, subprocess, numpy as np
sys.argv = ['x']; sys.path.insert(0, '.')
import mv7
from multiprocessing import Pool
b = np.linspace(0, 8027, 17).astype(int)
segdir = os.path.join(mv7.HERE, "segs7")
redo = [5, 13]
jobs = []
for k in redo:
    m = (b[k] + b[k + 1]) // 2
    jobs.append((int(b[k]), int(m), os.path.join(segdir, f"v9_seg{k:03d}a.mp4")))
    jobs.append((int(m), int(b[k + 1]), os.path.join(segdir, f"v9_seg{k:03d}b.mp4")))
with Pool(4) as p:
    p.map(mv7.render_chunk, jobs, chunksize=1)
lst = os.path.join(segdir, "list_v9.txt")
with open(lst, "w") as fh:
    for k in range(15):
        if k in redo:
            fh.write(f"file '{os.path.join(segdir, f'v9_seg{k:03d}a.mp4')}'\n")
            fh.write(f"file '{os.path.join(segdir, f'v9_seg{k:03d}b.mp4')}'\n")
        else:
            fh.write(f"file '{os.path.join(segdir, f'seg{k:03d}.mp4')}'\n")
    for t in range(4):
        fh.write(f"file '{os.path.join(segdir, f'tail{t}.mp4')}'\n")
out = os.path.join(mv7.HERE, "Coherence_Symphony_MV_v9.mp4")
subprocess.check_call(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "concat", "-safe", "0", "-i", lst,
                       "-i", os.environ["MV_AUDIO"], "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "copy",
                       "-movflags", "+faststart", out])
print("done", out)
