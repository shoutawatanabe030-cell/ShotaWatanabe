import os, sys, subprocess, numpy as np
sys.argv = ['x']; sys.path.insert(0, '.')
import mv7
from multiprocessing import Pool
n1 = 8027
b = np.linspace(0, n1, 17).astype(int)
a0 = int(b[15])
sub = np.linspace(a0, n1, 5).astype(int)
segdir = os.path.join(mv7.HERE, "segs7")
jobs = [(int(sub[k]), int(sub[k + 1]), os.path.join(segdir, f"tail{k}.mp4")) for k in range(4)]
print("tail frames", a0, n1, jobs)
with Pool(4) as p:
    outs = p.map(mv7.render_chunk, jobs, chunksize=1)
lst = os.path.join(segdir, "list_v7b.txt")
with open(lst, "w") as fh:
    for k in range(15):
        fh.write(f"file '{os.path.join(segdir, f'seg{k:03d}.mp4')}'\n")
    for o in outs:
        fh.write(f"file '{o}'\n")
out = os.path.join(mv7.HERE, "Coherence_Symphony_MV_v7b.mp4")
subprocess.check_call(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "concat", "-safe", "0", "-i", lst,
                       "-i", os.environ["MV_AUDIO"], "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "copy",
                       "-movflags", "+faststart", out])
print("done", out)
