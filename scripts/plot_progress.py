import sys, glob, os, csv
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
run = sys.argv[1] if len(sys.argv) > 1 else max(glob.glob("runs/microduck_*"), key=os.path.getmtime)
path = os.path.join(run, "progress.csv")
keys = [("mean_reward","Mean reward"),("mean_episode_length","Episode length"),("value_loss","Value loss"),("entropy","Entropy")]
fig, axes = plt.subplots(2, 2, figsize=(11, 7)); axes = axes.ravel()
fig.canvas.manager.set_window_title("Microduck training: " + os.path.basename(run))
def update(_):
    try:
        rows = list(csv.DictReader(open(path)))
    except Exception:
        return
    it = [int(r["iteration"]) for r in rows]
    for ax, (k, t) in zip(axes, keys):
        ax.clear()
        pts = [(i, float(r[k])) for i, r in zip(it, rows) if not (k in ("mean_reward","mean_episode_length") and float(r["mean_episode_length"]) == 0)]
        if pts:
            ax.plot(*zip(*pts), ".-", ms=3)
        ax.set_title(t); ax.set_xlabel("iteration"); ax.grid(alpha=.3)
    fig.tight_layout()
ani = FuncAnimation(fig, update, interval=3000, cache_frame_data=False)
update(0); plt.show()
