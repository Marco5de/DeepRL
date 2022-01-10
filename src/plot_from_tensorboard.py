import matplotlib.pyplot as plt
import json
import seaborn as sns
import numpy as np
import pandas as pd

JSON_PATH = "res/tb_data/tb_data.json"
SMOOTH_WINDOW = 5


def smooth(y, box_pts):
    box = np.ones(box_pts) / box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth


tb_data = []

with open(JSON_PATH) as f:
    tb_data = json.load(f)

df = pd.DataFrame(tb_data, columns=["id", "time_step", "reward"])
y = df["reward"].tolist()
df["reward"] = smooth(y, SMOOTH_WINDOW)

# todo - need multiple values per step to display ci
sns.set()
sns.lineplot(data=df, x="time_step", y="reward", ci="sd", estimator=np.nanmean, err_style="band")

plt.show()
