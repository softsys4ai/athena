# Calculate and plot the dissimilarity between BS and each type of AEs

import sys
import os
import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize


def adjust_fig_aspect(fig,aspect=1):
    '''
    Adjust the subplot parameters so that the figure has the correct
    aspect ratio.
    '''
    xsize,ysize = fig.get_size_inches()
    minsize = min(xsize,ysize)
    xlim = .4*minsize/xsize
    ylim = .4*minsize/ysize
    if aspect < 1:
        xlim *= aspect
    else:
        ylim /= aspect
    fig.subplots_adjust(left=.5-xlim,
                        right=.5+xlim,
                        bottom=.5-ylim,
                        top=.5+ylim)

# directory that contains Bening samples and all types of AEs
dissimilarity_fp = sys.argv[1]
result_dir=sys.argv[2]

df=pd.read_csv(dissimilarity_fp, sep='\t')
fig, ax = plt.subplots(1, 1)

# Get a color map
my_cmap = cm.get_cmap('tab10')
# For color map: get normalize function (takes data in range [vmin, vmax] -> [0, 1])
my_norm = Normalize(vmin=0, vmax=8)

num_AE_Types = len(df['Dissimilarity'])
X = np.arange(num_AE_Types) # use df["AEType"] if it needs to plot AE type in the bar chart directly
barplot = ax.bar(X, df['Dissimilarity'],
       color=my_cmap(my_norm(X // 3))) # three variants of AE types per attack

adjust_fig_aspect(fig, aspect=3)

plt.xticks([])
plt.ylabel("Normalized\n$l_2$ dissimilarity", fontsize=6)
ax.tick_params(labelsize=6)
fig.savefig(os.path.join(result_dir, "normalized_dissimilarity.pdf"), dpi=1200)
