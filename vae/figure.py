import sys
import numpy as np
import pandas as pd

import matplotlib
import matplotlib.colors as colors
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
#plt.switch_backend('PDF')
#plt.switch_backend('Agg')
#from matplotlib import rc
from matplotlib import rcParams
#Use tex
rcParams['text.usetex'] = True
rcParams['text.latex.preamble'] = [r'\usepackage{times}']
#serifs
rcParams['font.family']='serifs'


output_filename = sys.argv[1]
input_filenames = sys.argv[2:]

dfs = [pd.read_csv(f) for f in input_filenames]
dfs = [
    list(zip(dfs[:3], dfs[3:6])), 
    list(zip(dfs[6:9], dfs[9:12])),
    list(zip(dfs[12:15], dfs[15:18])),
]

iwa_tmc = ["IWAE", "TMC"]
std_stl_drg = ["Standard", "STL", "DReGs"]


#Basic measurements
nrows = 3
ncols = 3
points = 10         # Font size
fig_w_in = 6.5      # Plot width (inches)
panel_wh_ratio = 0.8
panel_lm_in = 0.65
panel_rm_in = 0.15
panel_tm_in = 0.2
panel_bm_in = 0.45
fig_lm_in = 0.
fig_rm_in = 0.
fig_tm_in = 0.
fig_bm_in = -0.9
y_labelpad = 5


panel_w_in = (fig_w_in - fig_lm_in - fig_rm_in)/ncols
panel_h_in = panel_wh_ratio * panel_w_in
fig_h_in = nrows*panel_h_in + fig_tm_in + fig_bm_in

panel_w_s  = panel_w_in  / fig_w_in
panel_h_s  = panel_h_in  / fig_h_in

panel_lm_s = panel_lm_in / fig_w_in
panel_rm_s = panel_rm_in / fig_w_in
panel_tm_s = panel_tm_in / fig_h_in
panel_bm_s = panel_bm_in / fig_h_in

fig_lm_s = fig_lm_in / fig_w_in
fig_rm_s = fig_rm_in / fig_w_in
fig_tm_s = fig_tm_in / fig_h_in
fig_bm_s = fig_bm_in / fig_h_in

pt_w_s = 1/72/fig_w_in
pt_h_s = 1/72/fig_w_in
char_w_s = pt_w_s*points
char_h_s = pt_h_s*points

adj_w_in = 0.05
adj_h_in = 0.2
adj_w_s = adj_w_in/fig_w_in
adj_h_s = adj_h_in/fig_h_in

def bottom_margin(row): 
    return (nrows - (row + 1))*panel_h_s + panel_bm_s + fig_bm_s
def left_margin(col):
    return col*panel_w_s + panel_lm_s + fig_lm_s
def rect(row, col):
    return [left_margin(col), bottom_margin(row), panel_w_s - panel_lm_s - panel_rm_s, panel_h_s - panel_tm_s - panel_bm_s]
def rect_adj(row, col, arow, acol):
    width = panel_w_s - panel_lm_s - panel_rm_s
    height = panel_h_s - panel_tm_s - panel_bm_s
    return [left_margin(col) + acol*(adj_w_s + width), bottom_margin(row) - arow*(adj_h_s+height), width, height]

def label(ax, s):
    lmbm, rmtm = ax.get_position().get_points()
    lm, bm = lmbm
    rm, tm = rmtm
    w = rm - lm
    h = tm - bm
    ax.figure.text(lm-3.3*char_w_s, tm+char_h_s/2, r'\textbf{' + s + r'}')

fig = plt.figure(figsize=(fig_w_in, fig_h_in))


def set_ylabel_coords(ax):
    lmbm, rmtm = ax.get_position().get_points()
    lm, bm = lmbm
    rm, tm = rmtm
    w = rm - lm
    h = tm - bm
    ax.yaxis.set_label_coords(lm-2.5*char_w_s, bm+h/2, transform = ax.figure.transFigure)

alphas = [0.5, 1]
cols = ['b', 'r'] #Blue is trained under IWAE, Red is trained under TMC
titles = ['factorised', 'non-factorised, small', 'non-factorised, large']
lws = [1, 0.3, 0.3]
alphas = [1, 1, 0.3]

#Loop over plots/models (fac/non-fac)
for p in range(3):
    ax = fig.add_axes(rect_adj(0, 0, 0, p))
    ax.set_ylim(bottom=-93.5, top=-90.5)
    ax.set_xlim(0, 1200)
    ax.set_xticks([])
    #ax.set_xlabel('epochs', fontsize=10)
    if p == 0:
        label(ax, "A")
        #pass
        #ax.set_ylabel("objective", fontsize=10)
        #set_ylabel_coords(ax)
    else:
        ax.set_yticks([])
    ax.set_title(titles[p], fontsize=10)

    #Loop over optimization types (std/stl/drg)
    for i in range(3):
        #Loop over training using IWAE/TMC
        for j in range(2):
            #Evaluating under IWAE/TMC
            #ax.plot(dfs[p][i][j]["epoch"], -dfs[p][i][j]["iwae"], cols[j], alpha=0.5, linewidth=lws[i])
            ax.plot(dfs[p][i][j]["epoch"], -dfs[p][i][j]["tmc"], cols[j], alpha=alphas[i], linewidth=lws[i])

ax.text(1., 0.5, '\\noindent Train: IWAE / TMC \\\\ \\indent Eval: TMC', transform=ax.transAxes, clip_on=False, rotation=-90, verticalalignment="center", multialignment="center")
#ax.text(1., 0.5, 'Train: IWAE / TMC', transform=ax.transAxes, clip_on=False, rotation=-90, verticalalignment="center", horizontalalignment="left")
#ax.text(1., 0.5, 'Eval: TMC', transform=ax.transAxes, clip_on=False, rotation=-90, verticalalignment="center", horizontalalignment="right", multialignment="center")

from matplotlib.lines import Line2D
from matplotlib.legend import Legend

import matplotlib.pyplot as plt
import matplotlib.text as mtext


class LegendTitle(object):
    def __init__(self, text_props=None):
        self.text_props = text_props or {}
        super(LegendTitle, self).__init__()

    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        title = mtext.Text(x0, y0, orig_handle, usetex=True, **self.text_props)
        handlebox.add_artist(title)
        return title

leg = ax.legend(
    [   "objective type",
        Line2D([0], [0], color='b'),
        Line2D([0], [0], color='r'),
        #"eval.\\ obj.",
        #Line2D([0], [0], color='k', alpha=0.5),
        #Line2D([0], [0], color='k', alpha=1.0),
        "",
        "var.\\ reduction",
        Line2D([0], [0], color='k', linewidth=lws[0], alpha=alphas[0]),
        Line2D([0], [0], color='k', linewidth=lws[1], alpha=alphas[1]),
        Line2D([0], [0], color='k', linewidth=lws[2], alpha=alphas[2]),
        "",
    ],
    [   "", "IWAE", "TMC",
        #"", "IWAE", "TMC",
        "", 
        "", "none", "STL", "DReGs",
        "", 
    ],
    #ncol=2,
    fontsize=10,
    frameon=False,
    #bbox_to_anchor=(0.82,0.7),
    bbox_to_anchor=(0.975,0.7),
    bbox_transform=fig.transFigure,
    handler_map={str: LegendTitle({'fontsize': 10})}
)
leg.get_title().set_fontsize(8)
#ax.add_artist(leg)


#Trained using IWAE, evaled using both
for p in range(3):
    ax = fig.add_axes(rect_adj(0, 0, 1, p))
    ax.set_ylim(bottom=-93.5, top=-90.5)
    ax.set_xlim(0, 1200)
    ax.set_xticks([])
    #ax.set_xlabel('epochs', fontsize=10)
    if p == 0:
        label(ax, "B")
        ax.set_ylabel("objective value", fontsize=10)
        #set_ylabel_coords(ax)
    else:
        ax.set_yticks([])
    #ax.set_title(titles[p], fontsize=10)

    #Loop over optimization types (std/stl/drg)
    for i in range(3):
        #Loop over training using IWAE/TMC
        j=0
        #Evaluating under IWAE/TMC
        ax.plot(dfs[p][i][j]["epoch"], -dfs[p][i][j]["iwae"], cols[0], alpha=alphas[i], linewidth=lws[i])
        ax.plot(dfs[p][i][j]["epoch"], -dfs[p][i][j]["tmc"], cols[1], alpha=alphas[i], linewidth=lws[i])

ax.text(1., 0.5, "Train: IWAE\\\\Eval: IWAE / TMC", transform=ax.transAxes, clip_on=False, rotation=-90, verticalalignment="center", multialignment="center")

#Trained using TMC, evaled using both.
for p in range(3):
    ax = fig.add_axes(rect_adj(0, 0, 2, p))
    ax.set_ylim(bottom=-93.5, top=-90.5)
    ax.set_xlim(0, 1200)
    ax.set_xticks([0, 1000])
    ax.set_xlabel('epochs', fontsize=10)
    if p == 0:
        label(ax, "C")
        #pass
        #ax.set_ylabel("objective", fontsize=10)
        #set_ylabel_coords(ax)
    else:
        ax.set_yticks([])
    #ax.set_title(titles[p], fontsize=10)

    #Loop over optimization types (std/stl/drg)
    for i in range(3):
        #Loop over training using IWAE/TMC
        j=1
        #Evaluating under IWAE/TMC
        ax.plot(dfs[p][i][j]["epoch"], -dfs[p][i][j]["iwae"], cols[0], alpha=alphas[i], linewidth=lws[i])
        ax.plot(dfs[p][i][j]["epoch"], -dfs[p][i][j]["tmc"], cols[1], alpha=alphas[i], linewidth=lws[i])

ax.text(1., 0.5, "Train: TMC\\\\Eval: IWAE / TMC", transform=ax.transAxes, clip_on=False, rotation=-90, verticalalignment="center", multialignment="center")


fig.savefig(output_filename, dpi=400)
