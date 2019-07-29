import sys
import numpy as np
import pandas as pd

import matplotlib
import matplotlib.colors as colors
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
plt.switch_backend('PDF')
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

dfl = [np.mean(pd.read_csv(f)["time"]) for f in input_filenames]
dfs = [
    list(zip(dfl[:3],    dfl[3:6])), 
    list(zip(dfl[6:9],   dfl[9:12])),
    list(zip(dfl[12:15], dfl[15:18])),
]
#dfs["fac/nfs/nfl"]["std/stl/drg"]["iwae/tmc"]

dfs2 = np.reshape(np.array(dfl), (3,2,3))
#dfs["fac/nfs/nfl"]["iwae/tmc"]["std/stl/drg"]

iwa_tmc = ["IWAE", "TMC"]
std_stl_drg = ["Standard", "STL", "DReGs"]






#Basic measurements
nrows = 1
ncols = 1
points = 10         # Font size
fig_w_in = 6.5/3      # Plot width (inches)
panel_wh_ratio = 0.8
panel_lm_in = 0.65
panel_rm_in = 0.15
panel_tm_in = 0.2
panel_bm_in = 0.45
fig_lm_in = 0.
fig_rm_in = 0.
fig_tm_in = 0.
fig_bm_in = 0.
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

ticks = np.arange(3)

ax = fig.add_axes(rect_adj(0, 0, 0, 0))
data = np.mean(dfs2, axis=0)

cols = ['b', 'r']
ax.bar(ticks-0.1, data[0, :], color=cols[0], width=0.2)
ax.bar(ticks+0.1, data[1, :], color=cols[1], width=0.2)
ax.set_xticks(ticks)
ax.set_xticklabels(("none", "STL", "DReGs"))
#ax.hlines(data[1, 0], *ax.get_xlim())
ax.set_ylabel("runtime (s)")
ax.set_xlabel("var. reduction")
ax.set_ylim(0, 80)
ax.legend(
    ["IWAE", "TMC"],
    fontsize=10,
    frameon=False,
    loc='upper left',
    bbox_to_anchor=(0., 1.),
    borderpad=0.,
)


fig.savefig(output_filename, dpi=400)

