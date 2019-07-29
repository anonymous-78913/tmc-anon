set macro

#### Sizes 
## Configurable
cols = 1                     # Number of columns
rows = 1                     # Number of rows

width = 6.5/3                # Plot width (inches)
axis_factor = 0.7           # Proportion of horizontal panel filled by axis
points = 10                  # Font size
height_factor = 0.95          # Ratio of panel height to panel width
rm_distance_pt = 8           # Distance of right margin from end of box
ylab_char_adjustment = 0.6


## Fixed
pw = (1.0)/cols                            # Panel width (screen)
ph = (1.0)/rows                            # Panel height (screen)
aw = axis_factor * pw                      # Axis width (screen)
ah = axis_factor * ph / height_factor      # Axis height (screen)
height = (height_factor*width)/cols * rows # Plot height (inches)
ptw = ((1.0)/72) / width                   # Point width (screen)
pth = ((1.0)/72) / height                  # Point height (screen)
cw = points*ptw                            # Char width (screen)
ch = points*pth                            # Char height (screen)

#Panel top/left
tp(row) = ph*(rows-(row-1))
lp(col) = pw*(col-1)

#Axis margin
tm(row) = tp(row) - 0.5*ch
bm(row) = tm(row) - ah
rm(col) = pw*col - rm_distance_pt*ptw
lm(col) = rm(col) - aw


RESET = "reset; load 'Set1.gp'; set style line 4 lc rgb '#777777'; set border 3; set tics nomirror out; "
set_rc(row, col) = 'row = '.row.'; col = '.col.'; '
tbm = 'set tmargin screen tm(row); set bmargin screen bm(row); '
lrm = 'set lmargin screen lm(col); set rmargin screen rm(col); '
#label(lab) = 'set label "\\textbf{'.lab.'}" offset -8.3,0.5 at screen lm(col), tm(row); '
label(lab) = 'set label "\\textbf{'.lab.'}" at screen lp(col)+cw/10, tp(row)-ch/2; '
setax(lab, row, col) = RESET.set_rc(row, col).tbm.lrm.label(lab)

YLAB = "at screen lp(col) + ylab_char_adjustment*cw, (tm(row) + bm(row))/2 rotate center"

set print "-"
print(tm(1))
print(bm(1))

#### Latex
#set term cairolatex pdf standalone rounded lw 2 size width in,height in \
#  font '\sfdefault,8' \
#  header '\usepackage[scaled]{helvet}\usepackage{xcolor,sfmath,bm,amsmath,amssymb,mathtools}'
set term cairolatex pdf standalone rounded lw 2 size width in,height in \
  font '10' \
  header '\usepackage{times,xcolor,bm,amsmath,amssymb,mathtools}'
set output ARG1

#### Generic formatting

#### Plot
set multiplot 

  methods = 'gt fc nf'
  method_labels = '\begin{scriptsize}ground-truth\end{scriptsize} \begin{scriptsize}factorised\end{scriptsize} \begin{scriptsize}non-factorised\end{scriptsize}'

  eval setax("", 1, 1)
  set style line 1 lc rgb '#888888' # gray
  set style line 2 lc rgb '#E41A1C' # red
  set style line 3 lc rgb '#377EB8' # blue
  set xlabel "K"
  set logscale x
  set format x "$10^{%L}$"
  #set xtics autofreq 10**3
  set xrange [10**0:10**3]
  #set key at graph 0, 0 bottom left
  set key samplen 0.7
  #set key spacing 0.7
  set key bottom
  set key right
  set key at graph 0.60, 0.1
  set label "$\\log \\operatorname{P}(x)$" @YLAB
  set ytics autofreq 5
  set yrange [-10:0]
  plot for [i=1:3] \
       ARG2 u "K":((stringcolumn("fc_nf") eq word(methods, i)) ? column("mean") : 1/0) \
       title word(method_labels, i) \
       linestyle i \
       with l
unset multiplot
