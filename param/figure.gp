set macro

#### Sizes 
## Configurable
cols = 4                     # Number of columns
rows = 1                     # Number of rows

width = 6.5                  # Plot width (inches)
axis_factor = 0.53            # Proportion of horizontal panel filled by axis
points = 10                  # Font size
height_factor = 0.90         # Ratio of panel height to panel width
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
tm(row) = tp(row) - ch
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

  methods = 'tmc smc vae grt'
  method_labels = '\begin{scriptsize}TMC\end{scriptsize} \begin{scriptsize}SMC\end{scriptsize} \begin{scriptsize}IWAE\end{scriptsize} \begin{scriptsize}GT\end{scriptsize}'

  eval setax("A", 1, 1)
  set style line 1 lc rgb '#E41A1C' # red
  set style line 2 lc rgb '#377EB8' # blue
  set style line 3 lc rgb '#4DAF4A' # green
  set xlabel "K"
  set logscale x
  set format x "$10^{%L}$"
  set xtics autofreq 10**3
  set xrange [10**0:10**6]
  set key at graph 1, 0 bottom right
  set key samplen 1
  set key spacing 0.7
  set key bottom
  set key right
  set key at graph 1.10, 0.00
  set label "$\\log \\operatorname{P}(x)$" @YLAB
  set ytics autofreq 200
  plot for [i=1:4] \
       ARG2 u "K":(((stringcolumn("method") eq word(methods, i)) && (column("N") == 128)) ? column("mean") : 1/0) \
       title word(method_labels, i) \
       linestyle i \
       with l

  eval setax("B", 1, 2)
  set xlabel "K"
  set logscale x
  set format x "$10^{%L}$"
  set xtics autofreq 10**3
  set xrange [10**0:10**6]
  set label "time (s)" @YLAB
  set ytics autofreq 0.2
  set format y "%.2f"
  unset key
  plot for [i=1:3] \
       ARG2 u "K":(((stringcolumn("method") eq word(methods, i)) && (column("N") == 128)) ? column("time") : 1/0) \
       title word(methods, i) \
       linestyle i \
       with l



  eval setax("C", 1, 3)
  set xlabel "N"
  set logscale x
  set format x "$10^{%L}$"
  set xtics autofreq 10**3
  set xrange [10**0:10**6]
  set yrange [-3.5:-1.5]
  unset key
  set label "$\\log \\operatorname{P}(x)/N$" @YLAB
  set ytics autofreq 1
  plot for [i=1:4] \
       ARG2 u "N":(((stringcolumn("method") eq word(methods, i)) && (column("K") == 128)) ? column("mean")/column("N") : 1/0) \
       title word(methods, i) \
       linestyle i \
       with l



  eval setax("D", 1, 4)
  set xlabel "N"
  set logscale x
  set format x "$10^{%L}$"
  set xtics autofreq 10**3
  set xrange [10**0:10**6]
  set label "time (s)" @YLAB
  set format y "$10^{\\phantom{-6}\\mathllap{%L}}$"
  set logscale y
  set yrange [10**(-6):10**0]
  set ytics autofreq 10**(-6), 10**3
  unset key
  plot for [i=1:3] \
       ARG2 u "N":(((stringcolumn("method") eq word(methods, i)) && (column("K") == 128)) ? column("time") : 1/0) \
       title word(methods, i) \
       linestyle i \
       with l

unset multiplot
