Full reproduction requires Python, PyTorch, R, GNUPlot and Latex

To reproduce Figure 1, run:
```
make param/figure.pdf
```

To reproduce Figure 2, run:
```
make non-fac/figure.pdf
```

To reproduce Figure 3, run:
```
make vae/figure.pdf
```

To run the additional experiment which uses a random factor-graph, use:
ipython complex/model.py

Of most interest is lines 250-350 in vae/main.py, which describe how to compute the various estimators.  However, this is special-case code written for this particular proof-of-concept.  I am currently working on a general codebase, with a probabilistic programming language that embodies these ideas.
