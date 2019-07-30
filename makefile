prefix = exec #singularity exec --nv ~/Dropbox/singularity/singularity16.img

cpus = $(shell python3 param/deps.py cpu)
gpus = $(shell python3 param/deps.py gpu)

param/%.row: param/run.py param/lvm.py
	$(prefix) python $< $@ $(wordlist 2, 100, $^)

param/cpu.csv: param/cat.py $(cpus)
	$(prefix) python $< $@ $(wordlist 2, 100, $^)

param/gpu.csv: param/cat.py $(gpus)
	$(prefix) python $< $@ $(wordlist 2, 999999, $^)

param/figure.tex: param/figure.gp param/gpu.csv
	$(prefix) gnuplot -c $< $@ $(wordlist 2, 999999, $^)

param/figure.pdf: param/figure.tex param/gpu.csv
	$(prefix) pdflatex --shell-escape --jobname=$(basename $@) $^



nfs = $(shell python3 non-fac/deps.py cpu)

non-fac/%.row: non-fac/run.py non-fac/model.py
	$(prefix) python $< $@ $(wordlist 2, 100, $^)

non-fac/results.csv: non-fac/cat.py $(nfs)
	$(prefix) python $< $@ $(wordlist 2, 1000000, $^)

non-fac/results_agg.csv: non-fac/agg.R non-fac/results.csv
	$(prefix) Rscript $< $@ $(wordlist 2, 100, $^)

non-fac/figure.tex: non-fac/figure.gp non-fac/results_agg.csv
	$(prefix) gnuplot -c $< $@ $(wordlist 2, 999999, $^)

non-fac/figure.pdf: non-fac/figure.tex
	$(prefix) pdflatex --shell-escape --jobname=$(basename $@) $^




vae/%.csv: vae/main.py
	$(prefix) python $< $@ $(wordlist 2, 999999, $^)

vae/figure.pdf: vae/figure.py vae/fac_iwa_std.csv vae/fac_iwa_stl.csv vae/fac_iwa_drg.csv vae/fac_tmc_std.csv vae/fac_tmc_stl.csv vae/fac_tmc_drg.csv vae/nfs_iwa_std.csv vae/nfs_iwa_stl.csv vae/nfs_iwa_drg.csv vae/nfs_tmc_std.csv vae/nfs_tmc_stl.csv vae/nfs_tmc_drg.csv vae/nfl_iwa_std.csv vae/nfl_iwa_stl.csv vae/nfl_iwa_drg.csv vae/nfl_tmc_std.csv vae/nfl_tmc_stl.csv vae/nfl_tmc_drg.csv
	$(prefix) ipython $< $@ $(wordlist 2, 100, $^)

vae/time.pdf: vae/time.py vae/time/fac_iwa_std.csv vae/time/fac_iwa_stl.csv vae/time/fac_iwa_drg.csv vae/time/fac_tmc_std.csv vae/time/fac_tmc_stl.csv vae/time/fac_tmc_drg.csv vae/time/nfs_iwa_std.csv vae/time/nfs_iwa_stl.csv vae/time/nfs_iwa_drg.csv vae/time/nfs_tmc_std.csv vae/time/nfs_tmc_stl.csv vae/time/nfs_tmc_drg.csv vae/time/nfl_iwa_std.csv vae/time/nfl_iwa_stl.csv vae/time/nfl_iwa_drg.csv vae/time/nfl_tmc_std.csv vae/time/nfl_tmc_stl.csv vae/time/nfl_tmc_drg.csv
	$(prefix) ipython $< $@ $(wordlist 2, 100, $^)
