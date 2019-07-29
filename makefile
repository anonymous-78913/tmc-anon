run = exec
local = singularity exec --nv ~/Dropbox/singularity/singularity16.img
cluster = run_lsf -c 5 -g 1 ./gpu_info.py singularity exec --nv ~/Dropbox/singularity/singularity16.img
#cluster = bsub -Is -n 5 -gpu "num=1:mps=no" -R"rusage[mem=25600]" -q slowpoke -m c04u07 singularity exec --nv ~/Dropbox/singularity/singularity18.img

cpus = $(shell python3 param/deps.py cpu)
gpus = $(shell python3 param/deps.py gpu)



param/%.row: param/run.py param/lvm.py
	$(local) python $< $@ $(wordlist 2, 100, $^)

param/cpu.csv: param/cat.py $(cpus)
	$(local) python $< $@ $(wordlist 2, 100, $^)

param/gpu.csv: param/cat.py $(gpus)
	$(local) python $< $@ $(wordlist 2, 999999, $^)

param/figure.tex: param/figure.gp param/gpu.csv
	$(local) gnuplot -c $< $@ $(wordlist 2, 999999, $^)

param/figure.pdf: param/figure.tex param/gpu.csv
	$(local) pdflatex --shell-escape --jobname=$(basename $@) $^



nfs = $(shell python3 non-fac/deps.py cpu)

non-fac/%.row: non-fac/run.py non-fac/model.py
	$(cluster) python $< $@ $(wordlist 2, 100, $^)

non-fac/results.csv: non-fac/cat.py $(nfs)
	$(local) python $< $@ $(wordlist 2, 1000000, $^)

non-fac/results_agg.csv: non-fac/agg.R non-fac/results.csv
	$(local) Rscript $< $@ $(wordlist 2, 100, $^)

non-fac/figure.tex: non-fac/figure.gp non-fac/results_agg.csv
	$(local) gnuplot -c $< $@ $(wordlist 2, 999999, $^)

non-fac/figure.pdf: non-fac/figure.tex
	$(local) pdflatex --shell-escape --jobname=$(basename $@) $^

vae/%.csv: vae/main.py
	$(local) python $< $@ $(wordlist 2, 999999, $^)

#vae/figure.pdf: vae/figure.py vae/fac_iwa_std.csv vae/fac_iwa_stl.csv vae/fac_iwa_drg.csv vae/fac_iwa_rws.csv vae/fac_tmc_std.csv vae/fac_tmc_stl.csv vae/fac_tmc_drg.csv vae/fac_tmc_rws.csv vae/nfs_iwa_std.csv vae/nfs_iwa_stl.csv vae/nfs_iwa_drg.csv vae/nfs_iwa_rws.csv vae/nfs_tmc_std.csv vae/nfs_tmc_stl.csv vae/nfs_tmc_drg.csv vae/nfs_tmc_rws.csv vae/nfl_iwa_std.csv vae/nfl_iwa_stl.csv vae/nfl_iwa_drg.csv vae/nfl_iwa_rws.csv vae/nfl_tmc_std.csv vae/nfl_tmc_stl.csv vae/nfl_tmc_drg.csv vae/nfl_tmc_rws.csv 
vae/figure.pdf: vae/figure.py vae/fac_iwa_std.csv vae/fac_iwa_stl.csv vae/fac_iwa_drg.csv vae/fac_tmc_std.csv vae/fac_tmc_stl.csv vae/fac_tmc_drg.csv vae/nfs_iwa_std.csv vae/nfs_iwa_stl.csv vae/nfs_iwa_drg.csv vae/nfs_tmc_std.csv vae/nfs_tmc_stl.csv vae/nfs_tmc_drg.csv vae/nfl_iwa_std.csv vae/nfl_iwa_stl.csv vae/nfl_iwa_drg.csv vae/nfl_tmc_std.csv vae/nfl_tmc_stl.csv vae/nfl_tmc_drg.csv
	$(local) ipython $< $@ $(wordlist 2, 100, $^)
	#$(local) ipython -i --matplotlib tk $< $@ $(wordlist 2, 100, $^)

vae/time.pdf: vae/time.py vae/time/fac_iwa_std.csv vae/time/fac_iwa_stl.csv vae/time/fac_iwa_drg.csv vae/time/fac_tmc_std.csv vae/time/fac_tmc_stl.csv vae/time/fac_tmc_drg.csv vae/time/nfs_iwa_std.csv vae/time/nfs_iwa_stl.csv vae/time/nfs_iwa_drg.csv vae/time/nfs_tmc_std.csv vae/time/nfs_tmc_stl.csv vae/time/nfs_tmc_drg.csv vae/time/nfl_iwa_std.csv vae/time/nfl_iwa_stl.csv vae/time/nfl_iwa_drg.csv vae/time/nfl_tmc_std.csv vae/time/nfl_tmc_stl.csv vae/time/nfl_tmc_drg.csv
	$(local) ipython $< $@ $(wordlist 2, 100, $^)
	#$(local) ipython -i --matplotlib tk $< $@ $(wordlist 2, 100, $^)

#vae/fac.csv

manuscript.pdf: paper.pdf
	gs -sDEVICE=pdfwrite -dNOPAUSE -dBATCH -dSAFER -dLastPage=10 -sOutputFile=$@ $<

supp.pdf: paper.pdf
	gs -sDEVICE=pdfwrite -dNOPAUSE -dBATCH -dSAFER -dFirstPage=11 -sOutputFile=$@ $<
