# engramtobe_simulation
Codes for simulations and analyses in Ghandour, Haga, Ohkawa et al., "Dual roles of idling moments for past and future memories".

Codes depend on Python 3, numpy, scipy, matplotlib, and a UNIX shell script (bash). With Anaconda (or Miniconda), you can run `conda env create -f env_engramtobe.yml` then `conda activate engramtobe` to create an appropreate environment.

For running the codes, execute `bash batch.sh` on a UNIX shell (we ran codes in Ubuntu 20.04 and 22.04). Otherwise, manually execute commands written in the `batch.sh`.

Codes execute 5 simulations for each of four conditions (with and without sleep, homeostatic plasticity OFF, LTD-OFF). Each simulation generates a .npz file that contains results. After 5 simulations, results are summarized to output .csv and .tiff files that show cell-type ratios, coincidence and correlation of neural activities, and matching ratios as presented in the paper. 

It took approximately 10 minutes to finish all calculations by a workstation with Xeon-W7-3445.
