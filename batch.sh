#!/bin/bash

for TEMP in sleepON sleepOFF
do
    for N in 1 2 3 4 5
    do
        echo --------------------------------------
        echo trial ${N}
        python sim.py ${TEMP}
        mv results.npz results${N}.npz
    done
    python analysis_plot_all.py
    mkdir data_${TEMP}
    mv *.npz *.csv *.txt *.tiff data_${TEMP}/
done
