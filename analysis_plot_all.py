#!/usr/bin/env python

import sys
import numpy
import scipy.stats
import matplotlib.pyplot as plt

Nshuffle = 100
Nsilentbin = 4000 
silent_amp = 0.01

#load results
Nfile = 5
files = []
for nf in range(Nfile):
    files.append("results"+str(nf+1)+".npz")
#engram, r_som_pre, r_som_awake1, r_som_sleep1, r_som_awake2, preplay_log, replay_log

engram = []
non_engram = []
r_som_awake1 = []
r_som_awake2 = []
preplay_log = []
replay_log = []

for f in files:
    results = numpy.load(f)
    tmp = results["engram"]
    engram.append(tmp)
    non_engram.append(numpy.logical_not(tmp))
    r_som_awake1.append(results["r_som_awake1"])
    r_som_awake2.append(results["r_som_awake2"])
    preplay_log.append(results["preplay_log"])
    replay_log.append(results["replay_log"])

Npattern, N_CA1 = r_som_awake1[0].shape
Nreplay = preplay_log[0].shape[0]

######################## analysis and plot ################################
#classifying cells
activity_count_threshold = 0.5
engram_edu = []
engram_spe = []
engram_tobe = []
nonengram_all = []
for nf in range(Nfile):
    awake2_active = (r_som_awake2[nf][1,:]>activity_count_threshold)
    awake2_inactive = numpy.logical_not(awake2_active)

    engram_edu.append(engram[nf]*awake2_active)
    engram_spe.append(engram[nf]*awake2_inactive)
    engram_tobe.append(non_engram[nf]*awake2_active)
    nonengram_all.append(non_engram[nf]*numpy.logical_not(engram_tobe[nf]))

#save cell type ratio 
celltype_ratio = ["engram, engram-to-be, non-engram\n"]
for nf in range(Nfile):
    edu_ratio = numpy.sum(engram_edu[nf])/N_CA1
    spe_ratio = numpy.sum(engram_spe[nf])/N_CA1
    tobe_ratio = numpy.sum(engram_tobe[nf])/N_CA1
    non_ratio = numpy.sum(nonengram_all[nf])/N_CA1
    celltype_ratio.append(f"data {nf}: {edu_ratio+spe_ratio:.3f}, {tobe_ratio:.3f}, {non_ratio:.3f}\n")
with open("celltype_ratio.txt", "w") as f:        
    f.writelines(celltype_ratio)

#matching_ratio (pattern A)
plt.figure(figsize=(3,3))
corpre_log = dict([])
corpre_log["engram"] = numpy.zeros(Nfile)
corpre_log["non-engram"] = numpy.zeros(Nfile)
corre_log = dict([])
corre_log["engram"] = numpy.zeros(Nfile)
corre_log["non-engram"] = numpy.zeros(Nfile)
for nf in range(Nfile):
    for idx, label, color in [(engram[nf], "engram", "red"), (non_engram[nf], "non-engram", "blue")]:
        preplay_log_norm = preplay_log[nf][:,idx]
        replay_log_norm = replay_log[nf][:,idx]
        preplay_log_norm = preplay_log_norm / numpy.linalg.norm(preplay_log_norm,axis=1,keepdims=True)
        replay_log_norm = replay_log_norm / numpy.linalg.norm(replay_log_norm,axis=1,keepdims=True)
        pattern = r_som_awake1[nf][0,idx]
        pattern = pattern / numpy.linalg.norm(pattern)
        corpre = numpy.mean(numpy.greater(preplay_log_norm@pattern,0.6))
        corre = numpy.mean(numpy.greater(replay_log_norm@pattern,0.6))
        if nf==0:
            plt.plot([0,1], [corpre, corre], "-o", label=label, color=color)
        else:
            plt.plot([0,1], [corpre, corre], "-o", color=color)
        corpre_log[label][nf] = corpre
        corre_log[label][nf] = corre
plt.title("Pattern A")
plt.ylabel("Matching ratio")
plt.xticks([0,1], ["pre-sleep", "post-sleep"])
plt.xlim([-0.2, 1.2])
plt.ylim([-0.05,1.05])
plt.legend(loc="center right")
plt.gca().spines["right"].set_visible(False)
plt.gca().spines["top"].set_visible(False)
plt.tight_layout()
plt.savefig("matching_ratio_A.tiff")
plt.close()
#save results in csv
numpy.savetxt("matching_ratio_A_presleep.csv", numpy.vstack([corpre_log["engram"], corpre_log["non-engram"]]).T, delimiter=",", fmt="%f")
numpy.savetxt("matching_ratio_A_postsleep.csv", numpy.vstack([corre_log["engram"], corre_log["non-engram"]]).T, delimiter=",", fmt="%f")

#matching ratio (pattern B)
plt.figure(figsize=(3,3))
corpre_log = dict([])
corpre_log["engram+engram-to-be"] = numpy.zeros(Nfile)
corpre_log["other non-engram"] = numpy.zeros(Nfile)
corre_log = dict([])
corre_log["engram+engram-to-be"] = numpy.zeros(Nfile)
corre_log["other non-engram"] = numpy.zeros(Nfile)
for nf in range(Nfile):
    for idx, label, color in [(numpy.logical_not(nonengram_all[nf]), "engram+engram-to-be", "red"), (nonengram_all[nf], "other non-engram", "blue")]:
    #for idx, label, color in [(numpy.ones_like(nonengram_all[nf], dtype=bool), "engram+engram-to-be", "red"), (nonengram_all[nf], "other non-engram", "blue")]:
        preplay_log_norm = preplay_log[nf][:,idx]
        replay_log_norm = replay_log[nf][:,idx]
        preplay_log_norm = preplay_log_norm / numpy.linalg.norm(preplay_log_norm,axis=1,keepdims=True)
        replay_log_norm = replay_log_norm / numpy.linalg.norm(replay_log_norm,axis=1,keepdims=True)
        pattern = r_som_awake2[nf][1,idx]
        pattern = pattern / numpy.linalg.norm(pattern)
        corpre = numpy.mean(numpy.greater(preplay_log_norm@pattern,0.6))
        corre = numpy.mean(numpy.greater(replay_log_norm@pattern,0.6))
        if nf==0:
            plt.plot([0,1], [corpre, corre], "-o", label=label, color=color)
        else:
            plt.plot([0,1], [corpre, corre], "-o", color=color)
        corpre_log[label][nf] = corpre
        corre_log[label][nf] = corre
plt.title("Pattern B")
plt.ylabel("Matching ratio")
plt.xticks([0,1], ["pre-sleep", "post-sleep"])
plt.xlim([-0.2, 1.2])
plt.ylim([-0.05,1.05])
plt.yticks(0.1*numpy.arange(4))
plt.legend(loc="upper right")
plt.gca().spines["right"].set_visible(False)
plt.gca().spines["top"].set_visible(False)
plt.tight_layout()
plt.savefig("matching_ratio_B.tiff")
plt.close()
#save results in csv
numpy.savetxt("matching_ratio_B_presleep.csv", numpy.vstack([corpre_log["engram+engram-to-be"], corpre_log["other non-engram"]]).T, delimiter=",", fmt="%f")
numpy.savetxt("matching_ratio_B_postsleep.csv", numpy.vstack([corre_log["engram+engram-to-be"], corre_log["other non-engram"]]).T, delimiter=",", fmt="%f")

##### mix silent bins #####
for nf in range(Nfile):
    preplay_log[nf] = numpy.concatenate( [ preplay_log[nf], silent_amp*numpy.random.rand(Nsilentbin, preplay_log[nf].shape[1]) ] )
    preplay_log[nf] = preplay_log[nf][numpy.random.permutation(preplay_log[nf].shape[0]),:]
    replay_log[nf] = numpy.concatenate( [ replay_log[nf], silent_amp*numpy.random.rand(Nsilentbin, replay_log[nf].shape[1]) ] )
    replay_log[nf] = replay_log[nf][numpy.random.permutation(replay_log[nf].shape[0]),:]

#correlation
plt.figure(figsize=(3,3))
cormean_log_pre = numpy.zeros([3,Nfile])
cormean_log_post = numpy.zeros([3,Nfile])
for nf in range(Nfile):
    xpos = 0
    for idx, label, color in [(engram[nf], "engram", "red"), (engram_tobe[nf], "engram-to-be", "green"), (nonengram_all[nf], "non-engram", "blue")]:
        #pre-sleep
        data = preplay_log[nf][:,idx]
        filt = numpy.logical_not(numpy.eye(data.shape[1], dtype=bool))
        cor = numpy.corrcoef(data.T)
        cor = cor[filt]
        cormean_pre = numpy.mean(cor)
        cormean_log_pre[xpos,nf] = cormean_pre
        #post-sleep
        data = replay_log[nf][:,idx]
        filt = numpy.logical_not(numpy.eye(data.shape[1], dtype=bool))
        cor = numpy.corrcoef(data.T)
        cor = cor[filt]
        cormean_post = numpy.mean(cor)
        cormean_log_post[xpos,nf] = cormean_post
        #plot
        if nf==0:
            plt.plot([0,1], [cormean_pre, cormean_post], "-o", label=label, color=color)
        else:
            plt.plot([0,1], [cormean_pre, cormean_post], "-o", color=color)
        xpos += 1
plt.ylabel("Correlation")
plt.xticks([0,1], ["pre-sleep", "post-sleep"])
plt.xlim([-0.2, 1.2])
plt.ylim([-0.05, 1.15])
plt.legend(loc="upper left")
plt.gca().spines["right"].set_visible(False)
plt.gca().spines["top"].set_visible(False)
plt.tight_layout()
plt.savefig("correlation.tiff")
plt.close()
#save results in csv
numpy.savetxt("correlation_presleep.csv", cormean_log_pre.T, delimiter=",", fmt="%f")
numpy.savetxt("correlation_postsleep.csv", cormean_log_post.T, delimiter=",", fmt="%f")

#coincidence ratio (post-sleep)
plt.figure(figsize=(6,3))
cormean_log = numpy.zeros([4,Nfile])
for nf in range(Nfile):
    xpos = 0
    for idx1, idx2 in [[engram_edu[nf],engram_tobe[nf]], [engram_spe[nf], engram_tobe[nf]], [engram_edu[nf], nonengram_all[nf]], [engram_spe[nf], nonengram_all[nf]]]:
        data1 = replay_log[nf][:,idx1]
        data2 = replay_log[nf][:,idx2]
        data1 = numpy.mean(data1, axis=1)
        data2 = numpy.mean(data2, axis=1)
        #coincidence
        data_norm = numpy.min([numpy.sum(data1),numpy.sum(data2)])
        cor = numpy.sum(data1*data2)/data_norm
        cormean = cor
        #shuffle
        cormean_shuffle = 0.0
        for r in range(Nshuffle):
            data1_shuffle = numpy.random.permutation(data1)
            data2_shuffle = numpy.random.permutation(data2)
            cor_shuffle = numpy.sum(data1_shuffle*data2_shuffle)/data_norm
            cormean_shuffle += cor_shuffle/Nshuffle
        cormean -= cormean_shuffle
        #plot
        plt.plot(xpos, cormean, "o", color="black")
        cormean_log[xpos,nf] = cormean
        xpos += 1
plt.ylabel("Coincidence ratio (normalized)")
plt.xticks([0,1,2,3], ["engram-educating\nengram-to-be", "engram-specific\nengram-to-be", "engram-educating\nnon-engram", "engram-specific\nnon-engram"])
plt.xlim([-0.5, 3.5])
plt.gca().spines["right"].set_visible(False)
plt.gca().spines["top"].set_visible(False)
plt.tight_layout()
plt.savefig("coincidence_postsleep.tiff")
plt.close()
#save results in csv
numpy.savetxt("coincidence_postsleep.csv", cormean_log.T, delimiter=",", fmt="%f")

