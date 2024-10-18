#!/usr/bin/env python

import sys
import numpy
import matplotlib.pyplot as plt

if sys.argv[1] == "sleepON":
    print("offline plasticity ON.")
    homeoON = True
    LTDON = True
elif sys.argv[1] == "sleepOFF":
    print("offline plasticity OFF.")
    homeoON = False
    LTDON = False
elif sys.argv[1] == "homeostasisOFF":
    print("homeostatic plasticity OFF.")
    homeoON = False
    LTDON = True
elif sys.argv[1] == "sleepLTDOFF":
    print("sleepLTD OFF.")
    homeoON = True
    LTDON = False
else:
    print("please specify sleepON, sleepOFF, homeostasisOFF, sleepLTDOFF.")
    exit()

#parameters
N_CA1 = 400
N_CA3 = 400
N_inh = 100
p_CA3 = 0.1
p_EtoI = 0.1
p_ItoE = 0.05

wbase_CA3 = 5.0 / (N_CA3*p_CA3)
wbase_EtoI = 16.0 /(N_CA1*p_EtoI)
wbase_ItoE = 8.0 /(N_inh*p_ItoE)
eta = 2.0 / (N_CA3*p_CA3)
engram_threshold = 0.5

beta_som = 5.0
thr_som = 1.0
noise_amp = 0.5

Npattern = 3
Nreplay = 1000
replay_bias = 0.8

tau = 2.0 #ms
sim_pitch = 0.1 #ms
sim_len_ms = 20.0 #ms
sim_len_sample = int(sim_len_ms/sim_pitch)
timearr_ms = sim_len_ms/sim_len_sample*numpy.arange(sim_len_sample)

#weights
w_CA3 = wbase_CA3 * numpy.random.rand(N_CA1, N_CA3)
w_EtoI = wbase_EtoI * (numpy.random.rand(N_inh, N_CA1)<p_EtoI)
w_ItoE = wbase_ItoE * (numpy.random.rand(N_CA1, N_inh)<p_ItoE)

#input patterns
pattern_CA3 = (numpy.random.rand(Npattern, N_CA3)<p_CA3)

#functions
def step(x):
    return x>0
def sigmoid(x, beta, thr):
    return 1.0/(1.0+numpy.exp(-beta*(x-thr)))
def relu(x):
    return numpy.maximum(x, 0.0)
def somrate(x):
    return sigmoid(x, beta_som, thr_som)
def fluctuate(N):
    return 0.5+1.0*numpy.random.rand(N)

def simulation_batch(r_CA3):
    #initialize activity
    r_exc = numpy.zeros(N_CA1)
    r_inh = numpy.zeros(N_inh)
    #setting inputs
    CA3_input = w_CA3 @ r_CA3 + noise_amp*numpy.random.randn(N_CA1)
    #simulation
    for t in range(sim_len_sample):
        input_E = CA3_input - w_ItoE@r_inh
        input_I = w_EtoI@r_exc
        r_exc = r_exc + sim_pitch * (-r_exc + somrate(input_E)) / tau
        r_inh = r_inh + sim_pitch * (-r_inh + somrate(input_I)) / tau
    #results:
    return r_exc

######### simulation ###########
print("simulation start")

#before learning
preplay_log = numpy.zeros([Nreplay, N_CA1])
print("pre-learning")
for i in range(Nreplay):
    CA3_activity = (numpy.random.rand(N_CA3)<p_CA3) * fluctuate(N_CA3)
    s = simulation_batch(CA3_activity)
    preplay_log[i,:] = s
r_som_pre = numpy.zeros([Npattern, N_CA1])
for i in range(Npattern):
    r_som_pre[i,:] = simulation_batch(pattern_CA3[i,:])

#awake 1 
print("awake 1")
LTPtag = numpy.zeros_like(w_CA3)
#learning
CA3_activity = pattern_CA3[0,:]
s = simulation_batch(CA3_activity)
engram = (s>engram_threshold)
w_CA3 = w_CA3 + eta * numpy.outer(engram, CA3_activity)
non_engram = numpy.logical_not(engram)
print("ratio of engram: ", numpy.mean(engram))
#response
r_som_awake1 = numpy.zeros([Npattern, N_CA1])
for i in range(Npattern):
    r_som_awake1[i,:] = simulation_batch(pattern_CA3[i,:])

#sleep
print("sleep")
CA3_engram = pattern_CA3[0,:]
CA3_nonengram = numpy.logical_not(CA3_engram)

#offline plasticity
#SWR-LTD
if LTDON:
    w_CA3 = relu(w_CA3 - eta * numpy.outer(non_engram, CA3_engram))
#scaling
if homeoON:
    w_CA3 = relu(w_CA3 - eta * numpy.outer(engram, CA3_nonengram))
if homeoON and LTDON:
    w_CA3 = relu(w_CA3 + eta * numpy.outer(non_engram, CA3_nonengram))
    #no LTD in non-engram -> no LTP

#simulation of activity
replay_log = numpy.zeros([Nreplay, N_CA1])
for i in range(Nreplay):
    if numpy.random.rand()<replay_bias:
        CA3_activity = pattern_CA3[0,:] * fluctuate(N_CA3)
    else:
        CA3_activity = (numpy.random.rand(N_CA3)<p_CA3) * fluctuate(N_CA3)
    s = simulation_batch(CA3_activity)
    replay_log[i,:] = s
r_som_sleep1 = numpy.zeros([Npattern, N_CA1])
for i in range(Npattern):
    r_som_sleep1[i,:] = simulation_batch(pattern_CA3[i,:])

#awake 2
print("awake 2")
#learning
CA3_activity = pattern_CA3[1,:]
s = simulation_batch(CA3_activity)
engram2 = (s>engram_threshold)
w_CA3 = w_CA3 + eta * numpy.outer(engram2, CA3_activity)
non_engram2 = numpy.logical_not(engram2)
#simulation
r_som_awake2 = numpy.zeros([Npattern, N_CA1])
for i in range(Npattern):
    r_som_awake2[i,:] = simulation_batch(pattern_CA3[i,:])

#sleep 2
print("sleep 2")
CA3_engram2 = pattern_CA3[1,:]
CA3_nonengram2 = numpy.logical_not(CA3_engram2)

#SWR-LTD
w_CA3 = relu(w_CA3 - eta * numpy.outer(non_engram2, CA3_engram2))
#scaling
w_CA3 = relu(w_CA3 - eta * numpy.outer(engram2, CA3_nonengram2))
w_CA3 = relu(w_CA3 + eta * numpy.outer(non_engram2, CA3_nonengram2))

#simulation of activity
replay_log_afterB = numpy.zeros([Nreplay, N_CA1])
for i in range(Nreplay):
    if numpy.random.rand()<replay_bias:
        CA3_activity = pattern_CA3[1,:] * fluctuate(N_CA3)
    else:
        CA3_activity = (numpy.random.rand(N_CA3)<p_CA3) * fluctuate(N_CA3)
    s = simulation_batch(CA3_activity)
    replay_log_afterB[i,:] = s
r_som_sleep2 = numpy.zeros([Npattern, N_CA1])
for i in range(Npattern):
    r_som_sleep2[i,:] = simulation_batch(pattern_CA3[i,:])

print("simulation end.")

#save results
results = []
numpy.savez("results.npz", 
    engram=engram, 
    r_som_pre=r_som_pre, 
    r_som_awake1=r_som_awake1, 
    r_som_sleep1=r_som_sleep1, 
    r_som_awake2=r_som_awake2,
    r_som_sleep2=r_som_sleep2, 
    preplay_log=preplay_log, 
    replay_log=replay_log, 
    replay_log_afterB=replay_log_afterB,
    )
