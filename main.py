"""
This script simulates two integrate-and-fire neurons coupled via synapses, including the refractory period 
(variable threshold). The neurons are analyzed for their response to constant input current, and the synchrony 
of their spikes is evaluated. The model uses typical parameters for synaptic conductances and membrane properties.
"""

import numpy as np
import matplotlib.pyplot as plt

# Parameters
E0 = -65  # mV, Resting potential
taum = 30  # ms, Membrane time constant (10-30ms)
taut = 5  # ms, Threshold time constant
taus1 = 10  # ms, Synaptic time constant for neuron 1
taus2 = 10  # ms, Synaptic time constant for neuron 2
r = 10  # Mohm, Membrane resistance
g = 1 / r
C = taum / r
dt = 0.01  # Time step
tmax = 150  # ms, Maximum time

# Threshold potentials
Vtl = -55  # mV, Lower threshold
Vth = 50  # mV, Upper threshold

# Synaptic parameters
dPs1 = 0.6  # (0.03-0.6)
dPs2 = 0.2
Es1 = -70  # mV, Reversal potential for synapse 1
Es2 = 0  # mV, Reversal potential for synapse 2
gsmax1 = 5 / r  # Maximum synaptic conductance for synapse 1
gsmax2 = 5 / r  # Maximum synaptic conductance for synapse 2

# Time vector
t = np.arange(0, tmax + dt, dt)
L = len(t)

# Initialize variables
V1 = np.zeros((L,))
Vt1 = np.zeros((L,))
Ps1 = np.zeros((L,))
V2 = np.zeros((L,))
Vt2 = np.zeros((L,))
Ps2 = np.zeros((L,))

# Input currents
I1 = 4  # nA
I2 = 4  # nA

# Initial conditions
V1[0] = E0  # mV
V2[0] = E0  # mV
Vt1[0] = Vtl  # mV
Vt2[0] = Vtl  # mV
Ps1[0] = 0
Ps2[0] = 0

gs1 = gsmax1 * Ps1[0]
gs2 = gsmax2 * Ps2[0]

# Simulation
idx_spike1 = []
idx_spike2 = []
for k in np.arange(L - 1):
    geq1 = g + gs1
    geq2 = g + gs2
    E0tot1 = (g * E0 + gs1 * Es1) / geq1
    E0tot2 = (g * E0 + gs2 * Es2) / geq2
    rtot1 = 1 / geq1
    rtot2 = 1 / geq2
    Vinf1 = E0tot1 + rtot1 * I1
    Vinf2 = E0tot2 + rtot2 * I2
    tau1 = C * rtot1
    tau2 = C * rtot2
    V1[k + 1] = (V1[k] - Vinf1) * np.exp(-dt / tau1) + Vinf1
    Vt1[k + 1] = (Vt1[k] - Vtl) * np.exp(-dt / taut) + Vtl
    Ps1[k + 1] = Ps1[k] * np.exp(-dt / taus1)
    gs1 = gsmax1 * Ps1[k + 1]
    V2[k + 1] = (V2[k] - Vinf2) * np.exp(-dt / tau2) + Vinf2
    Vt2[k + 1] = (Vt2[k] - Vtl) * np.exp(-dt / taut) + Vtl
    Ps2[k + 1] = Ps2[k] * np.exp(-dt / taus2)
    gs2 = gsmax2 * Ps2[k + 1]
    if V1[k + 1] > Vt1[k + 1]:
        V1[k + 1] = E0
        Vt1[k + 1] = Vth
        Ps2[k + 1] += dPs2 * (1 - Ps2[k + 1])
        idx_spike1.append(k + 1)
    if V2[k + 1] > Vt2[k + 1]:
        V2[k + 1] = E0
        Vt2[k + 1] = Vth
        Ps1[k + 1] += dPs1 * (1 - Ps1[k + 1])
        idx_spike2.append(k + 1)

# Spike timings and frequencies
if len(idx_spike1) > 0:
    T1 = t[idx_spike1[-1]] - t[idx_spike1[-2]]
    f1 = 1 / T1 * 1000

if len(idx_spike2) > 0:
    T2 = t[idx_spike2[-1]] - t[idx_spike2[-2]]
    f2 = 1 / T2 * 1000

spikes1 = np.zeros((L,))
spikes1[idx_spike1] = 1
spikes2 = np.zeros((L,))
spikes2[idx_spike2] = 1

# Plotting
plt.figure(figsize=(11, 8))
plt.subplot(2, 1, 1)
plt.plot(t, V1, 'k')
plt.plot(t, Vt1, 'r')
plt.title('Neuron 1')
plt.xlabel('Time (ms)')
plt.ylabel('Voltage (mV)')
plt.legend(['V', 'Vt'], loc='upper right')

plt.subplot(2, 1, 2)
plt.plot(t, V2, 'k')
plt.plot(t, Vt2, 'r')
plt.title('Neuron 2')
plt.xlabel('Time (ms)')
plt.ylabel('Voltage (mV)')
plt.legend(['V', 'Vt'], loc='upper right')

plt.suptitle('Membrane Potentials')
plt.tight_layout()
plt.show()

plt.figure(figsize=(11, 8))
plt.plot(t, V1, 'k', linewidth=1)
plt.plot(t, V2, 'b', linewidth=1)
plt.plot(t, Vt1, 'r', linewidth=1)
plt.xlabel('Time (ms)')
plt.ylabel('Voltage (mV)')
plt.legend(['V1', 'V2', 'Vt1'], loc='upper right')
plt.ylim([-66, -45])
plt.title('Potential Neurons 1 and 2')
plt.show()

plt.figure(figsize=(11, 8))
plt.subplot(2, 1, 1)
plt.plot(t, spikes1, 'k')
plt.xlabel('Time (ms)')
plt.title('Neuron 1 Spikes')
plt.ylim([0, 1.1])
plt.subplot(2, 1, 2)
plt.plot(t, spikes2, 'k')
plt.xlabel('Time (ms)')
plt.title('Neuron 2 Spikes')
plt.ylim([0, 1.1])

plt.suptitle('Spike Trains')
plt.tight_layout()
plt.show()

plt.figure(figsize=(11, 8))
plt.subplot(2, 1, 1)
plt.plot(t, Ps1, 'k', linewidth=1)
plt.xlabel('Time (ms)')
plt.title('Neuron 1 Synaptic Conductance')
plt.subplot(2, 1, 2)
plt.plot(t, Ps2, 'k', linewidth=1)
plt.xlabel('Time (ms)')
plt.title('Neuron 2 Synaptic Conductance')

plt.suptitle('Synaptic Conductances')
plt.tight_layout()
plt.show()
