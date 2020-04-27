import random as rnd
import numpy as np
import matplotlib.pyplot as plt

def load_data(filename,T):
    data_array = [T(line.strip()) for line in open(filename, 'r')]
    return data_array

def get_spike_train(rate,T,tau_ref):

    if 1<=rate*tau_ref:
        print("firing rate not possible given refractory period f/p")
        return []

    exp_rate=rate/(1-tau_ref*rate)
    spike_train=[]
    t=rnd.expovariate(exp_rate)

    while t< T:
        spike_train.append(t)
        t+=tau_ref+rnd.expovariate(exp_rate)
    return spike_train

# functions for calculating spike train, fano factor, coefficient variation, stimulus average non adjacent and adjacent

def spike_train_time(spikes, tau_ref):
    t = 0
    spike_train = []

    for spike in spikes:
        if(spike != 0):
            spike_train.append(spike*t)
        t += tau_ref

    return spike_train


def spike_counts(spikes, interval, T):
    interval_sec = interval / 1000
    count = np.diff([np.count_nonzero(spikes < t) for t in np.arange(0, T, interval_sec)])

    return count


def cal_fano_factor(spikes, counting, T):
    ls = []

    for i in counting:
        counts = spike_counts(spikes, i, T)
        ls.append(np.var(counts) / np.mean(counts))
    return ls


def calc_coef_var(spikes):
    interspike = np.diff(spikes)
    return np.std(interspike) / np.mean(interspike)


def acorr(n):
    # x = x -np.mean(x)
    result = np.correlate(n, n, mode='full')
    #     n = n -np.mean(n)
    #     result = np.correlate(n,n, mode='full')[len(n)-1:]
    #     result /= np.max(result)
    return result[int((result.size/2)-50):int((result.size/2)+50)]

def non_adj_stim_avg(spikes, stim, intervals):

    for t in intervals:
        blocks = []
        for i, x in enumerate(spikes):
            if x == 1 and i > 50 and spikes[i+t] == 1:
                blocks.append(stim[i-50+t:i+t])
        avg = [x / len(blocks) for x in np.sum(np.array(blocks), 0)]
        plt.plot(np.arange(0, 100, 2), avg, label=str(t*2) + 'ms')

    plt.title('Q5-1: STA for pairs of spikes (not necessarily adjacent)')
    plt.xlabel('Time/ms')
    plt.ylabel('Stimulus Average')
    plt.show()


def adj_stim_avg(spikes, stim, intervals):

    for t in intervals:
        blocks = []
        for i in range(len(spikes)):
            if i > 50 and spikes[i] == 1 and spikes[i+t] == 1 and 1 not in spikes[i+1:i+t]:
                blocks.append(stim[i-50+t:i+t])
        avg = [x / len(blocks) for x in np.sum(np.array(blocks), 0)]
        plt.plot(np.arange(0, 100, 2), avg, label=str(t*2) + 'ms')

    plt.title('Q5-2: STA for pairs of adjacent spikes')
    plt.xlabel('Time/ms')
    plt.ylabel('Stimulus Average')
    plt.show()


Hz=1.0
s=1.0
ms=0.001

# Question 1-1 -> tau_ref 0

rate = 35.0 * Hz
tau_ref = 0*ms
T = 1000*s

spike_train = get_spike_train(rate, T, tau_ref)
fano_factor = cal_fano_factor(spike_train, [10, 50, 100], T)
coefficient_variation = calc_coef_var(spike_train)

print(fano_factor)
print(coefficient_variation)


# Question 1-2 -> tau_ref 5

rate = 35.0 * Hz
tau_ref = 5*ms
T = 1000*s

spike_train = get_spike_train(rate, T, tau_ref)
fano_factor = cal_fano_factor(spike_train, [10, 50, 100], T)
coefficient_variation = calc_coef_var(spike_train)

print(fano_factor)
print(coefficient_variation)

# Question 2

rate = 50.0 * Hz
tau_ref = 2*ms
T = 20*60*s

spikes = load_data("rho.dat", int)
spike_train = spike_train_time(spikes, tau_ref)
fano_factor = cal_fano_factor(spike_train, [10, 50, 100], T)
coefficient_variation = calc_coef_var(spike_train)

print(fano_factor)
print(coefficient_variation)


# Question 3
spikes = load_data("rho.dat", int)
interval = 100 * ms
tau_ref = 2 * ms


plt.plot(np.linspace(0, 100, acorr(spikes[:8000]).size), acorr(spikes[:8000]))
plt.title("Q3: Autocorrelogram of Spike Trains from H1 in Fruit Fly.")
plt.xlabel("Interval/ms")
plt.ylabel("No.of Spikes")
plt.xticks(np.linspace(0, 100, 5), [-100, -50, 0, 50, 100])
plt.show()

# Question 4

stim = load_data("stim.dat", float)
blocks = [stim[i-50:i] for i, x in enumerate(spikes) if x == 1 and i > 50]
avg = [x / len(blocks) for x in np.sum(np.array(blocks), 0)]

plt.plot(np.arange(-100, 0, 2), avg)
plt.title('Q4: Spike Triggered Average over 100ms')
plt.xlabel('Time/ms')
plt.ylabel('Stimulus Average')
plt.show()

# Question 5 (1 and 2)- COMSM2127

non_adj_stim_avg(spikes, stim, [int(x/2) for x in [2, 10, 20, 50]])
adj_stim_avg(spikes, stim, [int(x/2) for x in [2, 10, 20, 50]])
