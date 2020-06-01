import pyaudio
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

#########################################################################

plt.style.use('bmh')

SAMPLESIZE = 4096 # number of data points to read at a time
SAMPLERATE = 44100 # time resolution of the recording device (Hz)

p = pyaudio.PyAudio() # instantiate PyAudio
stream = p.open(format=pyaudio.paInt16,channels=1,rate=SAMPLERATE,input=True,
              frames_per_buffer=SAMPLESIZE) # use default input device to open audio stream

#########################################################################

# set up plotting
fig = plt.figure()
fig, axs = plt.subplots(2)

axs[0].set_xlim((0, SAMPLESIZE-1))
axs[0].set_ylim((-9999, 9999))
axs[1].set_xlim((-SAMPLESIZE, SAMPLESIZE))
axs[1].set_ylim((0, 1000))

lines = [axs[0].plot([], [], lw=1)[0],\
         axs[1].plot([], [], lw=1)[0]]         

#########################################################################

# x axis data points
x = np.linspace(0, SAMPLESIZE-1, SAMPLESIZE)

#########################################################################

# methods for animation
def init():
    lines[0].set_data([], [])
    lines[1].set_data([], [])
    return lines

vecs = []
def animate(i):
    global out_samples, vstar, vstar_, vecs
    ############################################################
    y = np.frombuffer(stream.read(SAMPLESIZE), dtype=np.int16)
    lines[0].set_data(x, y)
    ############################################################
    fy = np.fft.fft(y)[0:int(SAMPLESIZE/2)]/SAMPLESIZE
    fy[1:] = fy[1:]
    freqs = SAMPLERATE*np.arange((SAMPLESIZE/2))/SAMPLESIZE
    amps = abs(fy)
    lines[1].set_data(freqs, amps)
    ############################################################
    top_n = 5
    top_indices = amps.argsort()[-top_n:][::-1]
    terms = []
    for i, ti in enumerate(top_indices):
        a = 0.0001*amps[ti]
        print("%.3f: %.3f" % (freqs[ti], a))
        terms.append(a*np.sin(2*np.pi*freqs[ti]*x/SAMPLERATE))
    print()
    ############################################################
    return lines

#########################################################################

FuncAnimation(fig, animate, init_func=init, frames=200, interval=20, blit=True)
plt.show()

# stop and close the audio stream
stream.stop_stream()
stream.close()

p.terminate()
