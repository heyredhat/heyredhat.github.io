import numpy as np
import strawberryfields as sf
from strawberryfields.ops import *

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons

prog = sf.Program(1)
with prog.context as q:
    Vac | q[0]
eng = sf.Engine('gaussian')
state = eng.run(prog).state

Q, P = np.linspace(-5, 5, 100), np.linspace(-5, 5, 100)
Qvals = state.x_quad_values(0, Q, P)
Pvals = state.p_quad_values(0, Q, P)

fig = plt.figure(figsize=(8,6))

q_axis = fig.add_subplot(121)
q_axis.set_xlabel('q', fontsize=14)
q_axis.set_ylabel('Pr(q)', fontsize=14)
q_plot, = q_axis.plot(Q, Qvals)

p_axis = fig.add_subplot(122)
p_axis.set_xlabel('p', fontsize=14)
p_axis.set_ylabel('Pr(p)', fontsize=14)
p_plot, = p_axis.plot(P, Pvals)

fig.tight_layout()
fig.subplots_adjust(bottom=0.33) 

qdisp_axis = plt.axes([0.25, 0.0, 0.65, 0.03], facecolor="lightgoldenrodyellow")
qdisp_slider = Slider(qdisp_axis, 'Q Displacement (real)', -10, 10, valinit=0)

pdisp_axis = plt.axes([0.25, 0.05, 0.65, 0.03], facecolor="lightgoldenrodyellow")
pdisp_slider = Slider(pdisp_axis, 'P Displacement (imag)', -10, 10, valinit=0)

rsqueeze_axis = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor="lightgoldenrodyellow")
rsqueeze_slider = Slider(rsqueeze_axis, 'Squeeze (r)', -10, 10, valinit=0)

thsqueeze_axis = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor="lightgoldenrodyellow")
thsqueeze_slider = Slider(thsqueeze_axis, 'Squeeze (theta)', 0, 2*np.pi, valinit=0)

rot_axis = plt.axes([0.25, 0.2, 0.65, 0.03], facecolor="lightgoldenrodyellow")
rot_slider = Slider(rot_axis, 'Rotation (theta)', 0, 2*np.pi, valinit=0)

def update(val):
    global eng, Q, P
    global fig, q_plot, p_plot
    global qdisp_slider, pdisp_slider, rsqueeze_slider, thsqueeze_slider, rot_slider
    eng.reset()
    prog = sf.Program(1)
    with prog.context as q:
        Sgate(rsqueeze_slider.val, thsqueeze_slider.val) | q[0]
        dz = qdisp_slider.val + 1j*pdisp_slider.val
        Dgate(np.abs(dz), np.angle(dz)) | q[0]
        Rgate(rot_slider.val) | q[0]
    state = eng.run(prog).state

    Qvals = state.x_quad_values(0, Q, P)
    Pvals = state.p_quad_values(0, Q, P)

    q_plot.set_ydata(Qvals)
    p_plot.set_ydata(Pvals)
    fig.canvas.draw_idle()

qdisp_slider.on_changed(update)
pdisp_slider.on_changed(update)
rsqueeze_slider.on_changed(update)
thsqueeze_slider.on_changed(update)
rot_slider.on_changed(update)

plt.show()