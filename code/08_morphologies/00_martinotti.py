from neuron import h
from PyNeuronToolbox import neuromorpho,morphology
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
fig = pyplot.figure()
xsc = np.array([0,100])
ysc = np.array([0,0])
view = (-90, -90)

## LAYER 2-3 MARTINOTTI Cell ##
neuromorpho.download('C100501A3', filename='martinotti_morph.swc')
martinotti = morphology.load('martinotti_morph.swc',use_axon=True)

pyplot.clf()
ax = fig.gca(projection='3d')
morphology.shapeplot(h, ax, sections=martinotti.dend, color='k')
morphology.mark_locations(h,martinotti.soma[0],0.5, color='r', ms=10)
ax.plot(xsc+90,ysc+200,'-r',lw=2)
ax.view_init(*view)
ax.set_axis_off()
pyplot.savefig('martinotti_morph.png')

import fig8_sushi_belt

fig8_sushi_belt.run_sims(h,'martinotti',view)

A = fig8_sushi_belt.get_uniform_distribution_model(h, 1e-4, 4.0)

