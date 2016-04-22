from neuron import h
from PyNeuronToolbox import neuromorpho,morphology
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
fig = pyplot.figure()
xsc = np.array([0,100])
ysc = np.array([0,0])
view = (90, -90)

## PV Cell ##
neuromorpho.download('AWa80213', filename='pv_morph.swc')
pvcell = morphology.load('pv_morph.swc',use_axon=False)

pyplot.clf()
ax = fig.gca(projection='3d')
morphology.shapeplot(h, ax, sections=pvcell.dend, color='k')
morphology.mark_locations(h,pvcell.soma[0],0.5, color='r', ms=10)
ax.plot(xsc+90,ysc-120,'-r',lw=2)
ax.view_init(*view)
ax.set_axis_off()
pyplot.savefig('pv_morph.png')

import fig8_sushi_belt

fig8_sushi_belt.run_sims(h,'pv_cell',view)
