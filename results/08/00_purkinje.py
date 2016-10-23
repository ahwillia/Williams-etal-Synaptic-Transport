from neuron import h
from PyNeuronToolbox import neuromorpho,morphology
import pylab as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
fig = plt.figure()
xsc = np.array([0,100])
ysc = np.array([0,0])
view = (90,-90)

## PURKINJE CELL ##
neuromorpho.download('Purkinje-slice-ageP43-6', filename='purkinje_morph.swc')
purkinje = morphology.load('purkinje_morph.swc', use_axon=False)

plt.clf()
ax = fig.gca(projection='3d')
morphology.shapeplot(h, ax, sections=purkinje.dend, color='k')
morphology.mark_locations(h,purkinje.soma[0], 0.5, color='r', ms=10)
ax.plot(xsc-5,ysc-20,'-r',lw=2)
ax.view_init(*view)
ax.set_axis_off()
plt.savefig('purkinje_morph.png')

import fig8_sushi_belt

#anim = fig8_sushi_belt.run_sims(h,'purkinje',view)
A,u,t,err = fig8_sushi_belt.run_sims(h,'purkinje',view)
