## LAYER 5 PYRAMIDAL ##
from neuron import h
from PyNeuronToolbox import neuromorpho,morphology
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
fig = pyplot.figure()
xsc = np.array([0,100])
ysc = np.array([0,0])

neuromorpho.download('32-L5pyr-28', filename='l5_morph.swc')
l5cell = morphology.load('l5_morph.swc',use_axon=False)

pyplot.clf()
ax = fig.gca(projection='3d')
morphology.shapeplot(h, ax, sections=l5cell.dend+l5cell.apic, color='k')
morphology.mark_locations(h,l5cell.soma[0],0.5, color='r', ms=10)
ax.plot(xsc+120,ysc-140,'-r',lw=2)
ax.view_init(90, -90)
ax.set_axis_off()
pyplot.savefig('l5_morph.png')
