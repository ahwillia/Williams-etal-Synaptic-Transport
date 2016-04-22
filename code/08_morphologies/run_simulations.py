%reset
from neuron import h
from PyNeuronToolbox import neuromorpho,morphology
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
fig = pyplot.figure()
xsc = np.array([0,100])
ysc = np.array([0,0])
%macro reload_neuron 1-9

## PURKINJE CELL ##
neuromorpho.download('Purkinje-slice-ageP43-6', filename='purkinje_morph.swc')
purkinje = morphology.load('purkinje_morph.swc',use_axon=False)

pyplot.clf()
ax = fig.gca(projection='3d')
morphology.shapeplot(h, ax, sections=purkinje.dend, color='k')
ax.plot(xsc-5,ysc-20,'-r',lw=2)
ax.view_init(90, -90)
ax.set_axis_off()
pyplot.savefig('purkinje_morph.png')

## LAYER 5 PYRAMIDAL ##
%reset
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
ax.plot(xsc+120,ysc-140,'-r',lw=2)
ax.view_init(90, -90)
ax.set_axis_off()
pyplot.savefig('l5_morph.png')
l5cell.delete()

# ## LAYER 2-3 MARTINOTTI Cell ##
# neuromorpho.download('C100501A3', filename='martinotti_morph.swc')
# martinotti = morphology.load_swc('martinotti_morph.swc',use_axon=False)

# pyplot.clf()
# ax = fig.gca(projection='3d')
# morphology.shapeplot(h, ax, sections=martinotti.dend, color='k')
# ax.plot(xsc+90,ysc+200,'-r',lw=2)
# ax.view_init(-90, -90)
# ax.set_axis_off()
# pyplot.savefig('martinotti_morph.png')
# martinotti.delete()

# ## PV Cell ##
# neuromorpho.download('AWa80213', filename='pv_morph.swc')
# pvcell = morphology.load_swc('pv_morph.swc',use_axon=False)

# pyplot.clf()
# ax = fig.gca(projection='3d')
# morphology.shapeplot(h, ax, sections=pvcell.dend, color='k')
# ax.plot(xsc+90,ysc-120,'-r',lw=2)
# ax.view_init(90, -90)
# ax.set_axis_off()
# pyplot.savefig('pv_morph.png')
# #pvcell.delete()

# pyplot.close('all')
