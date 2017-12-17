#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# This is a test procedure. It shows the distribution of the elevation angle
# of selected events.

import numpy
import matplotlib.pyplot as plt

from retro.event import EventIterator
from grand_tour import Topography

topo = Topography(42.928056, 86.741667, "../../share/topography", 25)
elevation = []
for event in EventIterator("../../events-ulastai.json"):
    energy, position, direction = event["tau_at_decay"]
    theta, phi = topo.local_to_angular(position, direction)
    elevation.append(90. - theta)

p, x = numpy.histogram(elevation, 40, density=True)
x = 0.5 * (x[1:] + x[:-1])

plt.style.use("deps/mplstyle-l3/style/l3.mplstyle")
plt.figure()
plt.plot(x, p, "ko-")
plt.show()
