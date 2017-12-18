#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# This is a test procedure. It plots the density of decaying taus as
# function of the tau energy. Use the `run-primary.py` script in order to
# generate the data.

import json
import numpy
import matplotlib.pyplot as plt

from retro import TAU_CTAU, TAU_MASS


# Load RETRO data
with open("share/wbb-density.json", "rb") as f:
    data = json.load(f)
    energy, density = map(numpy.array, (data["energy"], data["density"]))


# Load and process the NuTauSim data
nts = numpy.loadtxt("share/nutausim.dat", skiprows=1, comments="END")
n = nts[-1,3]
Emin, Emax = 1E+06, 1E+12
e = 10**(nts[:,4:] - 9.)
e1, e0 = e[:, 0], e[:, 1]
WBB = 2. / 3. * 1E-04
w = (WBB * TAU_MASS / TAU_CTAU) * numpy.log(Emax / Emin) / (e0 * e1)

b = numpy.log(energy)
dle = b[1] - b[0]
b = numpy.hstack((b - 0.5 * dle, (b[-1] + 0.5 * dle,)))

d_nts, e_nts = numpy.histogram(numpy.log(e1), b, weights = w / n)
d2_nts, _ = numpy.histogram(numpy.log(e1), b, weights = w**2 / n)
d2_nts = numpy.sqrt((d2_nts - d_nts**2) / n)
d_nts /= energy * dle
d2_nts /= energy * dle


# Plot the comparison
plt.style.use("deps/mplstyle-l3/style/l3.mplstyle")
plt.figure()
K = density[:, 0] > 0.
p = numpy.polyfit(numpy.log(energy[K]), numpy.log(density[K, 0]), 3)
plt.loglog(energy, numpy.exp(numpy.polyval(p, numpy.log(energy))), "r-")
plt.errorbar(energy, d_nts, yerr=numpy.sqrt(d2_nts**2 + density[:, 1]**2),
             fmt="ko")
plt.xlabel(r"$\tau$ energy (GeV)")
plt.ylabel(r"density (GeV$^{-1}$ m$^{-3}$ sr$^{-1}$ s$^{-1}$)")
plt.savefig("retro-primary-comparison.png")
plt.show()
