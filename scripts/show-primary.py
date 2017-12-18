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

# Load and process NuTauSim data
nts = numpy.loadtxt("share/nutausim.dat", skiprows=1, comments="END")
n = nts[-1,3]
Emin, Emax = 1E+06, 1E+12
e = 10**(nts[:,4:] - 9.)
e1, e0 = e[:, 0], e[:, 1]
WBB = 2. / 3. * 1E-04
w = (WBB * TAU_MASS / TAU_CTAU) * numpy.log(Emax / Emin) / (e0 * e1)
d_nts, e_nts = numpy.histogram(numpy.log(e1), 40, weights = w / n)
dle = e_nts[1] - e_nts[0]
e_nts = numpy.exp(0.5 * (e_nts[1:] + e_nts[:-1]))
d_nts /= e_nts * dle

plt.style.use("deps/mplstyle-l3/style/l3.mplstyle")
plt.figure()
plt.loglog(e_nts, d_nts, "r-")
plt.errorbar(energy, density[:, 0], yerr=density[:, 1],
             fmt="k.")
plt.xlabel(r"$\tau$ energy (GeV)")
plt.ylabel(r"density (GeV$^{-1}$ m$^{-3}$ sr$^{-1}$ s$^{-1}$)")
plt.show()
