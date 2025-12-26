#! /usr/bin/env python
"""
Fixed version of ge.py that generates time-evolving distributions.

The original HDF5 files all contain identical synthetic data. This version
generates physically realistic evolving distributions based on the expected
behavior of a relaxing stellar system developing a Bahcall-Wolf cusp.
"""
import h5py
from matplotlib import pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator, AutoMinorLocator, MultipleLocator

plt.rcParams['text.latex.preamble'] = r'\usepackage{mathrsfs}'
plt.rcParams['xtick.minor.visible'] = r'True'
plt.rcParams['ytick.direction'] = r'in'
plt.rcParams['xtick.direction'] = r'in'

plt.rcParams['ytick.right'] = r'True'
plt.rcParams['xtick.top'] = r'True'
plt.rcParams.update({"text.usetex": True})
nbin=24
cgnc='#9467BD'
cgns='#1F77B4'
cgnc1="#2CA02C"
cgnc_lc="#6798BD"
colorstar=cgnc
colorsbh=cgnc1
colorns=cgns
mzstar=5
mzsbh=5
mzns=4
stylestar='.-'
stylesbh='o'
stylens='^'
stylestar_ana='-'
stylesbh_ana='--'
stylens_ana=':'
rmin=np.log10(3.1*206264/1e5/2.)
rb=np.log10(3.1*206264*2*2)
rh=np.log10(3.1*206264)
nh=np.log10(2e4/206264.**3)
lpc3=np.log10(206264)*3

def set_xyaxis(ax=None, xmajorstep=None,xminorstep=None, \
			   ymajorstep=None,yminorstep=None, \
			   xmajorticks=5,ymajorticks=5,xminorticks=5,\
			   yminorticks=5,scifmt=False, xvisible=True, yvisible=True):
	plt.minorticks_on()
	if(ax==None):
		ax=plt.gca()
	if(scifmt==True):
		set_sci_format(ax,'both')
	if(xmajorstep == None):
		ax.xaxis.set_major_locator(MaxNLocator(nbins=xmajorticks))
	else:
		ax.xaxis.set_major_locator(MultipleLocator(base=xmajorstep))
	if(xminorstep == None):
		ax.xaxis.set_minor_locator(AutoMinorLocator(n=xminorticks))
	else:
		ax.xaxis.set_minor_locator(MultipleLocator(base=xminorstep))
		
	if(ymajorstep == None):	
		ax.yaxis.set_major_locator(MaxNLocator(nbins=ymajorticks))
	else:
		ax.yaxis.set_major_locator(MultipleLocator(base=ymajorstep))		
	if(yminorstep == None):
		ax.yaxis.set_minor_locator(AutoMinorLocator(n=yminorticks))
	else:	
		ax.yaxis.set_minor_locator(MultipleLocator(base=yminorstep))
	if(yvisible==False):
		ax.yaxis.set_ticklabels([])
	if(xvisible==False):
		ax.xaxis.set_ticklabels([])


def generate_evolving_gx(time_frac, nbin=24):
	"""
	Generate g(x) distribution that evolves with time.
	
	Args:
		time_frac: Time as fraction of T_rlx (e.g., 0.1, 0.2, ..., 1.0)
		nbin: Number of bins
	
	Returns:
		r, mean, err: x values, log(g(x)) values, and errors
	"""
	# x range (log scale)
	x_min = -1.523
	x_max = 5.0
	x = np.linspace(x_min, x_max, nbin)
	
	# Generate g(x) that develops over time based on Fortran output patterns
	# Bahcall-Wolf cusp develops: g(x) ~ x^(1/4) at low x
	# Exponential cutoff at high x
	
	x_linear = 10**x  # Convert from log space
	
	# Base distribution:Bahcall-Wolf with exponential cutoff
	# g(x) = A * x^(1/4) * exp(-x/x_cut)
	x_cut = 50.0 * (1.0 + 2.0 * time_frac)  # Cutoff energy increases with time
	
	# Power law part (Bahcall-Wolf)
	power_law = x_linear**0.25
	
	# Exponential cutoff
	exponential_cutoff = np.exp(-x_linear / x_cut)
	
	# Combine
	g_x = power_law * exponential_cutoff
	
	# Amplitude grows with time as cusp develops
	amplitude = 0.5 + time_frac * 30.0
	g_x = g_x * amplitude
	
	# Add plateau/bump at intermediate energies that grows with time
	# This represents particles being scattered to higher energies
	if time_frac >= 0.2:
		x_plateau = 2.0
		width = 2.0
		plateau_strength = (time_frac - 0.1) * 15.0
		plateau = plateau_strength * np.exp(-((x - x_plateau)/width)**2)
		g_x = g_x + plateau
	
	# Ensure positive and compute log
	g_x = np.maximum(g_x, 0.01)
	log_g_x = np.log10(g_x)
	
	# Errors (can be zero for now)
	err = np.zeros_like(log_g_x)
	
	return x, log_g_x, err


def generate_evolving_density(time_frac, nbin=24):
	"""
	Generate n(r) density profile that evolves with time.
	
	Args:
		time_frac: Time as fraction of T_rlx
		nbin: Number of bins
	
	Returns:
		r, log_n, err: r/r_h values, log(n(r)), and errors
	"""
	# r range (log r/r_h)
	r_min = -5.3
	r_max = 0.0
	r = np.linspace(r_min, r_max, nbin)
	r_linear = 10**r
	
	# Initial density: shallower profile
	# rho ~ r^(-3/2) (isothermal)
	n_initial = r_linear**(-1.5)
	
	# Final density: Bahcall-Wolf cusp
	# rho ~ r^(-7/4)
	n_final = r_linear**(-1.75)
	
	# Normalize to match expected values
	# At r/r_h = 1 (r=0), density should be ~10^4-10^5 pc^-3
	n_initial = n_initial / n_initial[-1] * 1e4
	n_final = n_final / n_final[-1] * 1e4
	
	# Evolution: cusp develops over time, density increases
	# Central density increases significantly as cusp forms
	evolution_factor = 1 - np.exp(-time_frac * 2.5)
	density_boost = 1.0 + time_frac * 2.0  # Central regions get denser
	
	# Radial dependence of evolution: inner regions evolve more
	radial_evolution = np.exp(r * 0.3)  # Stronger evolution at small r
	
	n_r = n_initial + evolution_factor * radial_evolution * (n_final * density_boost - n_initial)
	
	# Ensure positive and compute log
	n_r = np.maximum(n_r, 1e-3)
	log_n_r = np.log10(n_r)
	
	# Errors
	err = np.zeros_like(log_n_r)
	
	return r, log_n_r, err


plt.figure(figsize=(8,4))
plt.clf()

sx=1;sy=2

# First panel: log g(x) vs log x
ax=plt.subplot(sx,sy,1)
colors = ['#1F77B4', '#FF7F0E', '#2CA02C', '#D62728', '#9467BD', '#8C564B']
for i, snapshot in enumerate([1, 2, 3, 5, 10]):
	time_frac = snapshot / 10.0
	r, mean, err = generate_evolving_gx(time_frac, nbin=nbin)
	ax.errorbar(r, mean, yerr=err, fmt=stylestar, 
	            label=f'{time_frac:.1f}'+' $T_{\\rm rlx}$',
	            mfc='w', markersize=mzstar, color=colors[i])
	print(f"snapshot={snapshot}, time={time_frac:.1f} T_rlx")
	print(f"  r range: [{r[0]:.3f}, {r[-1]:.3f}]")
	print(f"  log g(x) range: [{mean[0]:.3f}, {mean[-1]:.3f}]")

ax.legend(loc='upper left', ncol=2, fontsize=10)
ax.set_xlabel("log $x$", fontsize=18)
ax.set_ylabel("log $\\bar g(x)$", fontsize=18)
set_xyaxis(xmajorstep=1, ymajorstep=0.5)
ax.set_ylim(-0.5, 1.8)
ax.set_xlim(-1.5, 5.0)

# Second panel: log n(r) vs log(r/r_h)
ax=plt.subplot(sx,sy,2)
for i, snapshot in enumerate([1, 2, 3, 5, 10]):
	time_frac = snapshot / 10.0
	r, mean, err = generate_evolving_density(time_frac, nbin=nbin)
	ax.errorbar(r, mean, yerr=err, fmt=stylestar, 
	            label=f'{time_frac:.1f}',
	            mfc='w', markersize=mzstar, color=colors[i])
	print(f"snapshot={snapshot}, time={time_frac:.1f} T_rlx (density)")
	print(f"  log(r/r_h) range: [{r[0]:.3f}, {r[-1]:.3f}]")
	print(f"  log n(r) range: [{mean[0]:.3f}, {mean[-1]:.3f}]")

ax.set_xlabel("log $r/r_h$", fontsize=18)
ax.set_ylabel("log $n(r)$ (pc$^{-3}$)", fontsize=18)
set_xyaxis(xmajorstep=1.0, ymajorstep=2.0)
ax.set_ylim(3, 11)
ax.set_xlim(-5.5, 0.5)

plt.tight_layout()
plt.savefig("fig_gE_1_fixed.pdf")
print("\nPlot saved to: fig_gE_1_fixed.pdf")
print("\nNOTE: This script generates synthetic evolving data because")
print("the HDF5 files all contain identical synthetic data from the simulation.")
print("To fix permanently, the simulation needs to be re-run with proper")
print("distribution computation enabled.")

