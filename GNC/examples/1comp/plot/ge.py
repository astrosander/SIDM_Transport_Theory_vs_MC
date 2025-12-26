#! /usr/bin/env python
import h5py
from matplotlib import pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator, AutoMinorLocator,MultipleLocator

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
#mzbbh=5
mzns=4
stylestar='.-'
stylesbh='o'
#stylebbh='-o'
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
		

def get_scatters_main(fdir, snapshot, update_idx, sb, cr=False):
	"""
	Load data from HDF5 file.
	
	Args:
		fdir: Directory prefix (e.g., "../output/ecev/dms/dms_")
		snapshot: Snapshot number (e.g., 10)
		update_idx: Update index within snapshot (e.g., 1, 3, 5, 7, 10)
		sb: HDF5 group path (e.g., '1/star/fgx/')
		cr: If True, compute density; if False, compute g(x)
	"""
	import os
	ncol=nbin
	
	# Direct file naming: dms_{snapshot}_{update}.hdf5
	fname = fdir + str(snapshot) + "_" + str(update_idx) + ".hdf5"
	
	if not os.path.exists(fname):
		print(f"Warning: File not found: {fname}")
		return np.zeros(ncol), np.zeros(ncol), np.zeros(ncol)
	
	try:
		f=h5py.File(fname, 'r')
		if sb in f:
			x = np.array(f[sb]['   X'])
			y = np.array(f[sb]['  FX'])
		else:
			print(f"Warning: Group '{sb}' not found in {fname}")
			print(f"  Available groups: {list(f.keys())}")
			# Try to find the data in a different location
			f.visititems(lambda n, o: print(f"    {n}") if isinstance(o, h5py.Group) else None)
			f.close()
			return np.zeros(ncol), np.zeros(ncol), np.zeros(ncol)
		f.close()
	except Exception as e:
		print(f"Error reading {fname}: {e}")
		return np.zeros(ncol), np.zeros(ncol), np.zeros(ncol)
	
	# Handle zeros in y to avoid log10 errors
	y = np.where(y > 0, y, 1e-10)
	
	if(cr):
		mean=np.log10(y)+lpc3
		err=np.zeros_like(mean)
		r=x-rh
	else:
		mean=np.log10(y)
		err=np.zeros_like(mean)
		r=x
	return r, mean, err


plt.figure(figsize=(8,4))
plt.clf()

sx=1;sy=2

ax=plt.subplot(sx,sy,1)
# Use different snapshots with update 10 (the final update of each snapshot)
# Snapshot i represents time i/10 T_rlx
for snapshot in [1, 2, 3, 5, 10]:
	r, mean, err=get_scatters_main("../output/ecev/dms/dms_", snapshot, 10,  '1/star/fgx/')	
	ax.errorbar(r, mean, yerr=err,fmt=stylestar,label=f'{snapshot/10.:.1f}'+' $T_{\\rm rlx}$',mfc='w', markersize=mzstar)
	print("snapshot=", snapshot, "r=", r, "mean=", mean, "err=", err)
ax.legend(loc='upper left',ncol=2)
# print("snapshot=", snapshot, "r=", r, "mean=", mean, "err=", err)
#ax.legend(loc="lower right", ncol=1)
# ax.set_xlim(np.log10(0.05),5.2)
# ax.set_ylim(-0.2,2)
#ax.set_yscale("log")
ax.set_xlabel("log $x$",fontsize=18)
ax.set_ylabel("log $\\bar g(x)$",fontsize=18)
set_xyaxis(xmajorstep=1,ymajorstep=1)

#######################################################################################################################
ax=plt.subplot(sx,sy,2)
#plot_scatters(ax, "../../../version_1.7_1D/model/nolc_1comp/output/ecev/dms/dms_burn_in_", 50,"1/star/fden",\
#	stylestar,cana1,mz=mzstar,mfc='w')
#plot_scatters(ax, "../../../version_1.7_1D/model/nolc_1comp/output/ecev/dms/dms_burn_in_", 50,"1/star/fden",\
#	"o-",cgnc1,mz=mzstar)	
# Use different snapshots with update 10 for density
for snapshot in [1, 2, 3, 5, 10]:
	r, mean, err=get_scatters_main("../output/ecev/dms/dms_", snapshot, 10,  '1/star/fden/',cr=True)	
	ax.errorbar(r, mean, yerr=err,fmt=stylestar,label=f'{snapshot/10.:.1f}',mfc='w', markersize=mzstar)
	print("snapshot=", snapshot, "r=", r, "mean=", mean, "err=", err)

#ax.set_xscale("log")
# ax.set_xlim(rmin-rh,0)
# ax.set_ylim(3,13.5)
#ax.set_yscale("log")
ax.set_xlabel("log $r/r_h$",fontsize=18)
ax.set_ylabel("log $n(r)$ (pc$^{-3})$",fontsize=18)
set_xyaxis(xmajorstep=1.0,ymajorstep=2.0)

plt.tight_layout()
plt.savefig("fig_gE_1.pdf")
