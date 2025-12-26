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


prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

nbin=24
cgnc='#9467BD'
cgns='#1F77B4'
#2CA02C
cgnc1="#2CA02C"
cana1='#FF7F0E'
cgnc_lc="#6798BD"
colorstar=cgnc
colorsbh=cgnc1
colorns=cgns
colorbd="#C4A000"
colorwd=colors[5]
mzstar=3
mzsbh=3
#mzbbh=5
mzns=3
stylestar='*-'
stylesbh='o-'
#stylebbh='-o'
stylens='^-'
stylewd='s-'
stylebd="d-"
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


def get_scatters_main(fdir, sm,sn,sb,cr=False):
	nums=1; nume=5; ncol=nbin
	x=np.zeros([nume-nums, ncol])
	y=np.zeros([nume-nums, ncol])
	for j in range(nums, nume):
		f=h5py.File(fdir+str(sm)+"_"+str(sn-j)+".hdf5", 'r' )
		#print(sn,j-nums)
		x[j-nums,:]=f[sb]['   X']
		y[j-nums,:]=f[sb]['  FX']
	if(cr):
		mean=np.mean(np.log10(y)+lpc3,axis=0)
		err=np.var(np.log10(y)+lpc3,axis=0)**0.5
		r=x[0,:]-rh
	else:
		mean=np.mean(np.log10(y),axis=0)
		err=np.var(np.log10(y),axis=0)**0.5
		r=x[0,:]
	return r,mean,err

plt.figure(figsize=(8,4))
plt.clf()

sx=1;sy=2

snap=7

ax=plt.subplot(sx,sy,1)

r, mean, err=get_scatters_main("../output/ecev/dms/dms_", snap,10,  '1/star/fgx/')	
ax.errorbar(r, mean, yerr=err,fmt=stylestar,color=colorstar,label='Stars (final)',mfc='w', markersize=mzstar)

r, mean, err=get_scatters_main("../output/ecev/dms/dms_", snap,10,  '2/bd/fgx/')	
ax.errorbar(r, mean, yerr=err,fmt=stylebd,color=colorbd,label='Brown dwarf (final)',mfc='w', markersize=mzstar)

r, mean, err=get_scatters_main("../output/ecev/dms/dms_", snap,10,  '3/wd/fgx/')	
ax.errorbar(r, mean, yerr=err,fmt=stylewd,color=colorwd,label='White dwarf (final)',mfc='w', markersize=mzstar)

r, mean, err=get_scatters_main("../output/ecev/dms/dms_", snap,10,  '4/ns/fgx/')	
ax.errorbar(r, mean, yerr=err,fmt=stylens,color=colorns,label='Neutron stars (final)',mfc='w', markersize=mzstar)

r, mean, err=get_scatters_main("../output/ecev/dms/dms_", snap,10,  '5/sbh/fgx/')	
ax.errorbar(r, mean, yerr=err,fmt=stylesbh,color=colorsbh,label='Black holes (final)',mfc='w', markersize=mzstar)

#ax.legend(loc='lower left',ncol=2)

#ax.legend(loc="lower right", ncol=1)
ax.set_xlim(np.log10(0.05),5.2)
ax.set_ylim(-4,0.5)
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
#for i in range(1,10+1,2):
#	r, mean, err=get_scatters_main("../output/ecev/dms/dms_", i,20,  '1/star/fden/',cr=True)	
#	ax.errorbar(r, mean, yerr=err,fmt=stylestar,label=str(i),mfc='w', markersize=mzstar)	

r, mean, err=get_scatters_main("../output/ecev/dms/dms_", snap,10,  '1/star/fden/',cr=True)	
ax.errorbar(r, mean, yerr=err,fmt=stylestar,color=colorstar,label='Stars (final)',mfc='w', markersize=mzstar)

r, mean, err=get_scatters_main("../output/ecev/dms/dms_", snap,10,  '2/bd/fden/',cr=True)	
ax.errorbar(r, mean, yerr=err,fmt=stylebd,color=colorbd,label='Brown dwarf (final)',mfc='w', markersize=mzstar)

r, mean, err=get_scatters_main("../output/ecev/dms/dms_", snap,10,  '3/wd/fden/',cr=True)	
ax.errorbar(r, mean, yerr=err,fmt=stylewd,color=colorwd,label='White dwarf (final)',mfc='w', markersize=mzstar)

r, mean, err=get_scatters_main("../output/ecev/dms/dms_", snap,10,  '4/ns/fden/',cr=True)	
ax.errorbar(r, mean, yerr=err,fmt=stylens,color=colorns,label='Neutron stars (final)',mfc='w', markersize=mzstar)

r, mean, err=get_scatters_main("../output/ecev/dms/dms_", snap,10,  '5/sbh/fden/',cr=True)	
ax.errorbar(r, mean, yerr=err,fmt=stylesbh,color=colorsbh,label='Black holes (final)',mfc='w', markersize=mzstar)

ax.legend(loc='upper right',ncol=1,fontsize=10)
ax.set_xlim(rmin-rh,0)
ax.set_ylim(3,13.5)
#ax.set_yscale("log")
ax.set_xlabel("log $r/r_h$",fontsize=18)
ax.set_ylabel("log $n(r)$ (pc$^{-3})$",fontsize=18)
set_xyaxis(xmajorstep=1.0,ymajorstep=2.0)

plt.tight_layout()
plt.savefig("fig_gE_5_lc.pdf")
