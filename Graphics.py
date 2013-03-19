import matplotlib.pyplot as plt
import numpy as np

from matplotlib import rcParams
from matplotlib.mlab import psd

rcParams['xtick.direction'] = 'in'
rcParams['ytick.direction'] = 'in'

from spectrum import *

def adjust_spines(ax,spines):
	''' Taken from http://matplotlib.org/examples/pylab_examples/spine_placement_demo.html '''
	for loc, spine in ax.spines.iteritems():
		if loc in spines:
			spine.set_position(('outward',10))
			spine.set_smart_bounds(True) #Doesn't work for log log plots
			spine.set_linewidth(3)
		else:
			spine.set_color('none') 
	if 'left' in spines:
		ax.yaxis.set_ticks_position('left')
	else:
		ax.yaxis.set_ticks([])

	if 'bottom' in spines:
		ax.xaxis.set_ticks_position('bottom')
	else:
		ax.xaxis.set_ticks([])

def power_spectrum(data,Fs=20000, save=False,show=True, cutoff=120):
	p = Periodogram(data,sampling=20000)
	p.run()
	p.plot()
	'''
	#stop = np.where(freqs>cutoff)[0][0]
	#print stop
	fig = plt.figure()
	ax = fig.add_subplot(111)
	spec, = ax.plot(freqs,db,'o-')
	adjust_spines(ax,['bottom','left'])
	ax.set_xlabel(r'frequency $\left(Hz\right)$')
	ax.set_ylabel(r'Power $\left(dB\right)$')
	'''
	if show:
		plt.show()
	if save:
		plt.savefig('example_ps.png',dpi=72)

def eig_spectrum(eigVals,cutoff=0.95,savename=None,show=False,save=True,savebase=None):
	#Assume the list is all of the eigenvalues
	rel = eigVals/eigVals.sum()
	x = np.arange(len(rel))+1
	fig = plt.figure()
	ax = fig.add_subplot(111)
	line, = ax.plot(x,rel)
	line.set_clip_on(False)
	adjust_spines(ax,['bottom','left'])
	ax.set_xlabel(r'$\LARGE \lambda$')
	ax.set_ylabel('Fraction of variance')
	ax.set_xlim(0,len(eigVals))
	
	cutoff_idx = np.where(np.cumsum(rel)>cutoff)[0][0]
	
	ax.axvline(x=cutoff_idx, color='r',linestyle='--', linewidth=2)
	ax.axhline(y=rel[cutoff_idx],color='r',linestyle='--',linewidth=2)
	ax.tick_params(direction='in')
	ax.annotate(r" {\Large $\mathbf{\lambda=%d}$}" % cutoff_idx,xy=(.25, .9), xycoords='axes fraction', 
											horizontalalignment='center', verticalalignment='center')
	plt.tight_layout()
	if save:
		print savebase
		plt.savefig(savebase+'_scree.png',dpi=100)
			
	if show:
		plt.show()
	plt.close()

def scree_plot(eigVals,cutoff=0.95,savename=None, show=False,save=True,savebase=None):
	#Assume the list is all of the eigenvalues
	rel = np.cumsum(eigVals)/eigVals.sum()
	x = np.arange(len(rel))+1
	print eigVals.shape
	fig = plt.figure()
	ax = fig.add_subplot(111)
	line, = ax.plot(x,rel)
	line.set_clip_on(False)
	adjust_spines(ax,['bottom','left'])
	ax.set_xlabel(r'$\LARGE \lambda$')
	ax.set_ylabel('Fraction of variance')
	ax.set_xlim(0,len(eigVals))
	
	cutoff_idx = np.where(rel>cutoff)[0][0]
	
	ax.axvline(x=cutoff_idx, color='r',linestyle='--', linewidth=2)
	ax.axhline(y=rel[cutoff_idx],color='r',linestyle='--',linewidth=2)
	ax.tick_params(direction='in')
	ax.annotate(r" {\Large $\mathbf{\lambda=%d}$}" % cutoff_idx,xy=(.25, .9), xycoords='axes fraction', 
											horizontalalignment='center', verticalalignment='center')
	plt.tight_layout()
	if save:
		print savebase
		plt.savefig(savebase+'_scree.png',dpi=100)
			
	if show:
		plt.show()
	plt.close()

def spike_validation(data,clusters,spiketimes=None,eiglist=None,nclus=None,savebase='res',waveforms=None,multi=False, show=False
					,save=True):
	best = clusters['models'][np.argmax(clusters['silhouettes'])]
	nclus = best.n_clusters if not nclus else nclus
	fig = plt.figure()
	plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=.97)
	#Clusters of waveforms projected onto the first two principal components
	ax = fig.add_subplot(2,2,1)
	ax.set_axis_bgcolor('white')
	colors = ['#4EACC5', '#FF9C34', '#4E9A06']
	labels_ = best.labels_
	centers = best.cluster_centers_
	unique_labels = np.unique(labels_)
	for n,col in zip(range(nclus),colors):
		my_members = labels_ == n 
		cluster_center = centers[n]
		ax.plot(data[my_members,0],data[my_members,1],'w',markerfacecolor=col,marker='.', markersize=6)
		plt.hold(True)
		ax.plot(cluster_center[0],cluster_center[1],'o',markerfacecolor=col,markeredgecolor='k',markersize=8)
	adjust_spines(ax,['bottom','left'])
	ax.set_ylabel('PC2')
	ax.set_xlabel('PC1')
	ax.tick_params(direction='in')
	
	if waveforms.size:
		wfs = fig.add_axes([0.17, 0.65, 0.1, 0.15])
		wfs.set_axis_bgcolor('none')
		artists = []
		for n,col in zip(range(nclus),colors):
			my_members = labels_[:-3000] == n
			line, = wfs.plot(np.average(waveforms[my_members,:],axis=0),col,linewidth=2)
			line.set_clip_on(False)	
		adjust_spines(wfs,['bottom','left'])
		
		wfs.spines['bottom'].set_position(('outward',-2)) 
		wfs.spines['left'].set_position(('outward',-2)) 
		
		wfs.set_yticks([0,100])
		wfs.set_yticklabels([r'$0$', r'$100 \; \mu V$'],rotation='vertical')
		
		wfs.set_xticks([0,100])
		wfs.set_xticklabels([r'$0$',r'$100 \; \mu s$'])
		
		wfs.spines['left'].set_bounds(0,100)
		wfs.spines['bottom'].set_bounds(0,100)
		
		wfs.tick_params(direction='in')
	sils = fig.add_subplot(2,2,2)
	sils.set_axis_bgcolor('none')
	markerline, stemlines,baseline =sils.stem(np.arange(len(clusters['silhouettes'])),clusters['silhouettes'])
	sils.tick_params(direction='in')
	sils.axhline(y=0.5,color='r',linestyle='--',linewidth=2)
	adjust_spines(sils,['bottom','left'])
	sils.set_xticks(np.arange(len(clusters['silhouettes']))+1)
	sils.set_yticks([-1,1])
	sils.set_ylabel('Silhouette coefficient')
	sils.set_xlabel('Number of clusters')
	sils.set_xlim((0.5,len(clusters['silhouettes'])))
	
	xmx=1300
	if spiketimes and spiketimes.size>1000:
		#break of up the spiketime vector based on clustering
		for n,col in zip(range(nclus),colors):
			my_members = labels_[:-3000]== n #Always add 3000 noise spikes
			these_isis = diff(spiketimes[my_members])
			if these_isis.size:
				isi = fig.add_subplot(2,2,4)
				_,_,patches=isi.hist(these_isis, histtype='stepfilled', range=(0,xmx),alpha=0.5,normed=True)
				adjust_spines(isi,['bottom','left'])
				plt.setp(patches,'facecolor',col)
		isi.tick_params(direction='in')
		isi.set_axis_bgcolor('none')
		isi.set_ylabel(r'Fraction of total spikes')
		isi.set_xlabel(r'Interspike interval $(ms)$')
		isi.set_xlim(xmax=xmx)
		
		short_isi = fig.add_axes([0.77, 0.26, 0.15, 0.20])
		short_isi.set_axis_bgcolor('none')
		_,_,spatches=short_isi.hist(np.diff(spiketimes[labels_[:-3000]!=1]),
									bins=200,range=(10,20), histtype='stepfilled', normed = True)
		adjust_spines(short_isi,['bottom','left'])
		short_isi.tick_params(direction='in')
		short_isi.set_ylabel(r'\# of Spikes')
		short_isi.set_yticks(arange(8))
		short_isi.set_xlabel(r'ISI $(ms)$')
		short_isi.set_xticklabels(np.arange(0,10)[::2])
		plt.setp(spatches,'facecolor',colors[1])
		
	if eiglist.size:
		eigfxns = fig.add_subplot(2,2,3)
		eigfxns.set_axis_bgcolor('none')
		eigfxns.tick_params(direction='in')
		#Assume 6 eigenfunctions
		nfxns =6
		span = len(eiglist[0,:])/2
		print span
		x = arange(2*span) if multi else np.arange(-span,span)
		for i in range(nfxns):
			eigfxns.plot(x,i+eiglist[i,:],'b',linewidth=2)
			plt.hold(True)
		adjust_spines(eigfxns,['bottom','left'])
		if multi:
			eigfxns.set_xlabel(r' $\left(\mu sec\right)$')
		else:
			eigfxns.set_xlabel(r'Time from spike peak $\left(\mu sec\right)$')
			eigfxns.set_yticklabels([' '] + [r' $e_{%d}$' %i for i in range(1,nfxns+1) ])
		eigfxns.set_ylabel(r'Eigenfunctions')
		#draw_sizebar(eigfxns)

	plt.tight_layout()
	plt.savefig(savebase+'_validation.png', bbox_inches='tight')
	if show:
		plt.show()

def voltage_trace(unfiltered=None,filtered=None,threshold = 0, roi=30000,spread=10000,save=None, 
					show=False, fs = 20000, downsampling= 10,savebase=None):
					
	fig = plt.figure()
	trace_panel = fig.add_subplot(211,axisbg='none')
	start = roi-spread
	stop = roi+spread

	traces, = trace_panel.plot(unfiltered[start:stop][::downsampling],'b') #Downsample just for display
		
	spike_panel = fig.add_subplot(212,axisbg='none',sharex=trace_panel)
	spikes, = spike_panel.plot(filtered[start:stop][::downsampling],'b')
		
	panels = [trace_panel,spike_panel]
	
	for panel in panels:
		adjust_spines(panel,['bottom','left'])
	
	trace_panel.set_xlabel(r'time $\left(s\right)$')
	trace_panel.set_ylabel(r'voltage $ \left(\mu V \right)$')
	trace_panel.set_xticklabels(np.arange(start/fs,1.5+stop/fs,0.5).astype(str))

	spike_panel.set_xlabel(r'time $\left(s\right)$')
	spike_panel.set_xticklabels(np.arange(start/fs,1.5+stop/fs,0.5).astype(str))
	spike_panel.set_ylabel(r'voltage $\left(\mu V \right)$')
	#Draw threshold
	spike_panel.axhline(y=threshold,linewidth=1,color='r',linestyle='--')
	spike_panel.axhline(y=-threshold,linewidth=1,color='r',linestyle='--')
	
	plt.tight_layout()
	if save:
		print savebase
		plt.savefig(savebase+'_voltage.png',dpi=100)
			
	if show:
		plt.show()