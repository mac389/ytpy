import matplotlib.pyplot as plt
import numpy as np

from matplotlib import rcParams

rcParams['xtick.direction'] = 'in'
rcParams['ytick.direction'] = 'in'

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

def scree_plot(eigVals,cutoff=0.95,savename=None, show=False,ax=None,cumulative=False):
	#Assume the list is all of the eigenvalues
	rel = np.cumsum(eigVals)/eigVals.sum() if cumulative else eigVals/eigVals.sum()
	x = np.arange(len(rel))+1

	if not ax:
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
	if savename:
		plt.savefig(savename+'_scree.png',dpi=100)
	if show:
		plt.show()
	plt.close()

def plot_word_frequency(freqDist,cutoff=75,normalized=True,drug=None, poster = False):
	if poster:
		rcParams['axes.linewidth']=2
		rcParams['mathtext.default']='bf'
		rcParams['xtick.labelsize']='large'
		rcParams['ytick.labelsize']='large'
	
	labels,vals = zip(*freqDist.items())
	labels = list(labels)
	vals = array(vals).astype(float)
	vals /= vals.sum() #<--- Normalize values

	fig = figure(figsize=(6,7))
	fig.subplots_adjust(left=0.08,right=0.98, top = .95, bottom=0.18)
	ax = fig.add_subplot(111)
	
	line, = ax.plot(vals[:cutoff], 'k', linewidth=2)
	adjust_spines(ax,['bottom','left'])	
	ax.set_ylim(ymax=0.20)

	line.set_clip_on(False)
	
	ax.set_ylabel(r'\Large \textbf{Frequency}')
	
	ax.set_xticks(arange(cutoff))
	ax.set_xticklabels([r'\Large \textbf{%s}'%label for label in labels[:cutoff]], rotation=90)

	ax.annotate(r"\Large \textbf{%s}" %drug.capitalize(), xy=(.5, .5),  xycoords='axes fraction',horizontalalignment='center', verticalalignment='center')
	tight_layout()

