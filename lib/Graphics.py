import matplotlib.pyplot as plt
import numpy as np

from matplotlib import rcParams

rcParams['xtick.direction'] = 'in'
rcParams['ytick.direction'] = 'in'
rcParams['text.usetex'] = True
def adjust_spines(ax,spines):
	''' Taken from http://matplotlib.org/examples/pylab_examples/spine_placement_demo.html '''
	for loc, spine in ax.spines.items():
		if loc in spines:
			spine.set_position(('outward',10)) # outward by 10 points
			spine.set_smart_bounds(True)
		else:
			spine.set_color('none') # don't draw spine
	# turn off ticks where there is no spine
	if 'left' in spines:
		ax.yaxis.set_ticks_position('left')
	else:
		# no yaxis ticks
		ax.yaxis.set_ticks([])

	if 'bottom' in spines:
		ax.xaxis.set_ticks_position('bottom')
	else:
		# no xaxis ticks
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

def plot_word_frequency(freqDist,cutoff=75,normalized=True,drug=None, poster = False, show=True,savename=None):
	if poster:
		rcParams['axes.linewidth']=2
		rcParams['mathtext.default']='bf'
		rcParams['xtick.labelsize']='large'
		rcParams['ytick.labelsize']='large'
	
	labels,vals = zip(*freqDist.items())
	labels = list(labels)
	vals = np.array(vals).astype(float)
	vals /= vals.sum() #<--- Normalize values

	fig = plt.figure(figsize=(12,8))
	ax = fig.add_subplot(111)
	plt.subplots_adjust(left=0.08)
	
	line, = ax.plot(vals[:cutoff], 'k', linewidth=2)
	#adjust_spines(ax,['bottom','left'])	
	ax.set_ylim(ymax=max(vals)+0.01)

	line.set_clip_on(False)
	
	ax.set_ylabel(r'\Large \textbf{Frequency}')
	ax.set_xticks(np.arange(cutoff))
	ax.set_xlim(0,cutoff)
	ax.set_xticklabels([r'\Large \textbf{%s}'%label for label in labels[:cutoff]], rotation=90)

	if drug:
		ax.annotate(r"\Large \textbf{%s}" %drug.capitalize(), 
			xy=(.5, .5),  xycoords='axes fraction',horizontalalignment='center', verticalalignment='center')

	if savename:
		plt.savefig(savename+'.png')

	if show:
		plt.show()