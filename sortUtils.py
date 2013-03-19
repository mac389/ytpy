from numpy import *
from scipy import *

import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from os.path import isfile, splitext
import neuroTools as postdoc

from scipy.stats import scoreatpercentile, percentileofscore
from time import time

from brian import *
from brian.library.electrophysiology import *

from itertools import product

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from time import time
from matplotlib import rcParams

from numpy.random import random
from scipy.linalg import svd

from scipy.signal import fftconvolve

rcParams['text.usetex'] = True
def update_progress(progress):
	print '\r[{0}] {1}%'.format('#'*(progress/10), progress)

def partition(filename, filtered=True):
	partitions = ['before','during','after'] #It would be great if these came out naturally
	data = (read_DDT(filename),300,7000,20000) if filtered else read_DDT(filename) #Default is to read the unfiltered trace
	chunkLength = len(data)/len(partitions) #Deliberately dividing by integer
	
	name,_ = splitext(filename)
	for chunk in range(len(partitions)):
		start = i*chunkLength
		stop = (i+1)*chunkLength
		savetxt(name+'.'+partitions[chunk],data[start:stop],delimiter='\t')
	print 'Everything saved'
	
def NEO(data):
	#From Kim and Kim (2000), an IEEE paper
	answer = data*data
	answer -= (roll(data,1)*roll(data,-1))
	return r_[0,answer[1:-1],0]
	
def get_waveforms(data,spiketimes,lookback=100,lookahead=100):
	offsets = arange(-lookback,lookahead)
	indices = array(spiketimes + offsets[:,None]).astype(int)
	ret = take(data,indices,mode='clip')
	ret[:,spiketimes<lookback]=0
	ret[:,spiketimes+lookahead>=len(data)]=0
	return ret

def get_channel_id(filename):
	name,_ = splitext(filename)
	return str(int(name[-3:]))

def STA(lfp,spikes,window):
	return get_waveforms(lfp,spikes,lookback=window/2,lookahead=window/2)
	
def make_lfp(ddtname=None,cutoff=50):
	return butter_bandpass_filter(read_DDT(ddtname),lowcut=.1,highcut=cutoff,fs=20000)

def add_noise(data,amplitude=10,count=3000):
	return hstack((data,amplitude*random(size=(data.shape[0],count))))

def savepass(data=None,savename=None):
	data.tofile(savename)

def switch_type(inputfile,desired):
	name,_ = splitext(inputfile)
	return name+'.'+desired
	
def detect_spikes(data,threshold, fast=False, refractory=10): 
	print 'Detecting spikes from filtered trace\n .Threshold is %0.2f' % float(threshold),'mV'	
	if fast:
		crossings = where(data>threshold)[0]
		intervals= diff(crossings)
		res = crossings[intervals>refractory]	
	else:
		res = spike_peaks(data,vc=threshold)
	return res

def save_waveforms(self):
	fig = plt.figure()
	waveform_panel = fig.add_subplot(211)
	waveform_panel.plot(self.data['wfs'][::self.skip])
	start = 20000
	stop = 40000
	energy_panel = fig.add_subplot(212)
	energy_panel.plot(self.data['energy'][(70*start):(80*start):10],'b')
	energy_panel.set_xlabel(r'time $\left(ms\right)$')
	energy_panel.set_ylabel(r'Energy $\left(mV^{2}\right)$')
	
	#Draw threshold
	energy_panel.axhline(y=self.data['constants']['threshold'],linewidth=1,color='r',linestyle='--')
	plt.savefig(self.IO['savepath']+'/'+self.IO['name']+'_waveforms.png',dpi=300)

		
def threshold(data=None):
	return 5*median(absolute(data-median(data)))

def filtered_trace(filename,lowcut=300,highcut=7000,sampling_rate=20000,show=False, verbose=False,save=True):
	#Assume that filename points to unfiltered data stored in a binary DDT format
	if verbose:
		print 'Filtering {} between {} and {} Hz'.format(filename,lowcut,highcut)
		print 'Assuming a sampling rate of {} Hz'.format(sampling_rate) 
	filtered_data = butter_bandpass_filter(read_DDT(filename),lowcut,highcut,sampling_rate)
	filtered_data /= .1
	if save:
		extension = '.filtered'
		name,_ = splitext(filename)
		name += extension
		filtered_data.tofile(name)
		
	if verbose: print 'Saved %s to file' % name
	if show: return filtered_data

def save_voltage_trace(unfiltered=None,filtered=None,threshold = 0, roi=30000,spread=10000,save=None):
	fig = plt.figure()
	trace_panel = fig.add_subplot(211)
	start = roi-spread
	stop = roi+spread
	trace_panel.plot(unfiltered,'b') #Downsample just for display
	trace_panel.set_xlabel(r'Time $\left(ms\right)$')
	trace_panel.set_ylabel(r'Voltage $ \left(\mu V \right)$')
	
	spike_panel = fig.add_subplot(212)
	spike_panel.plot(filtered,'b')
	spike_panel.set_xlabel(r'time $\left(ms\right)$')
	spike_panel.set_ylabel(r'Voltage $\left(\mu V \right)$')
	
	#Draw threshold
	spike_panel.axhline(y=threshold,linewidth=1,color='r',linestyle='--')
	spike_panel.axhline(y=-threshold,linewidth=1,color='r',linestyle='--')
	
	if save:
		plt.savefig(save+'_voltage.png',dpi=100)
	plt.close()
								
def princomp(A,numpc=3):
	print A.shape
	# computing eigenvalues and eigenvectors of covariance matrix
	M = (A-mean(A.T,axis=1)).T # subtract the mean (along columns)
	[latent,coeff] = linalg.eigh(cov(M))
	p = size(coeff,axis=1)
	idx = argsort(latent) # sorting the eigenvalues
	idx = idx[::-1]	   # in ascending order
	# sorting eigenvectors according to the sorted eigenvalues
	coeff = coeff[:,idx]
	latent = latent[idx] # sorting eigenvalues
	if numpc < p or numpc >= 0:
		coeff = coeff[:,range(numpc)] # cutting some PCs
	score = dot(coeff.T,M) # projection of the data in the new space
	return coeff,score,latent


def pca(data,fraction = 0.9):
	#first center A
	data = (data-data.mean(axis=0))/data.std(axis=0)
	u,s,vt = svd(data,full_matrices=False)
	v = vt.T
	#eigenvalues are returned sorted in increasing order
	ind = argsort(s)[::-1]
	u = u[:,ind]
	s = s[ind]
	v = v[:,ind]
	eigenvalues = s * s
	cumvariances = cumsum(eigenvalues)
	cumvariances /= cumvariances[-1]
	S = diag(s)
	npc = where(cumvariances > fraction)[0][0]
	print npc
	
	params = {'npc':npc,'cumvariances':cumvariances,'u':u,'s':s,'v':v}
	return dot(u[:,:npc],dot(S[:npc,:npc],v[:,:npc].T)),params

def toxy(data):
	x,y=[],[]
	[(x.append(a[0]), y.append(a[1])) for a in data]
	return x,y

def to_full_matrix(til_data):
	rnk = len(til_data)
	answer = zeros((rnk,rnk))
	for idx in xrange(rnk):
		span = len(til_data[idx])
		answer[idx,:span] = til_data[idx]
	return answer

def find_sing_val_cutoff(data,cutoff=0.95):
	data /= data.sum()
	return where(cumsum(data)>0.95)[0][0]


def cluster(data, threshold = 0.5,method='sk', preprocess=True):
	length = len(data)
	print data.shape
	nclus = 2
	nclusmax=8
	sil = [-1]
	models=[]
	if preprocess==True:
		print 'Preprocessing by scaling each row by its range'
		data /= (amax(data,axis=0)-amin(data,axis=0))[newaxis,:]
		print 'Now to cluster'	
	if method == 'sk':
		print 'Clustering using Scikits K-means implementation'
		print "This option returns a tuple of"
		print "\t\t (kmeans object, silhouette coefficients)"
		while nclus < nclusmax: #average(sil[-1]) < threshold and
			model = KMeans(init='k-means++',n_clusters=nclus) 
			#Assume data is propery preprocessed
			model.fit(data)
			labels = model.labels_
			#<-- can only sample this in chunks of 100
			print data.shape
			print 'Calculating silhouette_score '
			sil.append(silhouette_score(data,labels,metric='euclidean')) 
			models.append(model)
			print 'For %d clusters, the silhouette coefficient is %.03f'%(nclus,sil[-1])
			nclus += 1
		return (models,sil)
	elif method == 'pyclus':
		import Pycluster as pc
		print 'Clustering using the C Clustering library'
		print 'This option returns a dictionary with the distance matrix, silhouettes, and clusterids for each iteration.'
		res = []
		sil_co_one = 1
		sil_co = [1]
		#Assume 
		while sil_co_one > threshold and nclus < nclusmax:
			print 'No. of clus: %d'%nclus
			print 'Before kcluster'
			clustermap,_,_ = pc.kcluster(data,nclusters=nclus,npass=50)
			print 'After kcluster'
			centroids,_ = pc.clustercentroids(data,clusterid=clustermap)
			print 'After centroids'
	
			m = pc.distancematrix(data)
			
			print 'Finding mass'
			#Find the masses of all clusters
			mass = zeros(nclus)
			for c in clustermap:
				mass[c] += 1
		
			#Create a matrix for individual silhouette coefficients
			sil = zeros((len(data),nclus))
			
			print 'Evaluating pairwise distance'
			#Evaluate the distance for all pairs of points		
			for i in xrange(0,length):
				for j in range(i+1,length):
					d = m[j][i]
					
					sil[i, clustermap[j] ] += d
					sil[j, clustermap[i] ] += d
			
			#Average over cluster
			for i in range(0,len(data)):
				sil[i,:] /= mass
			
			print 'Sil co'	
			#Evaluate the silhouette coefficient
			s = 0
			for i in xrange(0,length):
				c = clustermap[i]
				a = sil[i,c] 
				b = min( sil[i, range(0,c) + range(c+1,nclus)])
				si = (b-a)/max(b,a) #silhouette coefficient of point i
				s+=si
						
			nclus += 1
			sil_co.append( s/length)
			sil_co_one = s/length
			print 'Sil co %.02f'%sil_co_one
			res.append({'clustermap':clustermap,
						'centroids':centroids,
						 'distances':m,
						 'mass':mass,
						 'silhouettes':sil_co})
		return res

def extract_waveforms(data,spiketimes,lookahead,lookback,onsets):	
	STA = spike_shape(data,onsets=onsets,before=lookback,after=lookahead)
	slope = slope_threshold(data,onsets=onsets,T=int(5*ms/defaultclock.dt))
	return (STA,slope)

def butter_bandpass(lowcut,highcut,fs,order=2):
	nyq = 0.5*fs
	low = lowcut/nyq
	high = highcut/nyq
	
	b,a = butter(order, [low, high], btype='band')
	return b,a

def continuous_function(data):
	#Convolve time series with a causal exponential to make a continuous rate function that one can do PCA on
	alpha = 0.01 # From p12 of Dayan + Abbott
	kernel = array(map(lambda x: alpha*alpha*x*exp(-alpha*x) ,range(20)))
	kernel *= (kernel>0) #Rectification for causality
	kernel /= kernel.sum() #Normalization
	#Don't worry about padding because these series are really long in length
	return convolve(kernel,data) # After about 20 ms will introduce spurious correlations
	
	
def low_d(u,s,v,cutoff=2):
	sigma = diag(s)
	sigma[cutoff:][cutoff:] = 0
	print u[:,:len(s)].shape
	print sigma.shape
	a_star = dot(u[:,:len(s)],sigma)
	a_star = dot(a_star,v[:,:cutoff])
	return a_star

def butter_bandpass_filter(data,*args, **kwargs):
	b,a = butter_bandpass(*args, **kwargs)
	return filtfilt(b,a,data) 

def draw_sizebar(ax):
	from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
	# draw a horizontal bar with length of 0.1 in Data coordinate
	# (ax.transData) with a label underneath.
	asb =  AnchoredSizeBar(ax.transData,
						  10,
						  r"$10$",
						  loc=8,
						  pad=0.1, borderpad=0.5, sep=5,
						  frameon=False)
	ax.add_artist(asb)

def read_DDT(filename,OFFSET=432):
	with open(filename,'rb') as stream:
		stream.seek(OFFSET)
		return fromfile(stream,dtype='int16')
