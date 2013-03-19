import numpy as np
import scipy as sci
from scipy.io import loadmat
from itertools import chain,izip_longest
from numpy.random import rand
from scipy.stats import scoreatpercentile
from matplotlib.pyplot import *
import matplotlib
from collections import defaultdict

from zlib import compress

matplotlib.rcParams['xtick.direction'] = 'out'
matplotlib.rcParams['ytick.direction'] = 'out'

import cPickle 
#Smooth from Scipy Cookbook
import numpy

def LZC(data): #Data must be zscored for this and the periodic trend removed; otherwise measurement confounded 
  return len(data)/float(len(compress(data))) 

def grouper(iterable, n, fillvalue=None):
    args = [iter(iterable)] * n
    return izip_longest(*args, fillvalue=fillvalue)
def size(nestedList):
	rowcount = 0
	colcount = []
	for row,column in enumerate(nestedList):
		rowcount += 1
		colcount.append(len(column))
	return (rowcount,tuple(colcount))
	
def xcorr(first,second,dt=0.001,window=0.2,savename=None):
	length = min(len(first),len(second))
	diffs = np.array([first[i]-second[j] for i in xrange(length) for j in xrange(length)])
	bins = np.arange(-window,window,dt)
	'''
	counts = []
	for i in bins:
		print (i-.5)*dt,(i+.5)*dt
		counts.append(sum((diffs>((i-.5)*dt)) & (diffs<((i+.5)*dt))))
	'''
	counts = np.bincount(np.digitize(diffs,bins))
	#counts -= len(diffs)*dt/float(max(first[-1],second[-1]))
	#counts /= float(length)
	if savename:
		cPickle.dump((bins,counts),open(savename,'wb'))
	return counts[1:-1]			
 
def delete(nestedList, indices):
	for index in indices:
		del nestedList[index]
	return nestedList 
            
def prints(toPrint, isIndex):
	if type(toPrint) is list: 
		if isIndex:
			for key,value in enumerate(toPrint): print key, ' : ', value
		elif not isIndex: 
			for _,value in enumerate(toPrint): print value
		else: 
			print 'Argument type not recognized.'

def spiketime2binary(spikeTimes, binSize=0.001):
	if len(spikeTimes) is 0: return -1
	spikeTimes = np.around(spikeTimes/binSize).astype(int)
	answer = np.zeros((max(spikeTimes),),  dtype=int)
	answer[spikeTimes-1]=1
	return answer
	
def binary2spiketime(spikeTimes,binSize=0.001,neuron_census=0):
	if len(spikeTimes) is 0: 
		return -1
	else:
		#return np.array([[timepoint for neuron,timepoint in np.where(spikeTimes==1)]])
		neuron_count = spikeTimes.shape[0] if neuron_census==0 else neuron_census
		if neuron_count == 1:
			return np.squeeze(np.array(np.where(spikeTimes==1)))
		else:
			answer= [[] for neuron in range(neuron_count)]
			for neuron,spikeTime in np.transpose(np.array(np.where(spikeTimes==1))):
				answer[neuron].append(spikeTime)
			return answer
			
			
def flatten(aList):
	return [item for sublist in aList for item in sublist]

def toString(spikeTrain): return ''.join([str(spike) for spike in spikeTrain])

def fromFileRasters(filename):
	data = loadmat(filename)
	
def makeKeysStrings(data):
	for key,_ in data.iteritems(): data[str(key)]=data.pop(key)
	return data

def raster_plot(spiketimes):
	trains = map(spiketime2binary,spiketimes)
	cutoff = map(min,trains)
	
	trains = array([train[:cutoff] for train in trains])
	
	fig = figure()
	ax = fig.add_subplot(111)
	ax.imshow(self.images, aspect='auto', cmap=cm.bone_r, interpolation='nearest')
	ax.set_bgcolor('none')
	adjust_spines(ax,['bottom','left'])
	ax.set_xlabel('Time (ms)')
	ax.set_ylabel('Neuron')
	return fig
		
def toArray(dictionary):
	rowCount =max(dictionary)+1
	colCount=max(max(x) for x in dictionary.values())+1
	for rowIndex,row in dictionary.iteritems():
		for colIndex, value in row.iteritems(): answer[rowIndex,colIndex] = value
	return answer
	
def iqr(data):
	return scoreatpercentile(data,75)-scoreatpercentile(data,25)

def sequences(length, alphabet): return [''.join(sequence) for sequence in product(alphabet,repeat=length)]

def allSequences(maxLength,alphabet): return [sequences(length,alphabet) for length in range(1,maxLength+1)]	

 
def permutations(iterable, r=None):
    # permutations('ABCD', 2) --> AB AC AD BA BC BD CA CB CD DA DB DC
    # permutations(range(3)) --> 012 021 102 120 201 210
    pool = tuple(iterable)
    n = len(pool)
    r = n if r is None else r
    if r > n:
        return
    indices = range(n)
    cycles = range(n, n-r, -1)
    yield tuple(pool[i] for i in indices[:r])
    while n:
        for i in reversed(range(r)):
            cycles[i] -= 1
            if cycles[i] == 0:
                indices[i:] = indices[i+1:] + indices[i:i+1]
                cycles[i] = n - i
            else:
                j = cycles[i]
                indices[i], indices[-j] = indices[-j], indices[i]
                yield tuple(pool[i] for i in indices[:r])
                break
        else:
            return
def product(*args, **kwds):
    # product('ABCD', 'xy') --> Ax Ay Bx By Cx Cy Dx Dy
    # product(range(2), repeat=3) --> 000 001 010 011 100 101 110 111
    pools = map(tuple, args) * kwds.get('repeat', 1)
    result = [[]]
    for pool in pools:
        result = [x+[y] for x in result for y in pool]
    for prod in result:
        yield tuple(prod)
        
def combinations(iterable, r):
    pool = tuple(iterable)
    n = len(pool)
    for indices in permutations(range(n), r):
        if sorted(indices) == list(indices):
            yield tuple(pool[i] for i in indices)


def make_step_rewards(step_length,padding_length):
	padding = tile(array([0]),(1,padding_length))
	return hstack((padding,tile(array([1]),(1, step_length)),padding))

def sigmoid(arg): return 1/1+exp(-arg)

def activation_function(associability, predicted_reward,actual_reward, global_inhibition,global_excitation):
	devRate = 0.2
	decayDevRate = 0.5
	maxDev = 0.4
	delta = associability*(actual_reward-predicted_reward)+global_excitation+global_inhibition + (isDruggie)*((1-decayDevRate)*devRate + decayDevRate*maxDev)
	return 1/(1+exp(delta-1))

def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return from_iterable(combinations(s, r) for r in range(len(s)+1))

def kleene(alphabet, baseAlphabet, seqLength):
	if seqLength is None: return alphabet
	else: return kleene(alphabet.extend(concatenate(alphabet, baseAlphabet)),seqLength-1)

def test():	
	base = 'abc'	
	test = permutations(list('abc'),2)
	test = [''.join(item) for item in test]
	test = [[item+thing for thing in base] for item in test]
	test = flatten(test)
	print test
	
def conditionalProbability(past, future, data):
	return data.count(past+future)/float(data.count(future))

def bin(n):
    '''convert denary integer n to binary string bStr'''
    bStr = ''
    if n < 0:  raise ValueError, "must be a positive integer"
    if n == 0: return '0'
    while n > 0:
        bStr = str(n % 2) + bStr
        n = n >> 1
    return bStr

def ternary(ifTrue,ifFalse,condition):
	if condition: return ifTrue
	else: return ifFalse

def flatten(aList):
	if type(aList) is list: return  [item for sublist in aList for item in sublist]
	else: print 'Cannot flatten type: ', type(aList)

def group2(iterator, count): return itertools.imap(None, *([ iter(iterator) ] * count))

def evenProcess(dataPoints, isNoise = False):
	#generate sample data for CSSR
	stateSequence = ''
	for step in range(dataPoints): stateSequence += ternary('A','BA',rand()>0.5)
	observationSequence = ''
	for state in stateSequence:
		if state is 'A' : observationSequence += ternary('1','0',rand()>(0.5 + isNoise*0.1*(rand()-.5)))
		elif state is 'B' : observationSequence += '1'
		else: print 'System in unknown state ', state
	return observationSequence
def nCorr(one,two):
	print 'One : ',one
	one /= sqrt(dot(one,one.conj()))
	print 'One normalized : ',one
	two /= sqrt(dot(two,two.conj()))
	return dot(one,two)

def combine(aList):
	answer = defaultdict(list)
	for key, value in aList: answer[key].append(value)
	return answer.items()
	
def STA(stimulus, response, maxDelay, isPlot = False): 
	 #assume each row of response is a trial
	estimate= array([[nCorr(stimulus[0:-delay],row[delay:]) for delay in range(1,maxDelay+1)] for row in response])
	answer= [(iqr(datum), median(datum)) for datum in transpose(estimate)]
	if isPlot:
	 	figure()
	 	errorbar(range(1,maxDelay+1), zip(*answer)[1], yerr=0.5*array(zip(*answer)[0]),fmt='--o')
	 	#have to plot backwards in time
	 	ax = gca()
	 	xlim((0, maxDelay+1))
	 	_,xMax = xlim()
	 	_,yMax = ylim()
	 	ylim(ymax=1.2*yMax)
	 	xticks(arange(xMax),[str(x) for x in range(int(xMax))])
	 	ax.set_xlim(ax.get_xlim()[::-1])
	 	xlabel('Time Before Spike (ms)')
	 	ylabel('Correlation')
	 	title('Spike Triggered Average')	 	
        show()
 	return answer
 	
if __name__=='__main__':
	dataPoints = 100
	print evenProcess(dataPoints)
	
		
	