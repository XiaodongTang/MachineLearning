import math


def splitDataSet(dataSet, axis, value):
	retDataSet = []
	for feat in dataSet:
		if feat[axis] == value:
			reducedFeat = feat[:axis]
			reducedFeat.extend(feat[axis+1:])
			retDataSet.append(reducedFeat)
	return retDataSet

def calEntropy(dataSet):
	num	= len(dataSet)
	labelCounts	= {}
	for feat in dataSet:
		label = feat[-1]
		if label not in labelCounts.keys():
			labelCounts[label] = 0
		labelCounts[label] += 1
	entropy = 0.0
	for key in labelCounts:
		prob = float(labelCounts[key]) / num
		entropy -= prob * math.log(prob,2)
	return entropy

def chooseFeature(dataSet):
	numFeat	= len(dataSet[0]) -1
	baseEntropy	= calEntropy(dataSet)
	bestInfo	= 0.0
	bestFeat	= -1
	for i in range(numFeat):
		featList 	= [exp[i] for exp in dataSet]
		unqFeat		= set(featList)
		newEntropy	= 0.0
		for v in unqFeat:
			subDataSet = splitDataSet(dataSet, i, v)
			prob	= len(subDataSet) / float(len(dataSet))
			newEntropy += prob * calEntropy(subDataSet)
		info	= baseEntropy - newEntropy
		print "feat: %s info: %s" % (i, info)
		if (info > bestInfo):
			bestInfo = info
			bestFeat = i
	print bestFeat
	return bestFeat
def readfile(filename):
	dataSet = []
	with open(filename, 'r') as filereads:
		for line in filereads:
			val = []
			message = line.strip().split(',')
			dataSet.append(message)
	chooseFeature(dataSet)


readfile("./feature3.txt")

