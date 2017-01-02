import codecs


def readanoms(filename):
	f = codecs.open(filename, 'r', 'utf-8')
	vec = list()
	cnt = 0
	for line in f:
		if cnt > 1:
			if 'Index:' in line:
				break
			tmp = line.split(' ')
			elem = tmp[0] + ' ' + tmp[1]
			vec.append(elem)
		cnt += 1
	return vec


def getindices(datafile, anomvec):
	f = codecs.open(datafile, 'r', 'utf-8')
	vec = list()
	for line in f:
		vec.append(line[:-1])
	indexvec = list()
	for elem in anomvec:
		indexvec.append(vec.index(elem))
	dimvec = [0] * 2600
	for elem in indexvec:
		dimvec[elem] = 1
	return dimvec


def addtovec(vec, dimvec):
	newvec = list()
	for i in range(len(vec)):
		newvec.append(vec[i] + dimvec[i])
	return newvec


def getresults(foldername):
	f = codecs.open(foldername + '.txt', 'w', 'utf-8')
	vec = [0] * 2600
	for i in range(200):
		v = readanoms('out/' + foldername + '/' + str(i) + '.txt')
		dimvec = getindices('timestamp/' + foldername + '/data.txt', v)
		vec = addtovec(vec, dimvec)
	print(vec, file=f)


def main():
	getresults('Test001')
	getresults('Test002')
	getresults('Test004')
	getresults('Test005')
	getresults('Test006')
	getresults('Test008')
	getresults('Test010')
	getresults('Test011')
	getresults('Test012')
	getresults('Test013')
	getresults('Test014')
	getresults('Test015')
	getresults('Test016')
	getresults('Test017')
	getresults('Test019')
	getresults('Test021')
	getresults('Test024')
	getresults('Test025')
	getresults('Test026')
	getresults('Test027')
	getresults('Test028')
	getresults('Test032')
	getresults('Test033')
	getresults('Test034')
	getresults('Test035')
	getresults('Test036')


if __name__ == '__main__':
	main()
