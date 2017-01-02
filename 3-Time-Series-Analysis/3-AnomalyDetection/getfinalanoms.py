import codecs


def reportanoms(vec, threshold, st, end):
	cnt = 0
	for i in range(st, end):
		if vec[i] > threshold:
			cnt += 1
	print(cnt)


def main():
	f = codecs.open('results/' + 'Test010' + '.txt', 'r', 'utf-8')
	vec = list()
	for line in f:
		vec = eval(line)
		break
	reportanoms(vec, 2, 0, 140)


if __name__ == '__main__':
	main()
