import os


def main():
	folders = os.listdir('Input_TS/')
	for foldername in folders:
		for i in range(200):
			if not os.path.exists('out/' + foldername):
				os.makedirs('out/' + foldername)
			os.system('python detectanoms.py Input_TS/' + foldername +'/'+str(i) + '.csv out1/' + foldername+ '/'+str(i) + '.txt')


if __name__ == '__main__':
	main()
