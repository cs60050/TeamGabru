from __future__ import print_function
from pyculiarity_mod.pyculiarity import detect_ts


import codecs
import numpy as np
import pandas as pd
import rpy2.robjects
import sys


def main():
	f = codecs.open(sys.argv[2], 'w', 'utf-8')
	data = pd.read_csv(sys.argv[1], usecols=['timestamp', 'count'])
	results = detect_ts(data, max_anoms=0.02, direction='both', only_last='day')
	print("RESULTS: ", results, file=f)


if __name__ == '__main__':
	main()
