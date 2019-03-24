import matplotlib.pyplot as plt
import numpy as np
import sys
from sklearn import svm
from utils.ReadData import *

if __name__ == "__main__":
	if len(sys.argv) != 3:
		print("Format: python LinearSVM.py TrainData TestData")
		exit(0)

	# Read Data
	TrainX, TrainY = ReadData(sys.argv[1])
	TestX, TestY = ReadData(sys.argv[2])

	# Transform Class(Digit)
	TrainY = TransformData(TrainY, 2)
	TestY = TransformData(TestY, 2)

	# Start Training~~
	logC = [-5, -3, -1, 1, 3]
	TrackW = []
	for i in logC:
		C = 10**i
		LinearSVM = svm.SVC(kernel = "linear", C = C)
		LinearSVM.fit(TrainX, TrainY)
		# print(C, LinearSVM.coef_[0], np.linalg.norm(LinearSVM.coef_[0]))
		TrackW.append(np.linalg.norm(LinearSVM.coef_[0]))

	# Plot the graph
	plt.figure("Linear Soft Margin SVM")
	plt.plot(logC, TrackW)
	plt.title("$||w||$ vs $log_{10}C$")
	plt.show()