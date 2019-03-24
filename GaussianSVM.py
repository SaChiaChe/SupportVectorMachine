import matplotlib.pyplot as plt
import numpy as np
import sys
from sklearn import svm
from utils.ReadData import *
from utils.Calculations import *

if __name__ == "__main__":
	if len(sys.argv) != 3:
		print("Format: python GaussianSVM.py TrainData TestData")
		exit(0)

	# Read Data
	TrainX, TrainY = ReadData(sys.argv[1])
	TestX, TestY = ReadData(sys.argv[2])

	# Transform Class(Digit)
	TrainY = TransformData(TrainY, 0)
	TestY = TransformData(TestY, 0)

	# Start Training~~
	logC = [-2, -1, 0, 1, 2]
	TrackDis = []
	for i in logC:
		C = 10**i
		GaussianSVM = svm.SVC(kernel = "rbf", gamma = 80, C = C)
		GaussianSVM.fit(TrainX, TrainY)
		W = TrainY[GaussianSVM.support_] * GaussianSVM.dual_coef_[0]
		TrackDis.append(cal_Distance(W))

	# Plot the graph
	plt.figure("Gaussian Soft Margin SVM")
	plt.plot(logC, TrackDis)
	plt.title("$Margin$ vs $log_{10}C$")
	plt.show()