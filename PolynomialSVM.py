import matplotlib.pyplot as plt
import numpy as np
import sys
from sklearn import svm
from utils.ReadData import *
from utils.Calculations import *

if __name__ == "__main__":
	if len(sys.argv) != 3:
		print("Format: python PolynomialSVM.py TrainData TestData")
		exit(0)

	# Read Data
	TrainX, TrainY = ReadData(sys.argv[1])
	TestX, TestY = ReadData(sys.argv[2])

	# Transform Class(Digit)
	TrainY = TransformData(TrainY, 4)
	TestY = TransformData(TestY, 4)

	# Start Training~~
	logC = [-5, -3, -1, 1, 3]
	TrackEin = []
	for i in logC:
		C = 10**i
		PolySVM = svm.SVC(kernel = "poly", degree = 2, coef0 = 1, gamma = 1, C = C)
		PolySVM.fit(TrainX, TrainY)
		TrackEin.append(cal_Error(PolySVM.predict(TrainX), TrainY))

	# Plot the graph
	plt.figure("Polynomial Soft Margin SVM")
	plt.plot(logC, TrackEin)
	plt.title("$E_{in}$ vs $log_{10}C$")
	plt.show()