import matplotlib.pyplot as plt
import numpy as np
import sys
from sklearn import svm
from utils.ReadData import *
from utils.Calculations import *

if __name__ == "__main__":
	if len(sys.argv) != 3:
		print("Format: python GaussianSVMwithValidation.py TrainData TestData")
		exit(0)

	# Read Data
	TrainX, TrainY = ReadData(sys.argv[1])
	TestX, TestY = ReadData(sys.argv[2])

	# Transform Class(Digit)
	TrainY = TransformData(TrainY, 0)
	TestY = TransformData(TestY, 0)

	# Experiment for 100 times
	TrackBestGamma = []
	for i in range(100):
		print("Experiment", i)
		# Cut Validation
		TrainX_, TrainY_, ValidtionX, ValidtionY = CutValidation(TrainX, TrainY, 1000)

		# Start Training~~
		logGamma = [-2, -1, 0, 1, 2]
		TrackEval = []
		for i in logGamma:
			Gamma = 10**i
			GaussianSVM = svm.SVC(kernel = "rbf", gamma = Gamma, C = 0.1)
			GaussianSVM.fit(TrainX_, TrainY_)
			TrackEval.append(cal_Error(GaussianSVM.predict(ValidtionX), ValidtionY))

		# Choose best from Eval
		print("Eval:", TrackEval)
		print("Best Gamma is 10**" + str(logGamma[TrackEval.index(min(TrackEval))]))
		TrackBestGamma.append(logGamma[TrackEval.index(min(TrackEval))])


	# Plot the graph
	plt.figure("Experiment with Gamma")
	plt.title("$Frequency$ vs $log_{10}Gamma$")
	plt.hist(TrackBestGamma, bins = 5, range = (-2.5, 2.5))
	plt.show()