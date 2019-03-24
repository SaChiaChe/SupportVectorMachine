import numpy as np

def ReadData(Filepath):
	X, Y = [], []
	with open(Filepath) as f:
		for line in f:
			Line = line.strip().split()
			Digit, Intensity, Symmetry = int(float(Line[0])), float(Line[1]), float(Line[2])
			Y.append(Digit)
			X.append([Intensity, Symmetry])

	return np.array(X), np.array(Y)

def TransformData(DataY, TargetDigit):
	TransformedY = []
	for i in DataY:
		if i == TargetDigit:
			TransformedY.append(1)
		else:
			TransformedY.append(-1)
	return np.array(TransformedY)