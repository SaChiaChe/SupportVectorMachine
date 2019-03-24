import numpy as np
from random import randint

def cal_Error(Predict, Data):
	N = len(Predict)
	Count = 0
	for i in range(N):
		if Predict[i] != Data[i]:
			Count += 1
	return Count/N

def cal_Distance(W):
	return 1/np.linalg.norm(W)

def CutValidation(DataX, DataY, N):
	Max = len(DataX)-1
	Pick = []
	for i in range(N):
		while(True):
			RandomInt = randint(0, Max)
			if RandomInt not in Pick:
				Pick.append(RandomInt)
				break
			else:
				continue
	
	mask = np.ones(Max+1, dtype = bool)
	mask[Pick] = False
	ValidtionX, ValidtionY = DataX[Pick], DataY[Pick]
	TrainX, TrainY = DataX[mask], DataY[mask]

	return TrainX, TrainY, ValidtionX, ValidtionY