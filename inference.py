
import numpy as np
from pandas import read_csv

from tensorflow.keras.models import load_model


def getCycles(ar, ts):
  seq = []
  arLen = len(ar)

  for i in range(int(np.ceil(arLen/ts))):
    
    i1 = i * ts
    i2 = i * ts + ts 

    if i2 <= arLen:
      # print(i, i1, i2)
      seq.append(ar[i1:i2, :])

    else:
      rem = (arLen - i1)   # the remaining elements to add to the cycle
      arrPadding = i1 - (ts - rem) # complete the cycle with elements from behind at the exact spot
      tempAr = list(range(i1, arLen)) + list(range(arrPadding, i1))
      # print(tempAr)
      seq.append(ar[tempAr, :])


  return np.stack(seq)

def test(path):
	# Normalize the input
	testDf = read_csv(path)

	dfn_mean = testDf.mean()
	dfn_std = testDf.std()
	testDfNorm = (testDf - dfn_mean) / dfn_std

	# Create sequences from test values.
	x_test = getCycles(testDfNorm.values, TIME_STEPS)
	print("Test input shape: ", x_test.shape)

	model = load_model(MODEL_PATH)

	faults = []
	for i in range(x_test.shape[0]):
		# reshape to 3-D single input to the model, i.e. (1, 100, 6)
		x_t = x_test[i].reshape((1, x_test.shape[1],  x_test.shape[2] ))
		x_test_pred = model.predict(x_t)
		# find the error for each feature predictions of 100 (cycle length)
		test_mae_loss = np.mean(np.abs(x_test_pred - x_test[i]), axis=1)

		# sum the errors across feature, i.e. 6 errors
		sum_test_mae_loss = np.sum(np.mean(np.abs(x_test_pred - x_t), axis=1), axis=1)

		# log error to account for the monotonic increase or decrease of fluctuations for inputs different from ideal case, 
		anomaly = np.log(sum_test_mae_loss[0]) > THRESHOLD

		if anomaly: faults.append(i)

	print("\n", len(faults), "of", x_test.shape[0], "faulty cycles")
	print("positions of faults: ", faults)

	# locate the exact observations thata are anomalies
	# fault_location = ind * timestep: ind * timestep + timestep
	loc_faults = []
	for idx in faults:
		loc_faults.extend(list(range(idx * TIME_STEPS, (idx * TIME_STEPS) + TIME_STEPS)))

		# remove the extra padding of anomaly from padding the cycles
		try:
			highestInd = loc_faults.index(len(testDf))
			loc_anomaly = loc_faults[: highestInd]
		except:
			# paddings not in anomaly
			pass

	print("\n observations with faults: ", loc_faults)

if __name__ == '__main__':
	TEST_PATH = "./test.csv"
	MODEL_PATH = "./models/model"

	#  resolution of timestep in microsecond
	TIME_STEPS = 100

	# MAE Reconstruction error obtained foe optimal(non-faulty) cycles with the model
	THRESHOLD = 0.8946312370470929

	test(TEST_PATH)