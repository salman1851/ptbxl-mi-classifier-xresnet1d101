import numpy as np
import os
from sklearn.model_selection import train_test_split
from params import *
import glob

# load and stack numpy arrays
X, y = [], []
for cls_name, cls_id in CLASS_MAP.items():
	npy_file_list = glob.glob(DATA_PATH + cls_name + "/*.npy")
	for f in npy_file_list:
		sig = np.load(f).astype(np.float32)
		sig = sig.T # transpose to the canonical "channels x samples" format
		X.append(sig)
		y.append(cls_id)
X = np.array(X)
y = np.array(y)

# split into train, test and validation
X_temp, X_test, y_temp, y_test = train_test_split(X, y, random_state = RANDOM_SEED, stratify = y, test_size=TRAIN_VAL_TEST[2])
val_ratio = TRAIN_VAL_TEST[1] / (TRAIN_VAL_TEST[0] + TRAIN_VAL_TEST[1])  
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, random_state = RANDOM_SEED, stratify = y_temp, test_size=val_ratio)

# print(len(X_train)/len(X), len(X_val)/len(X), len(X_test)/len(X))

# save as npy file in 'processed folder'
if not os.path.exists(PROCESSED_DATA_DIR):
	os.makedirs(PROCESSED_DATA_DIR)
np.save(PROCESSED_DATA_DIR+'X_train', X_train)
np.save(PROCESSED_DATA_DIR+'y_train', y_train)
np.save(PROCESSED_DATA_DIR+'X_val', X_val)
np.save(PROCESSED_DATA_DIR+'y_val', y_val)
np.save(PROCESSED_DATA_DIR+'X_test', X_test)
np.save(PROCESSED_DATA_DIR+'y_test', y_test)