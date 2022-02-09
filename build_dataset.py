import config
from imutils import paths
import numpy as np
import shutil
import os

def copy_images(imagePaths, folder):
	if not os.path.exists(folder):
		os.makedirs(folder)

	for path in imagePaths:
		imageName = path.split(os.path.sep)[-1]
		label = path.split(os.path.sep)[1]
		labelFolder = os.path.join(folder, label)
		#print(labelFolder)

		if not os.path.exists(labelFolder):
			os.makedirs(labelFolder)

		destination = os.path.join(labelFolder, imageName)
		shutil.copy(path, destination)


imagePaths = list(paths.list_images(config.data_path))
#print(imagePaths)d
np.random.shuffle(imagePaths)
#print(imagePaths)

valPathsLen = int(len(imagePaths) * config.val_split)
#print(valPathsLen)
trainPathsLen = len(imagePaths) - valPathsLen
#print(trainPathsLen)
trainPaths = imagePaths[:trainPathsLen]
valPaths = imagePaths[trainPathsLen:]
print(trainPaths)

copy_images(trainPaths, config.train)
copy_images(valPaths, config.val)