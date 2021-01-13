# This file is used to build a dataset from images
# Microsoft Kaggle Cats and Dogs dataset can be found here:
# https://www.microsoft.com/en-us/download/details.aspx?id=54765
import os
import cv2
import numpy as np
from tqdm import tqdm


REBUILD_DATA = True

DATADIR = os.path.join("PetImages")

class DogsCats():
    # 80px might be too high resolution for weaker machines, consider switching to 50
    IMG_SIZE = 80
    
    CATS = "Cat"
    DOGS = "Dog"
    LABELS = {CATS: 0, DOGS: 1}
    
    training_data = []
    catcount = 0
    dogcount = 0
    
    def make_training_data(self):
        for label in self.LABELS:
            for f in tqdm(os.listdir(DATADIR + os.sep + label)):
                try:
                    # Reading and resizing the image
                    path = os.path.join(DATADIR, label, f)
                    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                    img = cv2.resize(img, (self.IMG_SIZE, self.IMG_SIZE))
                    
                    # np.eye is used here to generate a one-hot encoded label, which is used for MSELoss
                    self.training_data.append([np.array(img), np.eye(2)[self.LABELS[label]]])

                    if label == self.CATS:
                        self.catcount += 1
                    elif label == self.DOGS:
                        self.dogcount += 1
                        
                # There are some files in the dataset which are not images, so just ignore those
                except Exception as e:
                    pass

        # Shuffling and saving the dataset
        np.random.shuffle(self.training_data)
        np.save("catsdogsdata80.npy", self.training_data)
        
        # Print the counters to check the class balance
        # This dataset is perfectly balanced, but this is a good practice
        print("CATS: ", self.catcount)
        print("DOGS: ", self.dogcount)


if REBUILD_DATA:
    dogscats = DogsCats()
    dogscats.make_training_data()
