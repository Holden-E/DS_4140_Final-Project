#general
import numpy as np
import pandas as pd
#sift
import cv2
from tqdm.auto import tqdm

siftVals = [14, 28, 56, 112, 224]


# import data
train_20_split_df = pd.read_pickle('data/train_20_split.pkl')
train_20_split_df.head()

# Load images; resize to 224x224 to be consistent with image sizes used in the other methods
imgs = train_20_split_df.waferMap.apply(
    lambda x: cv2.resize(x, (224, 224), interpolation=cv2.INTER_NEAREST_EXACT)
)  # Use .sample(N) to only do this for N images

#make a pkl file for each SIFT featurization
for val in siftVals:
    # Create SIFT object
    sift = cv2.SIFT_create()

    # Create BOWKMeansTrainer object
    bow_trainer = cv2.BOWKMeansTrainer(val)  # number of visual words

    # Extract SIFT features and add them to trainer
    keypoints = []
    descriptors = []
    for img in tqdm(imgs):
        kp, des = sift.detectAndCompute(img, None)
        bow_trainer.add(des)
        # Store keypoints and descriptors for later use to avoid recomputing
        keypoints.append(kp)
        descriptors.append(des)

    # Perform k-means clustering to obtain vocabulary of visual words (this takes a bit on 12k images)
    voc = bow_trainer.cluster()

    # Create BOWImgDescriptorExtractor object
    bow_extractor = cv2.BOWImgDescriptorExtractor(sift, cv2.BFMatcher(cv2.NORM_L2))

    # Set vocabulary for extractor
    bow_extractor.setVocabulary(voc)

    # Extract bag of visual words histograms for images
    features = []
    for i, img in enumerate(tqdm(imgs)):
        bow_descriptor = bow_extractor.compute(img, keypoints[i])
        features.append(bow_descriptor)

    # The resulting 'features' variable contains the bag of visual words histograms for each image
    # This is a list of 2D arrays (each array/histogram is 1 x n_visual_words); use np.vstack
    features = np.vstack(features)

    # As before, we can use this to create a new column in our dataframe
    sift_vectors = [features[i] for i in range(len(features))]
    train_20_split_df["sift"] = sift_vectors

    #save as pickle files
    fp = 'data/train_20_split_sift' + str(val) + '.pkl'
    train_20_split_df.to_pickle(path = fp)
