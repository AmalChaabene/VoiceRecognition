import librosa
import os
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import librosa.display

DATA_PATH = "./data/"


# Input: Folder Path
# Output: Tuple (Label, Indices of the labels, one-hot encoded labels)
def get_labels(path=DATA_PATH):
    labels = os.listdir(path)
    label_indices = np.arange(0, len(labels))
    return labels, label_indices, to_categorical(label_indices)


# convert wav to mfcc vectors
def wav2mfcc(file_path, max_len=249):
    wave, sr = librosa.load(file_path, mono=True, sr=None) #monaural / stereophonic sound (un seul channel/+ieurs channels)
    #sr=44.1khz
    mfcc = librosa.feature.mfcc(wave, sr, n_mfcc=40, n_fft=1764,hop_length=882)  #sampling rate / default hop value 512
#taaille de la fenetre=40ms
#default=2048 " a recording with a duration of 60 seconds will contain 2,646,000 samples."
    print(file_path)

    #Encode redundancy in MFCCs: By overlapping frames (at > 50 %) we have increased redundancy in the MFCCs. However, this redundancy can be exploited in improved delta-MFCCs and delta-delta-MFCCs computation as we will have a more smooth difference function approximation
    
    # A ce niveau il faut définir une taille générale pour tous les audios pour que ConvNet fonctionne correctemen
    # si la longueur de l'audio est < à la longueure maximale on termine par 0 
    # Sinon on coupe l'audio
    
    #1er cas
    #print("shape")
    #print(mfcc.shape)
        
    #plt.figure(figsize=(10, 4))
    #librosa.display.specshow(mfcc, x_axis='time')
    #plt.colorbar()
    #plt.title('MFCC')
    #plt.tight_layout()
    if (max_len > mfcc.shape[1]):
        pad_width = max_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')

    # 2ème cas
    else:
        mfcc = mfcc[:, :max_len]
    
    return mfcc

#on va maintenant définir une fonction qui va lire tous les fichiers audios dans un dossier donné et transformer chaque audio en des vecteurs et les sauvegarder dans un fichier .npy en lui donnant le nom du label

def save_data_to_array(path=DATA_PATH, max_len=249):
    labels, _, _ = get_labels(path)

    for label in labels:
        # Init mfcc vectors
        mfcc_vectors = []

        wavfiles = [path + label + '/' + wavfile for wavfile in os.listdir(path + '/' + label)]
        for wavfile in tqdm(wavfiles, "Saving vectors of label - '{}'".format(label)):
            mfcc = wav2mfcc(wavfile, max_len=max_len)
            mfcc_vectors.append(mfcc)
        np.save(label + '.npy', mfcc_vectors)


# Dans cette partie, nous allons diviser notre dataset en training et testing parts.
# nous allons utiliser la fonction train_test_split de la librairie sklearn qui va générer automatiquement cette division

def get_train_test(split_ratio=0.7):
    # Get available labels
    labels, indices, _ = get_labels(DATA_PATH)

    # Getting first arrays
    X = np.load(labels[0] + '.npy')
    print("train");
    print(X.shape)
    y = np.zeros(X.shape[0])

    # Append all of the dataset into one single array, same goes for y
    for i, label in enumerate(labels[1:]):
        x = np.load(label + '.npy')

        X = np.vstack((X, x)) #vertical
        y = np.append(y, np.full(x.shape[0], fill_value= (i + 1)))

    assert X.shape[0] == len(y)

    return train_test_split(X, y, test_size= (1 - split_ratio), shuffle=True)
