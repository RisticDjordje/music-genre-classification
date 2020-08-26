import librosa
import numpy as np
import os
import pathlib
import csv
import warnings
warnings.filterwarnings('ignore')

print("Starting...")

header = 'filename chroma_stft spectral_centroid spectral_bandwidth rolloff zero_crossing_rate'
for i in range(1, 21):
    header += f' mfcc{i}'
header += ' genre'
header = header.split()

file = open('data.csv', 'w', newline='')
with file:
    writer = csv.writer(file)
    writer.writerow(header)

times = '1.5s 2s 5s'.split()
genres = 'jazz klasika pop rnb rock turbo'.split()
for t in times:
    for g in genres:
        pathlib.Path(f'baza/{t}/{g}').mkdir(parents=True, exist_ok=True)
        for filename in os.listdir(f'baza/{t}/{g}'):

            songname = f'baza/{t}/{g}/{filename}'

            y, sr = librosa.load(songname, mono=True, duration=5)

            chroma_stft = librosa.feature.chroma_stft(y, sr)
            spec_cent = librosa.feature.spectral_centroid(y, sr)
            spec_bw = librosa.feature.spectral_bandwidth(y, sr)
            rolloff = librosa.feature.spectral_rolloff(y, sr)
            mfcc = librosa.feature.mfcc(y, sr)
            zcr = librosa.feature.zero_crossing_rate(y)

            to_append = f'{filename} {np.mean(chroma_stft)} {np.mean(spec_cent)} {np.mean(spec_bw)} {np.mean(rolloff)} {np.mean(zcr)}'
            for e in mfcc:
                to_append += f' {np.mean(e)}'
            to_append += f' {g}'

            file = open('data.csv', 'a', newline='')
            with file:
                writer = csv.writer(file)
                writer.writerow(to_append.split())

print("Finished!")


















