import warnings

#from cv2 import eigenNonSymmetric
warnings.filterwarnings('ignore')

import os
import numpy as np
import librosa
import tqdm
#import h5py as h5
from glob import glob
import cv2

from collections import OrderedDict
import os.path as osp

#for audio
sample_rate = 8000
window_size =512 
hop_size = 250
mel_bins = 32
fmin = 5
fmax = 4660
frames_per_second_audio = sample_rate // hop_size
frames_per_second_video = 2

#for video
color = True #use RGB image (True) or grayscale image (False)
skip = True #Get frames at interval(True) or continuously (False)

img_rows, img_cols = 128, 128
    
#Video Processing

def video_frames(filename, width, height, depth, color=False, skip=True):
    cap = cv2.VideoCapture(filename)
    nframe = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    print("Total number of video frames are:", nframe)
    
    if skip:
        frames = [x * nframe / depth for x in range(depth)]
    else:
        frames = [x for x in range(depth)]
    framearray = []
    for i in range(depth):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frames[i])
        ret, frame = cap.read()
        frame = cv2.resize(frame, (height, width), interpolation = cv2.INTER_CUBIC)            
        if color:
            framearray.append(frame)
        else:
            framearray.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))

    cap.release()

    X = np.array(framearray)    #vid3d.video3d(video_dir, color=color, skip=skip)
    #print("CV2 video shape:", X.shape)

    if color:
        return np.array(X).transpose((1, 2, 0, 3))
    else:
        return np.array(X).transpose((1, 2, 0, 0))


#Audio Processing

def read_audio(audio_path, target_fs=None):
    audio, fs = librosa.load(audio_path)
    
    if target_fs is not None and fs != target_fs:
        audio = librosa.resample (audio, orig_sr = fs, target_sr = target_fs)
        fs = target_fs
    audio_duration = librosa.get_duration(y=audio,sr=fs)
    return audio,fs,audio_duration

def compute_mel_spec(audio, sample_rate,window_size,hop_size,mel_bins,frames_num, fmin,fmax):

    # Compute short-time Fourier transform
    stft_matrix = librosa.core.stft(y=audio, n_fft=window_size, hop_length=hop_size, window=np.hanning(window_size), center=True, dtype=np.complex64, pad_mode='reflect').T
    melW = librosa.filters.mel(sr=sample_rate, n_fft=window_size, n_mels=mel_bins, fmin=fmin, fmax=fmax).T
    
    # Mel spectrogram
    mel_spectrogram = np.dot(np.abs(stft_matrix) ** 2, melW)
    # Log mel spectrogram
    logmel_spc = librosa.core.power_to_db(mel_spectrogram, ref=1.0, amin=1e-10, top_db=None)
    #logmel_spc = np.expand_dims(logmel_spc, axis=0)
    logmel_spc = logmel_spc.astype(np.float32)
    #print("The shape of logmel_spc:", logmel_spc.shape)
    #logmel_spc = np.array(logmel_spc).transpose((2, 1, 0))
    logmel_spc = logmel_spc[0 : frames_num]
    #print("The shape of spectrogram is:", logmel_spc.shape)
        
    return logmel_spc

import pandas as pd

def get_csv_annoations(path):
    #TODO read CSV and convert to list
    """
    for example, csv file
    class,start,end
    0,2,100
    1,100,200
    3,200,300
    """
    #print("The csv file name is:", path)
    #df = pd.read_csv(path, header = None, skiprows = 1, names=["sound_event_recording", "start_time", "end_time"])
    df = pd.read_csv(path, header = None, skiprows = 1)
    data_label= df.values.tolist()
    
    return data_label

def get_csv_annoations_org(path):
    file_paths = OrderedDict()
    for root, _, filenames in os.walk(path):
        for filename in filenames:
            if filename.endswith('csv'):
                name = filename.split('.')[0]
                #TODO read CSV and convert to list
                """
                for example, xml file
                class,start,end
                0,2,100
                1,100,200
                3,200,300
                """
                path = osp.join(root, filename)
                #print("The csv file name is:", path)
                #df = pd.read_csv(path, header = None, skiprows = 1, names=["sound_event_recording", "start_time", "end_time"])
                df = pd.read_csv(path, header = None, skiprows = 1)
                file_paths[name] = df.values.tolist()
    return file_paths




def main(output_directory, audio, video, csv, lb_to_idx):
    #file name of the 
    file = audio.split('\\')[-1]
    file_name = file.split('.')[0]
    print(file_name)
    #Read audio
    audio, fs, audio_dur = read_audio(audio_path=audio, target_fs=sample_rate)
           
    #frames_num = int(2 * audio_dur) #required number of frames per second * duration of audio
    frames_num_audio = int(frames_per_second_audio * audio_dur)
    frames_num_video = int(frames_per_second_video * audio_dur)  
    
    #Audio processing
    mel_spc=compute_mel_spec(audio, fs, window_size, hop_size, mel_bins, frames_num_audio, fmin, fmax)
    print("The shape of spectrogram is:", mel_spc.shape)
    
    #Video processing
    video = video_frames(video,width=img_rows,height=img_cols,depth=frames_num_video,color=color,skip=skip)
    print("The shape of video is:", video.shape)
    
    #CSV processing
    csv_annot = get_csv_annoations(csv)
    print("The shape of csv_annot is:", csv_annot)
    
    #Mask Generation
    mask = np.zeros_like(mel_spc)
    gts = csv_annot
    #print("The annotation name is:",self.annotation_lists[name])
    #print("The GT are:",name, gts)
    
    for (cls, start, end) in gts:
        #print("The csv annoation is:", cls, start, end)
        index_label = lb_to_idx[cls] if cls in lb_to_idx.keys () else 0     #One hot encoding
        #print("The index label is:", index_label)
        mask[int(start*frames_per_second_audio):int(end*frames_per_second_audio)] = index_label
    
    print("The shape of mask is:", mask.shape)    
    
    output_directory = os.path.join(output_directory,file_name)
    np.savez(file =output_directory,csv_label=csv_annot, audio_2D=mel_spc, video_3D=video, mask_2D=mask)


if __name__ == '__main__':
    audio_directory = "D:\\Data_Analytics\\Harmoix\\Dataset_Harmonix\\Data\\Audio"
    video_directory = "D:\\Data_Analytics\\Harmoix\\Dataset_Harmonix\\Data\\Video"
    csv_directory = "D:\\Data_Analytics\\Harmoix\\Dataset_Harmonix\\Data\\segments_csv"
    output_directory = "D:\\Data_Analytics\\Harmoix\\Dataset_Harmonix\\Data\\AV_CSV_NPZ"

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    audios = list(glob(os.path.join(audio_directory, '*.mp3')))
    videos = list(glob(os.path.join(video_directory, '*.mp4')))
    csv = list(glob(os.path.join(csv_directory, '*.csv')))
    
    #print(audios)
    #print(videos)
    #print(csv)
    
    #For label
    labels=['intro', 'verse', 'chorus', 'outro', 'silence', 'bridge', 'prechorus', \
            'instrumental', 'breakdown', 'solo', 'postchorus', 'chorus_instrumental', \
            'opening', 'quiet', 'gtr', 'break', 'verseinst', 'verse_slow', 'bre', 'drumroll', \
            'gtrbreak', 'bigoutro', 'vocaloutro', 'fadein', 'instrumentalverse', 'introverse', \
            'intropt', 'chorusinst', 'inst', 'mainriff', 'postverse', 'oddriff', 'end', 'slow', \
            'synth', 'outroa', 'fast', 'slowverse', 'instintro', 'altchorus', 'instchorus', 'miniverse', \
            'quietchorus', 'versepart', 'chorushalf', 'guitarsolo', 'introchorus', 'choruspart', 'preverse',\
             'stutter', 'raps', 'guitar', 'instbridge', 'worstthingever', 'build', 'saxobeat', 'intchorus', \
             'rhythmlessintro', 'transition', 'section', 'versea', 'transitiona', 'refrain']
    
    lb_to_idx = {lb: idx for idx, lb in enumerate(labels)}

    #print(audios,videos)
    for i in tqdm.tqdm(range(len(audios))):
        main(output_directory, audios[i],videos[i], csv[i], lb_to_idx)
        
        