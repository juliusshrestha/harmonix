
import os
#import sys
#sys.path.insert(1, os.path.join(sys.path[0], 'utils'))
import numpy as np
import librosa
import tqdm
from glob import glob

sample_rate = 32000
window_size = 2048
hop_size = 1000

mel_bins = 128
#fmin = 5       # Hz
#fmax = 4660

min_audio_length = 120 # 300 = 2x60 = 2 min

frames_per_second = sample_rate // hop_size



def read_audio(audio_path, target_fs=None):
    #(audio, fs) = soundfile.read(audio_path)
    audio, fs = librosa.load(audio_path)

    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
        
    if target_fs is not None and fs != target_fs:
        audio = librosa.resample(audio, orig_sr=fs, target_sr=target_fs)
        fs = target_fs
    
    audio_dur = librosa.get_duration(y=audio, sr=fs)
    #print("The audio duration is: ", audio_dur)
    
    audio_samples = int(sample_rate * min_audio_length)
    
    #Zero padded if audio duration is less than 5 min
    if len(audio) < audio_samples:
        #print("The original audio length:", len(audio))
        audio = np.hstack((audio, np.zeros((audio_samples - len(audio),))))
        #print("The zero-padded audio length:", len(audio))
        audio_dur = min_audio_length
        
    #For clipping   
    #elif n_sample > audio_samples:
    #    signal = signal[int((n_sample-n_sample_fit)/2):int((n_sample+n_sample_fit)/2)]
    
      
    return audio, fs, audio_dur


def compute_mel_spec(audio_name, sample_rate, window_size, hop_size, mel_bins, frames_num):    
                
    # Compute short-time Fourier transform
    stft_matrix = librosa.core.stft(y=audio_name, n_fft=window_size, hop_length=hop_size, window=np.hanning(window_size), center=True, dtype=np.complex64, pad_mode='reflect').T
    #melW = librosa.filters.mel(sr=sample_rate, n_fft=window_size, n_mels=mel_bins, fmin=fmin, fmax=fmax).T
    melW = librosa.filters.mel(sr=sample_rate, n_fft=window_size, n_mels=mel_bins).T
    
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


def main(mp4_directory, hdf5_directory, utterance):
    hdf5_file_mp4 = hdf5_directory + os.path.sep + utterance[len(mp4_directory):] 
    base_hdf5_file = os.path.splitext(hdf5_file_mp4)[0]  #Remove the .mp4 extension

    #Read audio
    audio, fs, audio_dur = read_audio(audio_path=utterance, target_fs=sample_rate)
           
    frames_num = int(frames_per_second * audio_dur) #Total temporal frames = 64*10 =640
            
    #For training (Save each .hdf5 file separately
    feature=compute_mel_spec(audio, fs, window_size, hop_size, mel_bins, frames_num)
    print("The shape of spectrogram is:", feature.shape)
    
    wav_filename = '{}.mp3'.format(base_hdf5_file.split('.')[0])
    npz_output_directory = os.path.join(hdf5_directory,wav_filename)
    npz_output_directory = npz_output_directory.split(".")[0]
    #np.save(os.path.join(hdf5_directory, '{}.npy'.format(wav_filename.split('.')[0])), feature)
    np.savez(npz_output_directory, audio_name=utterance, feature=feature)

    
if __name__ == '__main__':
   
    mp4_directory = 'D:\\Data_Analytics\\Harmoix\\Dataset_Harmonix\\Audio\\'
    hdf5_directory = 'D:\\Data_Analytics\\Harmoix\\Dataset_Harmonix\\Final_Data\\final_audio\\' 
    
    if not os.path.exists(hdf5_directory):
        os.makedirs(hdf5_directory)
    
    for utterance in tqdm.tqdm(list(glob(os.path.join(mp4_directory, '*.mp3'))), position=2, leave=False):
        print("The input utterance dir", utterance)
        main(mp4_directory, hdf5_directory, utterance)  
        
    else:
        raise Exception('Incorrect arguments!')
    
    