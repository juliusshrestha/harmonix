# coding=utf-8

import os
import os.path as osp
import pandas as pd
import numpy as np
from collections import OrderedDict

from keras.utils import OrderedEnqueuer
from keras.utils import Sequence

n_class = 63

def multi_thread_infinite_sequence(generator, shuffle=False, workers=2,
                                   max_queue_size=5):
    """Iterate indefinitely over a Sequence.

    # Arguments
        seq: Sequence object

    # Returns
        Generator yielding batches.
    """
    enqueuer = OrderedEnqueuer(generator, use_multiprocessing=True,
        shuffle=shuffle)
    enqueuer.start(workers=workers, max_queue_size=max_queue_size)
    output_generator = enqueuer.get()
    while True:
        for item in output_generator:
            yield item



class AudioGenerator(Sequence):
    def __init__(self, npy_path, csv_path, width, batch_size, lable2index, frames_per_second = 8000 // 250, shuffle=True):
        '''
        :param mp3_path: the root path of audio folder
        :param width: the length of audio want to randomly cropped
        :param xml_path: the path of annotation
        :param batch_size: batch size
        :param shuffle:  whether to shuffle each epoch
        '''
        self.shuffle = shuffle
        self.width = width
        self.batch_size = batch_size
        self.label2index = lable2index
        self.frames_per_second = frames_per_second
        self.data_lists = self.get_audio_paths(npy_path)
#         print("The data list are: ", self.data_lists)
        
        print("The csv_path is:", csv_path)
        
        self.annotation_lists = self.get_annoations(csv_path)
        print("The annotation_lists are: ", self.annotation_lists)
        
        assert self.batch_size <= self.data_length()
        self.on_epoch_end()

    def data_length(self):
        return len(self.data_lists.keys())

    def __len__(self):
        return self.data_length() // self.batch_size

    def on_epoch_end(self):
        self.index = np.arange(self.data_length())

        if self.shuffle:
            np.random.shuffle(self.index)

    def get_audio_paths(self, path):
        file_paths = OrderedDict()
        for root, _, filenames in os.walk(path):
            for filename in filenames:
                if filename.endswith('npz'):
                    name = filename.split('.')[0]
                    file_paths[name] = osp.join(root, filename)
                    #print("The audio name are:", name, file_paths[name])
                    
        return file_paths

    def get_annoations(self, path):
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
    
    #https://www.kaggle.com/carlolepelaars/bidirectional-lstm-for-audio-labeling-with-keras
    def normalize(self, img):
        '''
        Normalizes an array 
        (subtract mean and divide by standard deviation)
        '''
        eps = 0.001
        if np.std(img) != 0:
            img = (img - np.mean(img)) / np.std(img)
        else:
            img = (img - np.mean(img)) / eps
        return img

    def __getitem__(self, index):
        slices = self.index[index * self.batch_size:(index + 1) * self.batch_size]
        x_list, y_list = [], []
        for slice in slices:
            # TODO load the mp3 file into numpy file
            #print(self.data_lists.keys())
            name = list(self.data_lists.keys())[slice]
            # shape (height, width, channels(1))
            
            #converting into spectrogram
            #print("The audio name is: ", self.data_lists[name])
            
            loadeddata = np.load(self.data_lists[name])   
            x =  loadeddata["audio_2D"]
            
            #x = np.load(self.data_lists[name], allow_pickle=True)    #if .npy file
            print("The original audio shape is: ", x.shape)
            #x = np.array(x).transpose((1, 0))
            #print("The reshaped audio shape is: ", x.shape)
            
            mask = np.zeros_like(x)
            gts = self.annotation_lists[name]
            #print("The annotation name is:",self.annotation_lists[name])
            #print("The GT are:",name, gts)
            
            for (cls, start, end) in gts:
                index_label = self.label2index[cls] if cls in self.label2index.keys () else 0
                mask[int(start*self.frames_per_second):int(end*self.frames_per_second)] = index_label
                
            random_crop_start = np.random.randint(0, mask.shape[0] - self.width - 1)
            #print("The random_crop_start value is:", random_crop_start)
            
            y = mask[random_crop_start:random_crop_start + self.width]
            
            #convert them into one-hot
            from keras.utils import to_categorical
            # h, w, c
            y_one_hot = to_categorical(y, num_classes= n_class)
            y_one_hot = y_one_hot[:, 0:1, :]
            #****************** Maks Processing End****************************#
            
            x = x[random_crop_start:random_crop_start + self.width]
            #print("The cropped segment of audio is:", x.shape)
                        
            x = np.expand_dims(x, axis=2)
            
            #Transpose input spectrogram and corresponding mask
            #x = np.array(x).transpose((1, 0, 2))
            #y_one_hot = np.array(y_one_hot).transpose((1, 0, 2))
            
            #print("The input spectrogram is:", x.shape)
            #print("The input label is:", y_one_hot.shape)
            
            x_list.append(self.normalize(x))
            y_list.append(y_one_hot)
        
        #print("The list shape is:", len(x_list), len(y_list))
        
        return np.stack(x_list, axis=0), np.stack(y_list, axis=0)

if __name__ == '__main__':
    #labels = ['Background', 'Zungmori', 'Jinyangzo', 'Zungzungmori', 'Zajinmori', 'Aniri', 'AniriChangzo', 'Changzo','Hweemori', 'Semachi', 'Danjungmari', 'Sichang']
    #labels = ['Background', 'Zungmori', 'Jinyangzo', 'Zungzungmori', 'Zajinmori', 'Aniri', 'AniriChangzo']
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
    generator = AudioGenerator(npy_path='D:\\Data_Analytics\\Harmoix\\Dataset_Harmonix\\Data\\AV_CSV_NPZ', \
                               csv_path = 'D:\\Data_Analytics\\Harmoix\\Dataset_Harmonix\\Data\\segments_csv', \
                               width = 1920, batch_size=4, lable2index= lb_to_idx, shuffle=True)
    
    data_len = generator.data_length()
    print("The data length is: ", data_len)
   
#     generator = multi_thread_infinite_sequence(generator, shuffle=True)\
    generator = iter(generator)
    
    for i in range(10):
        x, y = next(generator)
        print(x.shape, y.shape)

        #rgb, nir = x[0][:, :, 0:3], x[0][:, :, 3]
        #label = y[0]

        # visual_image([rgb, nir], [label], '%s.jpg' % i)
'''
#https://www.youtube.com/watch?v=XyX5HNuv-xE  
#https://github.com/bnsreenu/python_for_microscopists
DB: https://drive.google.com/file/d/1HWtBaSa-LTyAMgf2uaz1T9o1sTWDBajU/view

#class weight from sklearn
from sklearn.utils import class_weight
class_weights = class_weight.compute_class_weight('balabeced', np.unique(mask_reshape_encoded, mask_reshape_encoded_org_shape))
print("The class weights are...", class_weights) # Use it in training
'''