import os
import numpy as np
import time
import pickle
import warnings
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping, CSVLogger, ReduceLROnPlateau
#from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger, ReduceLROnPlateau
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, roc_auc_score, roc_curve, auc
from scipy import interp
from keras import optimizers
warnings.filterwarnings('ignore')

from HR_Net import seg_hrnet

from data_generator import AudioGenerator
from metrics import pixel_accuracy,Mean_IOU

labels=['intro', 'verse', 'chorus', 'outro', 'silence', 'bridge', 'prechorus', 'instrumental', 'breakdown', 'solo', 'postchorus', 'chorus_instrumental', 'opening', 'quiet', 'gtr', 'break', 'verseinst', 'verse_slow', 'bre', 'drumroll', 'gtrbreak', 'bigoutro', 'vocaloutro', 'fadein', 'instrumentalverse', 'introverse', 'intropt', 'chorusinst', 'inst', 'mainriff', 'postverse', 'oddriff', 'end', 'slow', 'synth', 'outroa', 'fast', 'slowverse', 'instintro', 'altchorus', 'instchorus', 'miniverse', 'quietchorus', 'versepart', 'chorushalf', 'guitarsolo', 'introchorus', 'choruspart', 'preverse', 'stutter', 'raps', 'guitar', 'instbridge', 'worstthingever', 'build', 'saxobeat', 'intchorus', 'rhythmlessintro', 'transition', 'section', 'versea', 'transitiona', 'refrain']

lb_to_idx = {lb: idx for idx, lb in enumerate(labels)}

train_db_path = 'D:\\Data_Analytics\\Harmoix\\Dataset_Harmonix\\data_train_test\\train\\final_audio\\'
train_label_csv = 'D:\\Data_Analytics\\Harmoix\\Dataset_Harmonix\\data_train_test\\train\\segments_csv\\'
val_db_path = 'D:\\Data_Analytics\\Harmoix\\Dataset_Harmonix\\data_train_test\\val\\final_audio\\'
val_label_csv = 'D:\\Data_Analytics\\Harmoix\\Dataset_Harmonix\\data_train_test\\val\\segments_csv'

def plot_history(history, result_dir):
    plt.plot(history.history['acc'], marker='.')
    plt.plot(history.history['val_acc'], marker='.')
    plt.title('model accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.grid()
    plt.legend(['accuracy', 'val_accuracy'], loc='lower right')
    plt.savefig(os.path.join(result_dir, 'model_accuracy.png'))
    plt.show()
    plt.close()

    plt.plot(history.history['loss'], marker='.')
    plt.plot(history.history['val_loss'], marker='.')
    plt.title('model loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.grid()
    plt.legend(['loss', 'val_loss'], loc='upper right')
    plt.savefig(os.path.join(result_dir, 'model_loss.png'))
    plt.show()
    plt.close()

           
def save_history(history, result_dir):
    loss = history.history['loss']
    acc = history.history['acc']
    val_loss = history.history['val_loss']
    val_acc = history.history['val_acc']
    nb_epoch = len(acc)

    with open(os.path.join(result_dir, 'History_result.txt'), 'w') as fp:
        fp.write('epoch\tloss\tacc\tval_loss\tval_acc\n')
        for i in range(nb_epoch):
            fp.write('{}\t{}\t{}\t{}\t{}\n'.format(
                i, loss[i], acc[i], val_loss[i], val_acc[i]))


def create_folder(fd):
    if not os.path.exists(fd):
        os.makedirs(fd)

#https://www.kaggle.com/carlolepelaars/bidirectional-lstm-for-audio-labeling-with-keras
def normalize(img):
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

#Different Loss functions

from keras.losses import categorical_crossentropy
from keras import backend as K

def tversky(y_true, y_pred, smooth=1e-12, alpha=0.5):
    '''
    :param y_true:
    :param y_pred:
    :param smooth:
    :param alpha: default 0.5 equal to dice loss, alpha < 0.5:Increase (prediction is not in this category, it is false (
     Actually this type)) sensitivity, alpha> 0.5: Increase (the prediction is this type, is false (actually it is this type)) sensitivity
    :return:
    '''
    nb_classes = 63
     #y_pred.get_image_shape()[-1]
    if nb_classes > 1:
        axis = (0, 1, 2)
        y_true_pos = y_true
        y_pred_pos = y_pred
    else:
        axis = None
        y_true_pos = K.flatten(y_true)
        y_pred_pos = K.flatten(y_pred)

    # At the same time right
    true_pos = K.sum(y_true_pos * y_pred_pos, axis=axis)
    # False negative, right, but wrong
    false_neg = K.sum(y_true_pos * (1 - y_pred_pos), axis=axis)
    # False positive, not right, but classified as right
    false_pos = K.sum((1 - y_true_pos) * y_pred_pos, axis=axis)

    dem = K.sum((true_pos + smooth) / (
            true_pos + alpha * false_neg + (1 - alpha) * false_pos + smooth))
    return nb_classes - dem


#Working Well
def cce_tversky_loss(y_true, y_pred, alpha=0.5):
    '''
    :param y_true: [?, h, w, c] one hot coding
    :param y_pred: [?, h, w, c] probability
    :return:
    '''
    loss = categorical_crossentropy(y_true, y_pred) + tversky(y_true, y_pred,
                                                              alpha=alpha)
    return loss
  

if __name__ == '__main__':
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    
    
    nb_batch = 4
    num_epoch =3000
    nb_classes = 63
    chunk_length = 1920 #320x7 (320 = 10 sec)
    
    
    #Size of the specctrogram inputs
    spc_rows, spc_cols, channel  = 1920, 128, 1
    
    
    #audio_input = (128, 1292, 2)
    
    #Define the output path
    output = 'D:\\Data_Analytics\\Harmoix\\Dataset_Harmonix\\output'  
    create_folder(output)
    
    ######################## Loading data ###############################
    
    train_generator = AudioGenerator(npz_path=train_db_path, csv_path = train_label_csv, width = chunk_length, batch_size=nb_batch, lable2index= lb_to_idx, shuffle=True)
    nb_train_samples = train_generator.data_length()
    train_steps_per_epoch= int(nb_train_samples // nb_batch)
    
    val_generator = AudioGenerator(npz_path=val_db_path, csv_path = val_label_csv, width = chunk_length, batch_size=nb_batch, lable2index= lb_to_idx, shuffle=True)
    nb_val_samples = val_generator.data_length()
    val_steps_per_epoch= int(nb_val_samples // nb_batch)
    #train_generator = iter(generator)
    '''
    #data_reader = DataReader()
    #train_generator = data_reader.generator_train(nb_batch)
    nb_train_samples = train_generator.data_length()
    #nb_train_samples = data_reader.train_files_count()
    train_steps_per_epoch= int(nb_train_samples // nb_batch)
    
    #val_generator = data_reader.generator_val(nb_batch)
    nb_val_samples = data_reader.val_files_count()
    val_steps_per_epoch= int(nb_val_samples // nb_batch)
    '''
    ######################## Network training Start ###############################
    
    #filepath=output +"AE_AV_model_BOOST_UP2.hdf5"
    filepath=output +"AV_HR_BEST.h5"
    csv_logger = CSVLogger(os.path.join(output + "CSV_Logger" + '-' + 'AE_SlowFast' + str(time.time()) + '.csv'))
    Earl_Stop = EarlyStopping(patience=50)
    #tensorboard = TensorBoard(log_dir="Audio_model_Summary/{}".format(time.time()))
    # if the accuracy does not increase over 10 epochs, we reduce the learning rate by half.
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=0.0001, verbose=1)
    checkpoint = ModelCheckpoint(filepath , monitor='val_mean_iou', verbose=1,save_best_only=True,save_weights_only=False, mode='max',period=1)
    #callbacks_list = [checkpoint,tensorboard,reduce_lr, csv_logger, Earl_Stop]
    callbacks_list = [checkpoint, reduce_lr, csv_logger]
    
    #network = video_audio_net(video_input, audio_input_mel, audio_input_hcqt)
    # visualization params
    metric_list = ['accuracy', 'mse',Mean_IOU, pixel_accuracy]
    
    network = seg_hrnet(nb_batch, spc_rows, spc_cols, channel, nb_classes)    #For HR_net
    '''
    network = deeplabv3(input_tensor=None,
              input_shape=(spc_rows, spc_cols, channel),
              categorical_num=nb_classes,
              backbone='xception',
              OS=16, alpha=1.)
    '''
    network.compile(loss="categorical_crossentropy", optimizer=optimizers.Adam(lr=1e-3),metrics=metric_list)  #Best 
    #network.compile(loss=cce_tversky_loss, optimizer=optimizers.Adam(lr=1e-3),metrics=metric_list) 

    history = network.fit_generator(generator=train_generator, steps_per_epoch=train_steps_per_epoch, epochs=num_epoch,
                    validation_data=val_generator, validation_steps=val_steps_per_epoch, callbacks=callbacks_list)
      
    model_json = network.to_json()
    
    if not os.path.isdir(output):
        os.makedirs(output)
    with open(os.path.join(output, 'AV_HR_model.json'), 'w') as json_file:
        json_file.write(model_json)
    
    network.save_weights(os.path.join(output, 'Rhythm_HR_model.h5'))

    plot_history(history, output)
    save_history(history, output)
    