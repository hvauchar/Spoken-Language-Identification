#!/usr/bin/env python
# coding: utf-8

# In[1]:


import webrtcvad
import pyaudio
import wave
import os, sys
import librosa
import numpy as np
import matplotlib.pyplot as plt
import librosa.display
import tensorflow as tf
with tf.gfile.FastGFile("retrained_graphC.pb", 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name='')


audio_path = input('Paste the path to the audio to Classify hear ')

with tf.Session() as sess:
    # Feed the image_data as input to the graph and get first prediction
    softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    image_path = 'tmp.png'
    try:
        y, sr = librosa.load(audio_path)
        y = librosa.resample(y, sr, 22050)
        try:
            l = librosa.get_duration(y=y, sr=sr)
            if(l >= 0.5 and l <= 10):
                S = librosa.feature.melspectrogram(y, sr=sr, n_mels=128)
                log_S = librosa.core.amplitude_to_db(S, ref=np.max)
                fig = plt.figure(figsize=(12,4))
                ax = plt.Axes(fig, [0., 0., 1., 1.])
                ax.set_axis_off()
                fig.add_axes(ax)
                librosa.display.specshow(log_S, sr=sr, x_axis='time', y_axis='mel')
                plt.savefig(image_path)
                plt.close()
            elif(l > 10 and l <15):
                y1 = y[:int(len(y)*0.75)]
                S = librosa.feature.melspectrogram(y1, sr=sr, n_mels=128)
                log_S = librosa.core.amplitude_to_db(S, ref=np.max)
                fig = plt.figure(figsize=(12,4))
                ax = plt.Axes(fig, [0., 0., 1., 1.])
                ax.set_axis_off()
                fig.add_axes(ax)
                librosa.display.specshow(log_S, sr=sr, x_axis='time', y_axis='mel')
                plt.savefig(image_path)
                plt.close()
        except Exception as e:
                print(e)
    except Exception as e:
            print(e)
    # Read in the image_data
    image_data = tf.gfile.FastGFile(image_path, 'rb').read()

    # Loads label file, strips off carriage return
    label_lines = [line.rstrip() for line
                       in tf.gfile.GFile("retrained_labels.txt")]

    predictions = sess.run(softmax_tensor,              {'DecodeJpeg/contents:0': image_data})

    # Sort to show labels of first prediction in order of confidence
    top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
    print('%s (score = %.5f)' % (label_lines[top_k[0]], predictions[0][top_k[0]]))
    plt.bar(label_lines,predictions[0])
    plt.show()
    print(predictions[0],label_lines)


# In[ ]:


stream.stop_stream()
stream.close()
audio.terminate()

