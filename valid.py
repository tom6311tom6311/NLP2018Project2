import sys
import os
import time
import datetime
import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.callbacks import EarlyStopping
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session


TESTING_DATA_PATH = 'data/TEST_FILE.txt'
GLOVE_EMBEDDER_PATH = 'data/glove.twitter.27B.50d.txt'
MODEL_PATH = str(sys.argv[2])
PREDICTION_PATH = str(sys.argv[3])

CLASS_TO_RELATION = [
  'Cause-Effect(e1,e2)',
  'Cause-Effect(e2,e1)',
  'Instrument-Agency(e1,e2)',
  'Instrument-Agency(e2,e1)',
  'Product-Producer(e1,e2)',
  'Product-Producer(e2,e1)',
  'Content-Container(e1,e2)',
  'Content-Container(e2,e1)',
  'Entity-Origin(e1,e2)',
  'Entity-Origin(e2,e1)',
  'Entity-Destination(e1,e2)',
  'Entity-Destination(e2,e1)',
  'Component-Whole(e1,e2)',
  'Component-Whole(e2,e1)',
  'Member-Collection(e1,e2)',
  'Member-Collection(e2,e1)',
  'Message-Topic(e1,e2)',
  'Message-Topic(e2,e1)',
  'Other',
]

os.environ["CUDA_VISIBLE_DEVICES"] = str(sys.argv[1])


config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
set_session(tf.Session(config=config))

def progress(count, total, suffix=''):
  bar_len = 60
  filled_len = int(round(bar_len * count / float(total)))
  percents = round(100.0 * count / float(total), 1)
  bar = '#' * filled_len + '-' * (bar_len - filled_len)
  sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', suffix))
  sys.stdout.flush()

def grabContext(sentence):
  idx_start = sentence.find('<e1>') + 4
  idx_end = sentence.find('</e2>')
  context = [w for w in sentence[idx_start:idx_end].replace('</e1>', '').replace('<e2>', '').replace(',','').replace('.','').replace('?','').replace('!','').replace(':','').lower().split(' ') if w.isalpha() or '\'' in w]
  if (len(context) == 0):
    print(sentence)
  return context

def load_glove():
  print('loading glove dict...')
  lines = []
  with open(GLOVE_EMBEDDER_PATH, 'r') as glove_file:
    lines = glove_file.readlines()
    glove_file.close()
  glove_dict = {}
  num_lines = len(lines)
  for idx, line in enumerate(lines):
    line_arr = line.replace('\n','').split(' ')
    glove_dict[line_arr[0]] = np.array([float(n) for n in line_arr[1:]])
    if (idx % 100 == 0):
      progress(idx + 1, num_lines)
  print('\n')
  return glove_dict
  

def load_data(path, embed_dict):
  print('preprocessing sentences...')
  lines = []
  with open(path, 'r') as f:
    lines = f.readlines()
    f.close()

  sentence_ids = []
  contexts = []
  for idx, line in enumerate(lines):
    if (line == '\n'):
      break
    [sentence_id, sentence] = line.split('\t')
    context = grabContext(sentence)
    context = [embed_dict[w] for w in context if w in embed_dict]
    if (len(context) == 0):
      context = np.zeros_like(embed_dict['apple'])
    else:
      context = np.average(np.array(context), axis=0)
    sentence_ids.append(sentence_id)
    contexts.append(context)
    progress(idx + 1, len(lines))
  contexts = np.array(contexts)

  return contexts, sentence_ids

if __name__ == '__main__':
  glove_dict = load_glove()
  print('loading testing data...')
  contexts, sentence_ids = load_data(TESTING_DATA_PATH, glove_dict)

  model = load_model(MODEL_PATH)

  print('predicting...')
  predicted = model.predict(contexts)
  predicted = np.argmax(predicted, axis=1)

  with open(PREDICTION_PATH, 'w') as prediction_file:
    for idx, sentence_id in enumerate(sentence_ids):
      prediction_file.write(sentence_id + '\t' + CLASS_TO_RELATION[predicted[idx]] + '\n')
    prediction_file.close()
