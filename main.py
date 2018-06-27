import sys
import os
import time
import datetime
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

MODEL_PATH = str(sys.argv[2])
TRAINING_DATA_PATH = 'data/TRAIN_FILE.txt'
GLOVE_EMBEDDER_PATH = 'data/glove.42B.300d.txt'
CLASS_OCCUR_THRES = 20

MLP_DIMS = [int(dim) for dim in sys.argv[3].split('_')]


RELATIONS_CLASS_MAP = {
  'Cause-Effect(e1,e2)': 0,
  'Cause-Effect(e2,e1)': 1,
  'Instrument-Agency(e1,e2)': 2,
  'Instrument-Agency(e2,e1)': 3,
  'Product-Producer(e1,e2)': 4,
  'Product-Producer(e2,e1)': 5,
  'Content-Container(e1,e2)': 6,
  'Content-Container(e2,e1)': 7,
  'Entity-Origin(e1,e2)': 8,
  'Entity-Origin(e2,e1)': 9,
  'Entity-Destination(e1,e2)': 10,
  'Entity-Destination(e2,e1)': 11,
  'Component-Whole(e1,e2)': 12,
  'Component-Whole(e2,e1)': 13,
  'Member-Collection(e1,e2)': 14,
  'Member-Collection(e2,e1)': 15,
  'Message-Topic(e1,e2)': 16,
  'Message-Topic(e2,e1)': 17,
  'Other': 18
}

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

  print('Grab contexts...')
  raw_contexts = []
  raw_labels = []
  for idx, line in enumerate(lines):
    if (idx % 4 == 0):
      raw_context = grabContext(line)
      raw_contexts.append(raw_context)
    elif (idx % 4 == 1):
      raw_labels.append(RELATIONS_CLASS_MAP[line[:-1]])
    progress(idx + 1, len(lines))
  raw_labels = np.array(raw_labels)

  print('\nComputing class occurrence dictionary...')
  class_occur_dict = {}
  for sentence_id, raw_context in enumerate(raw_contexts):
    for w in raw_context:
      if w not in class_occur_dict:
        class_occur_dict[w] = [raw_labels[sentence_id]]
      elif raw_labels[sentence_id] not in class_occur_dict[w]:
        class_occur_dict[w].append(raw_labels[sentence_id])
      progress(sentence_id + 1, len(raw_contexts))

  labels = np.eye(len(RELATIONS_CLASS_MAP))[raw_labels]

  print('\nEmbedding')
  contexts = []
  for sentence_id, raw_context in enumerate(raw_contexts):
    context = [embed_dict[w] for w in raw_context if w in embed_dict and len(class_occur_dict[w]) <= CLASS_OCCUR_THRES]
    if (len(context) == 0):
      context = np.zeros_like(embed_dict['apple'])
    else:
      context = np.average(np.array(context), axis=0)
    contexts.append(context)
    progress(sentence_id + 1, len(raw_contexts))
  print('\n')
  contexts = np.array(contexts)

  p = np.random.permutation(len(contexts))
  contexts = contexts[p]
  labels = labels[p]

  return contexts, labels

if __name__ == '__main__':
  glove_dict = load_glove()
  print('loading training data...')
  contexts, labels = load_data(TRAINING_DATA_PATH, glove_dict)

  model = Sequential()
  model.add(Dense(MLP_DIMS[0], activation='relu', input_dim=glove_dict['apple'].shape[0]))
  for dim in MLP_DIMS[1:]:
    model.add(Dense(dim, activation='relu'))
    model.add(Dropout(0.1))
  model.add(Dense(len(RELATIONS_CLASS_MAP), activation='softmax'))
  model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
  model.summary()

  print('training...')
  model.fit(
    contexts,
    labels,
    epochs=1000,
    batch_size=64,
    validation_split=0.1,
    callbacks=[EarlyStopping(monitor='loss', patience=5)])
  model.save(MODEL_PATH, overwrite=True, include_optimizer=False)
