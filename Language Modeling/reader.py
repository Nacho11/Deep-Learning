from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os

import numpy as np
import tensorflow as tf

'''This function reads the words from the given file, convert them
to lowercase and replaces the new line character with an end of sentence special word <eos>'''
def _read_words(filename):
  with tf.gfile.GFile(filename, "r") as f:
    return f.read().lower().replace("\n", "<eos>").split()

'''This function takes in the file and creates builds the vocabulary
It returs 2 dictionaries one word_to_id which consists of a mapping from each distinct word to a unique id
id_to_word this dictonary consists of the reverse mapping i.e the id of the word to the word'''
def _build_vocab(filename):
  data = _read_words(filename)

  new_data = []
  for each_word in data:
      if '$' in each_word or '-' in each_word or '\'' in each_word or '!' in each_word or '_' in each_word:
          continue
      else:
          new_data.append(each_word)

  counter = collections.Counter(new_data)
  count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

  words, _ = list(zip(*count_pairs))
  word_to_id = dict(zip(words, range(len(words))))
  id_to_word = dict((v, k) for k, v in word_to_id.items())
  #print(word_to_id)
  return word_to_id, id_to_word


'''This function takes in the word_to_id dictonary and the filename and returns
the list of numbers which represent the words so if there is a sentence like -
Welcome to this wonderful world! if the unique ids are 8900, 76, 800, 9000, 8979
This sentence would be represented as [8900, 76, 800, 9000, 8979]'''
def _file_to_word_ids(filename, word_to_id):
  data = _read_words(filename)
  return [word_to_id[word] for word in data if word in word_to_id]


'''This function converts the text files into their numerical representations'''
def cbt_raw_data(data_path=None, prefix="cbt"):
  train_path = os.path.join(data_path, prefix + "_train_1.txt")
  valid_path = os.path.join(data_path, prefix + "_valid.txt")
  test_path = os.path.join(data_path, prefix + "_test.txt")

  word_to_id, id_2_word = _build_vocab(train_path)
  train_data = _file_to_word_ids(train_path, word_to_id)
  valid_data = _file_to_word_ids(valid_path, word_to_id)
  test_data = _file_to_word_ids(test_path, word_to_id)
  return train_data, valid_data, test_data, word_to_id, id_2_word

''' This function goes through the data file and creates batches for input
The input that needs to be given is sliced in the way of 2 rows of batches each row having
5 words as the num_steps, which is time_step into a LSTM is 5'''
def cbt_iterator(raw_data, batch_size, num_steps):
  raw_data = np.array(raw_data, dtype=np.int32)

  data_len = len(raw_data) #The overall length of data i.e the nmber of words
  batch_len = data_len // batch_size #The length of each batch
  data = np.zeros([batch_size, batch_len], dtype=np.int32)
  for i in range(batch_size):
    data[i] = raw_data[batch_len * i:batch_len * (i + 1)]

  epoch_size = (batch_len - 1) // num_steps

  if epoch_size == 0:
    raise ValueError("epoch_size == 0, decrease batch_size or num_steps")

  for i in range(epoch_size):
    x = data[:, i*num_steps:(i+1)*num_steps]
    y = data[:, i*num_steps+1:(i+1)*num_steps+1]
    yield (x, y)
