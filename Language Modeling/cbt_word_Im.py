from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import sys
sys.stdout = sys.stderr
import numpy as np
import tensorflow as tf

import reader

flags = tf.flags
logging = tf.logging

flags.DEFINE_string(
    "model", "small",
    "A type of model. Possible options are: small, medium, large.")
flags.DEFINE_string("data_path", None,
                    "Where the training/test data is stored.")
flags.DEFINE_string("save_path", None,
                    "Model output directory.")
flags.DEFINE_bool("sample_mode", False,
                  "Must have trained model ready. Only does sampling")
flags.DEFINE_string("file_prefix", "cbt",
                  "will be looking for file_prefix.train.txt, file_prefix.test.txt and file_prefix.valid.txt in data_path")
flags.DEFINE_string("seed_for_sample", "we have no",
                  "supply seeding phrase here. it must only contain words from vocabluary")

FLAGS = flags.FLAGS


class CBTModel(object):

  def __init__(self, is_training, config):

    self.batch_size = batch_size = config.batch_size
    self.num_steps = num_steps = config.num_steps # Time Step
    size = config.hidden_size # Each hidden layer size
    vocab_size = config.vocab_size

    self._input_data = tf.placeholder(tf.int32, [batch_size, num_steps])
    self._targets = tf.placeholder(tf.int32, [batch_size, num_steps])

    #Build the LSTM Network
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(size, forget_bias=0.0, state_is_tuple=True)
    if is_training and config.keep_prob < 1:
      lstm_cell = tf.contrib.rnn.DropoutWrapper(
          lstm_cell, output_keep_prob=config.keep_prob)
    cell = tf.contrib.rnn.MultiRNNCell([lstm_cell] * config.num_layers, state_is_tuple=True)

    #Setting the initial state
    self._initial_state = cell.zero_state(batch_size, tf.float32)
    #print(self._initial_state)

    #The word ids are embedded into a dense representation before feeding into the LSTM
    with tf.device("/cpu:0"):
      embedding = tf.get_variable(
          "embedding", [vocab_size, size], dtype=tf.float32)
      inputs = tf.nn.embedding_lookup(embedding, self._input_data)

    #Applying dropout to reduce overfitting
    if is_training and config.keep_prob < 1:
      inputs = tf.nn.dropout(inputs, config.keep_prob)


    outputs = []
    state = self._initial_state

    #For each time step t - get the value of the cell output and the state
    with tf.variable_scope("RNN"):
      for time_step in range(num_steps):
        if time_step > 0: tf.get_variable_scope().reuse_variables()
        (cell_output, state) = cell(inputs[:, time_step, :], state)
        outputs.append(cell_output)

    output = tf.reshape(tf.concat(axis=1, values=outputs), [-1, size])

    #Vriable for storing the weights. This is of the dimension [hidden_layer_size, overall vocabulary]
    #as you learn for each word in the dataset
    softmax_w = tf.get_variable(
        "softmax_w", [size, vocab_size], dtype=tf.float32)

    #The bias value
    softmax_b = tf.get_variable("softmax_b", [vocab_size], dtype=tf.float32)

    #Calculating the logits - wx+b
    logits = tf.matmul(output, softmax_w) + softmax_b
    #This draws a single sample from the logits, this is for generating samples
    self.sample = tf.multinomial(logits, 1)

    #Calculate the loss using contrib sequence loss by example
    loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
        [logits],
        [tf.reshape(self._targets, [-1])],
        [tf.ones([batch_size * num_steps], dtype=tf.float32)])

    #Averaging across batched and updating the cost
    self._cost = cost = tf.reduce_sum(loss) / batch_size
    self._final_state = state
    if not is_training:
      return

    self._lr = tf.Variable(0.0, trainable=False)
    tvars = tf.trainable_variables()
    # Applying gradient clipping so as to make sure that the gradients do not explode
    grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars),
                                      config.max_grad_norm)

    #Two optimizers used to perform gradient descent
    if config.optimizer == 'AdamOptimizer':
      optimizer = tf.train.AdamOptimizer()
    else:
      optimizer = tf.train.GradientDescentOptimizer(self._lr)

    # This is one part of minimizing the loss - applies gradients to the variables - the training process
    self._train_op = optimizer.apply_gradients(
        zip(grads, tvars),
        global_step=tf.contrib.framework.get_or_create_global_step())

    #Update the learning rate
    self._new_lr = tf.placeholder(
        tf.float32, shape=[], name="new_learning_rate")
    self._lr_update = tf.assign(self._lr, self._new_lr)

#This function updates the learning rate after each epoch
  def assign_lr(self, session, lr_value):
    session.run(self._lr_update, feed_dict={self._new_lr: lr_value})

  @property
  def input_data(self):
    return self._input_data

  @property
  def targets(self):
    return self._targets


  @property
  def initial_state(self):
    return self._initial_state

  @property
  def cost(self):
    return self._cost

  @property
  def final_state(self):
    return self._final_state

  @property
  def lr(self):
    return self._lr

  @property
  def train_op(self):
    return self._train_op

'''These are the configurations for the model
Overall there are 3 configurations that you can play around with.
The small config takes relatively small time compared to the medium config
Each variable of the config is explained below'''
class SmallConfig(object):
  """Small config."""
  optimizer = 'AdamOptimizer' #Optimizer used for gradient descent
  init_scale = 0.1
  learning_rate = 1.0 #The learning rate for gradient descent
  max_grad_norm = 5 #The gradient clipping parameter - this is the minimum value for any gradient
  num_layers = 2 #Number of layers in the RNN
  num_steps = 20 #Time step for each cell
  hidden_size = 200 #Number of lstm cells in each layer
  max_epoch = 4
  max_max_epoch = 6
  keep_prob = 1.0 #Probability that each element is kept during dropout
  lr_decay = 0.5
  batch_size = 20
  vocab_size = 14244


class MediumConfig(object):
  """Medium config."""
  optimizer = 'GradientDescentOptimizer'
  init_scale = 0.05
  learning_rate = 1.0
  max_grad_norm = 5
  num_layers = 2
  num_steps = 35
  hidden_size = 650
  max_epoch = 6
  max_max_epoch = 10
  keep_prob = 0.5
  lr_decay = 0.8
  batch_size = 20
  vocab_size = 14244

class LargeConfig(object):
  """Large config."""
  optimizer = 'GradientDescentOptimizer'
  init_scale = 0.04
  learning_rate = 1.0
  max_grad_norm = 10
  num_layers = 3
  num_steps = 35
  hidden_size = 700
  max_epoch = 8
  max_max_epoch = 10
  keep_prob = 0.35
  lr_decay = 1 / 1.15
  batch_size = 20
  vocab_size = 14244

'''This method generates samples of length num_samples provided the seed data'''
def do_sample(session, model, data, num_samples):

  samples = []
  state = session.run(model.initial_state)
  fetches = [model.final_state, model.sample]
  sample = None
  for x in data:
    feed_dict = {}
    feed_dict[model.input_data] = [[x]]
    #print(model.initial_state)
    for layer_num, (c, h) in enumerate(model.initial_state):
      feed_dict[c] = state[layer_num].c
      feed_dict[h] = state[layer_num].h

    state, sample = session.run(fetches, feed_dict)
  if sample is not None:
    samples.append(sample[0][0])
  else:
    samples.append(0)
  k = 1
  while k < num_samples:
    feed_dict = {}
    feed_dict[model.input_data] = [[samples[-1]]]
    for layer_num, (c, h) in enumerate(model.initial_state):
      feed_dict[c] = state[layer_num].c
      feed_dict[h] = state[layer_num].h
    state, sample = session.run(fetches, feed_dict)
    samples.append(sample[0][0])
    k += 1
  return samples


def run_epoch(session, model, data, is_train=False, verbose=False):
  """Runs the model on the given data."""
  epoch_size = ((len(data) // model.batch_size) - 1) // model.num_steps
  start_time = time.time()
  costs = 0.0
  iters = 0
  state = session.run(model.initial_state)

  for step, (x, y) in enumerate(reader.cbt_iterator(data, model.batch_size, model.num_steps)):
    if is_train:
      fetches = [model.cost, model.final_state, model.train_op]
    else:
      fetches = [model.cost, model.final_state]
    feed_dict = {}
    feed_dict[model.input_data] = x
    feed_dict[model.targets] = y
    for layer_num, (c, h) in enumerate(model.initial_state):
      feed_dict[c] = state[layer_num].c
      feed_dict[h] = state[layer_num].h

    if is_train:
      cost, state, _ = session.run(fetches, feed_dict)
    else:
      cost, state = session.run(fetches, feed_dict)

    costs += cost
    iters += model.num_steps

    if verbose and step % (epoch_size // 10) == 10:
      print("%.3f perplexity: %.3f speed: %.0f wps" %
            (step * 1.0 / epoch_size, np.exp(costs / iters),
             iters * model.batch_size / (time.time() - start_time)))

  return np.exp(costs / iters)

#This function prints the generated samples
def pretty_print(items, id2word):
  return ' '.join([id2word[x] for x in items])

#This function returns the config depending on the input config specified
def get_config():
  if FLAGS.model == "small":
    return SmallConfig()
  elif FLAGS.model == "medium":
    return MediumConfig()
  elif FLAGS.model == "large":
    return LargeConfig()
  elif FLAGS.model == "test":
    return TestConfig()
  else:
    raise ValueError("Invalid model: %s", FLAGS.model)

def main(_):
  start_time = time.time()

#If the data path is invalid raise an error
  if not FLAGS.data_path:
    raise ValueError("Must set --data_path to CBT data directory")

  #Read Data
  raw_data = reader.cbt_raw_data(FLAGS.data_path, FLAGS.file_prefix)
  train_data, valid_data, test_data, word_to_id, id_2_word = raw_data
  vocab_size = len(word_to_id)
  print('Distinct terms: %d' % vocab_size)
  config = get_config()
  config.vocab_size = config.vocab_size if config.vocab_size < vocab_size else vocab_size
  eval_config = get_config()
  eval_config.vocab_size = eval_config.vocab_size if eval_config.vocab_size < vocab_size else vocab_size
  eval_config.batch_size = 1
  eval_config.num_steps = 1

  seed_for_sample = FLAGS.seed_for_sample.split()

  with tf.Graph().as_default():
    initializer = tf.random_uniform_initializer(-config.init_scale,
                                                config.init_scale)
    with tf.name_scope("Train"):

      with tf.variable_scope("Model", reuse=None, initializer=initializer):
        m = CBTModel(is_training=True, config=config)
        tf.summary.scalar("Training Loss", m.cost)
        tf.summary.scalar("Learning Rate", m.lr)

    with tf.name_scope("Valid"):
      with tf.variable_scope("Model", reuse=True, initializer=initializer):
        mvalid = CBTModel(is_training=False, config=config)
        tf.summary.scalar("Validation Loss", mvalid.cost)

    with tf.name_scope("Test"):
      with tf.variable_scope("Model", reuse=True, initializer=initializer):
        mtest = CBTModel(is_training=False, config=eval_config)

    saver = tf.train.Saver(name='saver', write_version=tf.train.SaverDef.V2)
    sv = tf.train.Supervisor(logdir=FLAGS.save_path, save_model_secs=0, save_summaries_secs=0, saver=saver)

    old_valid_perplexity = 10000000000.0

    with sv.managed_session() as session:
      #This prints the samples
      if FLAGS.sample_mode:
        while True:
          inpt = input("Enter your sample prefix: ")
          cnt = int(input("Sample size: "))
          seed_for_sample = inpt.split()
          print("Seed: %s" % pretty_print([word_to_id[x] for x in seed_for_sample], id_2_word))
          print("Sample: %s" % pretty_print(do_sample(session, mtest, [word_to_id[word] for word in seed_for_sample],
                                                      cnt), id_2_word))

      for i in range(config.max_max_epoch):

        print("Seed: %s" % pretty_print([word_to_id[x] for x in seed_for_sample], id_2_word))
        print("Sample: %s" % pretty_print(do_sample(session, mtest, [word_to_id[word] for word in seed_for_sample],
                                                    max(5 * (len(seed_for_sample) + 1), 10)), id_2_word))

        lr_decay = config.lr_decay ** max(i - config.max_epoch, 0.0)
        m.assign_lr(session, config.learning_rate * lr_decay)
        print("Epoch: %d Learning rate: %.3f" % (i + 1, session.run(m.lr)))
        train_perplexity = run_epoch(session, m, train_data, is_train=True, verbose=True)
        print("Epoch: %d Train Perplexity: %.3f" % (i + 1, train_perplexity))
        print(time.time() - start_time)
        valid_perplexity = run_epoch(session, mvalid, valid_data)
        print("Epoch: %d Valid Perplexity: %.3f" % (i + 1, valid_perplexity))
        #Saving the session so that it can be used later
        if valid_perplexity < old_valid_perplexity:
          old_valid_perplexity = valid_perplexity
          sv.saver.save(session, FLAGS.save_path, i)
        elif valid_perplexity >= 1.3*old_valid_perplexity:
          if len(sv.saver.last_checkpoints)>0:
            sv.saver.restore(session, sv.saver.last_checkpoints[-1])
          break
        else:
          if len(sv.saver.last_checkpoints)>0:
            sv.saver.restore(session, sv.saver.last_checkpoints[-1])
          lr_decay *=0.5

      test_perplexity = run_epoch(session, mtest, test_data)
      print("Test Perplexity: %.3f" % test_perplexity)
      print(time.time() - start_time)

if __name__ == "__main__":
  tf.app.run()
