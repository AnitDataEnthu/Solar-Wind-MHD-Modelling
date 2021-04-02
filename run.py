# Copyright 2019 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Main function to run the code."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import numpy as np
import time
from data.createtestdata import createtestsample
from data.createTestDataFromYangModel import createTestDataFromYangModel
from data.createSubsequentFrames import create_train_data
from src.data_provider import datasets_factory
from src.models.model_factory import Model
import src.trainer as trainer
import tensorflow as tf
from src.utils import preprocess
import pandas as pd
# -----------------------------------------------------------------------------
model_number=1#int(float(input("Which subsequent model to train")))
print("Preparing training for model_number",model_number)
all_model=[]
model_no=1
# for i in range(0,135,5):
#     model_dict={}
#     start=i
#     end=i + 91
#     save_dir='checkpoints/model-' + str(model_no) + '/ckpt-e3d_imme/'
#     train_data_paths='data/PSI/model-' + str(model_no) + '/PSI-train.npz'
#     valid_data_paths='data/PSI/model-' + str(model_no) + '/PSI-valid.npz'
#     gen_frm_dir='results/_mnist_e3d_lstm/model-' + str(model_no) + '/'

#     model_dict['model_no']=model_no
#     model_dict['start']=start
#     model_dict['end']=end
#     model_dict['save_dir']=save_dir
#     model_dict['train_data_paths']=train_data_paths
#     model_dict['valid_data_paths']=valid_data_paths
#     model_dict['gen_frm_dir']=gen_frm_dir
#     all_model.append(model_dict)
#     model_no+=1

# df=pd.DataFrame(all_model)
# df.to_csv("model_Details.csv")
df=pd.read_csv("model_Details.csv")

start=int(df[df['model_no']==model_number]['start'][0])
end=int(df[df['model_no']==model_number]['end'][0])
save_dir=df[df['model_no']==model_number]['save_dir'][0]
train_data_paths=df[df['model_no']==model_number]['train_data_paths'][0]
valid_data_paths=df[df['model_no']==model_number]['valid_data_paths'][0]
gen_frm_dir=df[df['model_no']==model_number]['gen_frm_dir'][0]

FLAGS = tf.app.flags
if not tf.gfile.Exists(save_dir) :
    tf.gfile.MakeDirs(save_dir)
if not tf.gfile.Exists(train_data_paths) :
    tf.gfile.MakeDirs(train_data_paths)
if not tf.gfile.Exists(valid_data_paths) :
    tf.gfile.MakeDirs(valid_data_paths)
if not tf.gfile.Exists(gen_frm_dir) :
    tf.gfile.MakeDirs(gen_frm_dir)


FLAGS.DEFINE_string('train_data_paths', train_data_paths, 'train data paths.')
FLAGS.DEFINE_string('valid_data_paths', valid_data_paths, 'validation data paths.')
FLAGS.DEFINE_string('save_dir', save_dir, 'dir to store trained net.')           ##test
FLAGS.DEFINE_string('gen_frm_dir', gen_frm_dir, 'dir to store result.')  
FLAGS.DEFINE_integer('frame_start', start, 'starting index of the file from 0-141') 
FLAGS.DEFINE_integer('frame_end', end, 'ending index of the file from 0-141') 
FLAGS.DEFINE_boolean('is_training', True, 'training or testing',)
FLAGS.DEFINE_boolean('is_PSI_testing', False, 'training or testing')                      ##test
FLAGS.DEFINE_boolean('integrate_Yang_Model', False , 'get sample from Yang Model')         ##test
FLAGS.DEFINE_string('dataset_name', 'mnist', 'The name of dataset.')
FLAGS.DEFINE_integer('input_length', 5, 'input length.')
FLAGS.DEFINE_integer('total_length', 10, 'total input and output length.')
FLAGS.DEFINE_integer('img_width', 64, 'input image width.')                                      ##test
FLAGS.DEFINE_integer('PSI_patch_size', 64, 'for recomibing the images')                          ##test
FLAGS.DEFINE_integer('testSample',12, 'test Sample')
FLAGS.DEFINE_integer('img_channel', 1, 'number of image channel.')
FLAGS.DEFINE_integer('patch_size', 4, 'patch size on one dimension.')
FLAGS.DEFINE_boolean('reverse_input', False,
                     'reverse the input/outputs during training.')
FLAGS.DEFINE_string('model_name', 'e3d_lstm', 'The name of the architecture.')
FLAGS.DEFINE_string('pretrained_model', '', '.ckpt file to initialize from.')     ##test
FLAGS.DEFINE_string('num_hidden', '64,64,64,64',
                    'COMMA separated number of units of e3d lstms.')
FLAGS.DEFINE_integer('filter_size', 5, 'filter of a e3d lstm layer.')
FLAGS.DEFINE_boolean('layer_norm', True, 'whether to apply tensor layer norm.')
FLAGS.DEFINE_boolean('scheduled_sampling', True, 'for scheduled sampling')
FLAGS.DEFINE_integer('sampling_stop_iter', 50000, 'for scheduled sampling.')
FLAGS.DEFINE_float('sampling_start_value', 1.0, 'for scheduled sampling.')
FLAGS.DEFINE_float('sampling_changing_rate', 0.00002, 'for scheduled sampling.')

FLAGS.DEFINE_float('lr', 0.001, 'learning rate.')
FLAGS.DEFINE_integer('batch_size', 4, 'batch size for training.')                                  ##test
FLAGS.DEFINE_integer('max_iterations', 80000, 'max num of steps.')
FLAGS.DEFINE_integer('display_interval', 1,
                     'number of iters showing training loss.')
FLAGS.DEFINE_integer('test_interval', 1000, 'number of iters for test.')
FLAGS.DEFINE_integer('snapshot_interval', 1000,
                     'number of iters saving models.')
FLAGS.DEFINE_integer('num_save_samples', 10, 'number of sequences to be saved.')
FLAGS.DEFINE_integer('n_gpu', 2,
                     'how many GPUs to distribute the training across.')
FLAGS.DEFINE_boolean('allow_gpu_growth', True, 'allow gpu growth')
def main(_):
  """Main function."""

  if tf.gfile.Exists(FLAGS.save_dir) and (FLAGS.is_training) :
    tf.gfile.DeleteRecursively(FLAGS.save_dir)
  tf.gfile.MakeDirs(FLAGS.save_dir)
  if tf.gfile.Exists(FLAGS.gen_frm_dir):
    tf.gfile.DeleteRecursively(FLAGS.gen_frm_dir)
  tf.gfile.MakeDirs(FLAGS.gen_frm_dir)

  gpu_list = np.asarray(
      os.environ.get('CUDA_VISIBLE_DEVICES', '-1').split(','), dtype=np.int32)
  FLAGS.n_gpu = len(gpu_list)
  print('Initializing models')

  model = Model(FLAGS)

  if FLAGS.is_training:
      create_train_data(FLAGS.frame_Start,FLAGS.frame_End,"/data/PSI/model-1/PSI-train.npz","/data/PSI/model-1/PSI-valid.npz")
      train_wrapper(model)
  elif FLAGS.is_PSI_testing:
      if(FLAGS.integrate_Yang_Model):
        createTestDataFromYangModel(FLAGS.PSI_patch_size,FLAGS.testSample)
      else :
        createtestsample(FLAGS.PSI_patch_size,FLAGS.testSample)
      PSI_test_wrapper(model)
  else :
      test_wrapper(model)


def schedule_sampling(eta, itr):
  """Gets schedule sampling parameters for training."""
  zeros = np.zeros(
      (FLAGS.batch_size, FLAGS.total_length - FLAGS.input_length - 1,
       FLAGS.img_width // FLAGS.patch_size, FLAGS.img_width // FLAGS.patch_size,
       FLAGS.patch_size**2 * FLAGS.img_channel))
  if not FLAGS.scheduled_sampling:
    return 0.0, zeros

  if itr < FLAGS.sampling_stop_iter:
    eta -= FLAGS.sampling_changing_rate
  else:
    eta = 0.0
  random_flip = np.random.random_sample(
      (FLAGS.batch_size, FLAGS.total_length - FLAGS.input_length - 1))
  true_token = (random_flip < eta)
  ones = np.ones(
      (FLAGS.img_width // FLAGS.patch_size, FLAGS.img_width // FLAGS.patch_size,
       FLAGS.patch_size**2 * FLAGS.img_channel))
  zeros = np.zeros(
      (FLAGS.img_width // FLAGS.patch_size, FLAGS.img_width // FLAGS.patch_size,
       FLAGS.patch_size**2 * FLAGS.img_channel))
  real_input_flag = []
  for i in range(FLAGS.batch_size):
    for j in range(FLAGS.total_length - FLAGS.input_length - 1):
      if true_token[i, j]:
        real_input_flag.append(ones)
      else:
        real_input_flag.append(zeros)
  real_input_flag = np.array(real_input_flag)
  real_input_flag = np.reshape(
      real_input_flag,
      (FLAGS.batch_size, FLAGS.total_length - FLAGS.input_length - 1,
       FLAGS.img_width // FLAGS.patch_size, FLAGS.img_width // FLAGS.patch_size,
       FLAGS.patch_size**2 * FLAGS.img_channel))
  return eta, real_input_flag


def train_wrapper(model):
  """Wrapping function to train the model."""
  if FLAGS.pretrained_model:
    model.load(FLAGS.pretrained_model)
  # load data
  train_input_handle, test_input_handle = datasets_factory.data_provider(
      FLAGS.dataset_name,
      FLAGS.train_data_paths,
      FLAGS.valid_data_paths,
      FLAGS.batch_size * FLAGS.n_gpu,
      FLAGS.img_width,
      seq_length=FLAGS.total_length,
      is_training=True)

  eta = FLAGS.sampling_start_value

  for itr in range(1, FLAGS.max_iterations + 1):
    if train_input_handle.no_batch_left():
      train_input_handle.begin(do_shuffle=True)
    ims = train_input_handle.get_batch()
    if FLAGS.dataset_name == 'penn':
      ims = ims['frame']
    ims = preprocess.reshape_patch(ims, FLAGS.patch_size)

    eta, real_input_flag = schedule_sampling(eta, itr)

    trainer.train(model, ims, real_input_flag, FLAGS, itr)

    if itr % FLAGS.snapshot_interval == 0:
      model.save(itr)

    if itr % FLAGS.test_interval == 0:
      trainer.test(model, test_input_handle, FLAGS, itr)

    train_input_handle.next()


def test_wrapper(model):
  model.load(FLAGS.save_dir+FLAGS.pretrained_model)
  test_input_handle = datasets_factory.data_provider(
      FLAGS.dataset_name,
      FLAGS.train_data_paths,
      FLAGS.valid_data_paths,
      FLAGS.batch_size * FLAGS.n_gpu,
      FLAGS.img_width,
      FLAGS.total_length,
      is_training=False)
  trainer.test(model, test_input_handle, FLAGS, 'test_result')

def PSI_test_wrapper(model):
    model.load(FLAGS.save_dir + FLAGS.pretrained_model)
    test_input_handle=datasets_factory.data_provider(
        FLAGS.dataset_name,
        FLAGS.train_data_paths,
        FLAGS.valid_data_paths,
        FLAGS.batch_size * FLAGS.n_gpu,
        FLAGS.img_width,
        FLAGS.total_length,
        is_training=False)
    trainer.PSI_test(model,test_input_handle,FLAGS,'PSI_test_result',FLAGS.PSI_patch_size)

if __name__ == '__main__':
  start_time=time.time()
  tf.app.run()
  print("--- %s seconds ---" % (time.time() - start_time))
