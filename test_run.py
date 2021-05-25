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
import pandas as pd
from data.createTestDataSusequentModels import create_test_data
from data.createTestDataFromYangModel import createTestDataFromYangModel
from src.data_provider import datasets_factory
from src.models.model_factory import Model
import src.trainer as trainer
from src.utils import preprocess
import tensorflow as tf
from generateSSIM_graph import generatecsv
import warnings
warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------

FLAGS = tf.app.flags.FLAGS
def main(_,model_number=1,testSample=1,recursive=False,recombination="Max Pooling"):
    del_all_flags(tf.flags.FLAGS)
    FLAGS=tf.app.flags

    FLAGS.DEFINE_string('train_data_paths','data/PSI/model-1/PSI-train.npz','train data paths.')
    FLAGS.DEFINE_string('valid_data_paths','data/PSI/model-1/PSI-valid.npz','validation data paths.')
    FLAGS.DEFINE_string('save_dir','checkpoints/model-1/ckpt-e3d_imme/','dir to store trained net.')  ##test
    FLAGS.DEFINE_string('gen_frm_dir','results/_mnist_e3d_lstm','dir to store result.')
    FLAGS.DEFINE_string('pretrained_model','model.ckpt-80000','.ckpt file to initialize from.')  ##test
    FLAGS.DEFINE_integer('testSample',1,'test Sample')
    FLAGS.DEFINE_integer('frame_start',0,'starting index of the file from 0-141')
    FLAGS.DEFINE_integer('frame_end',9,'ending index of the file from 0-141')
    tf.app.flags.DEFINE_boolean('is_PSI_testing',True,'training or testing')  ##test
    tf.app.flags.DEFINE_boolean('integrate_Yang_Model',False,'get sample from Yang Model')  ##test
    FLAGS.DEFINE_integer('model_no',1,'Model_number')
    FLAGS.DEFINE_string('recombination','Advanced Polar technique','Recombination technique to use')
    tf.app.flags.DEFINE_boolean('is_training',False,'training or testing')
    FLAGS.DEFINE_string('dataset_name','mnist','The name of dataset.')
    FLAGS.DEFINE_integer('input_length',5,'input length.')
    FLAGS.DEFINE_integer('total_length',10,'total input and output length.')
    FLAGS.DEFINE_integer('img_width',64,'input image width.')  ##test
    FLAGS.DEFINE_integer('PSI_patch_size',64,'for recomibing the images')  ##test

    FLAGS.DEFINE_integer('img_channel',1,'number of image channel.')
    FLAGS.DEFINE_integer('patch_size',4,'patch size on one dimension.')
    FLAGS.DEFINE_boolean('reverse_input',False,'reverse the input/outputs during training.')
    FLAGS.DEFINE_string('model_name','e3d_lstm','The name of the architecture.')

    FLAGS.DEFINE_string('num_hidden','64,64,64,64','COMMA separated number of units of e3d lstms.')
    FLAGS.DEFINE_integer('filter_size',5,'filter of a e3d lstm layer.')
    FLAGS.DEFINE_boolean('layer_norm',True,'whether to apply tensor layer norm.')
    FLAGS.DEFINE_boolean('scheduled_sampling',True,'for scheduled sampling')
    FLAGS.DEFINE_integer('sampling_stop_iter',50000,'for scheduled sampling.')
    FLAGS.DEFINE_float('sampling_start_value',1.0,'for scheduled sampling.')
    FLAGS.DEFINE_float('sampling_changing_rate',0.00002,'for scheduled sampling.')

    FLAGS.DEFINE_float('lr',0.001,'learning rate.')
    FLAGS.DEFINE_integer('batch_size',4,'batch size for training.')  ##test
    FLAGS.DEFINE_integer('max_iterations',80000,'max num of steps.')
    FLAGS.DEFINE_integer('display_interval',1,'number of iters showing training loss.')
    FLAGS.DEFINE_integer('test_interval',1000,'number of iters for test.')
    FLAGS.DEFINE_integer('snapshot_interval',1000,'number of iters saving models.')
    FLAGS.DEFINE_integer('num_save_samples',10,'number of sequences to be saved.')
    FLAGS.DEFINE_integer('n_gpu',2,'how many GPUs to distribute the training across.')
    FLAGS.DEFINE_boolean('allow_gpu_growth',True,'allow gpu growth')

    FLAGS=tf.app.flags.FLAGS


    if tf.gfile.Exists(FLAGS.save_dir) and (FLAGS.is_training):
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
        train_wrapper(model)
    elif FLAGS.is_PSI_testing:
      generatecsv(testSample)
      if(FLAGS.integrate_Yang_Model):
        createTestDataFromYangModel(FLAGS.PSI_patch_size,testSample)
      else :
        if recursive==True:
            callconfig(model_number,testSample,recombination)
            PSI_test_wrapper(model)
            del_all_flags(tf.flags.FLAGS)

        elif recursive==False:
            callconfig(model_number,testSample,recombination)
            create_test_data(FLAGS.frame_start,FLAGS.frame_end,FLAGS.valid_data_paths,FLAGS.testSample)
            PSI_test_wrapper(model)
            del_all_flags(tf.flags.FLAGS)
        return True
  # else :
  #     test_wrapper(model)


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


# def test_wrapper(model):
#   model.load(FLAGS.save_dir+FLAGS.pretrained_model)
#   test_input_handle = datasets_factory.data_provider(
#       FLAGS.dataset_name,
#       FLAGS.train_data_paths,
#       FLAGS.valid_data_paths,
#       FLAGS.batch_size * FLAGS.n_gpu,
#       FLAGS.img_width,
#       FLAGS.total_length,
#       is_training=False)
#   trainer.test(model, test_input_handle, FLAGS, 'test_result')

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


def callconfig(model_number,testSample,recombination):
    df=pd.read_csv("model_Details.csv")
    print("Running Model Number: ",model_number)
    model_number=int(model_number)
    start=int(df[df['model_no'] == model_number]['start'].reset_index(drop=True)[0])
    end=int(df[df['model_no'] == model_number]['end'].reset_index(drop=True)[0])
    save_dir=df[df['model_no'] == model_number]['save_dir'].reset_index(drop=True)[0]
    train_data_paths=df[df['model_no'] == model_number]['train_data_paths'].reset_index(drop=True)[0]
    valid_data_paths=df[df['model_no'] == model_number]['valid_data_paths'].reset_index(drop=True)[0]
    gen_frm_dir=df[df['model_no'] == model_number]['gen_frm_dir'].reset_index(drop=True)[0]

    if tf.gfile.Exists(FLAGS.gen_frm_dir):
        tf.gfile.DeleteRecursively(FLAGS.gen_frm_dir)
    tf.gfile.MakeDirs(FLAGS.gen_frm_dir)

    FLAGS.frame_start=start
    FLAGS.frame_end=end
    FLAGS.testSample=testSample
    FLAGS.train_data_paths=train_data_paths
    FLAGS.valid_data_paths=valid_data_paths
    FLAGS.save_dir=save_dir
    FLAGS.gen_frm_dir=gen_frm_dir
    FLAGS.model_no=model_number
    FLAGS.recombination=recombination
    print(FLAGS.frame_start)
    print(FLAGS.frame_end)
    print(FLAGS.testSample)
    print(FLAGS.train_data_paths)
    print(FLAGS.valid_data_paths)
    print(FLAGS.save_dir)
    print(FLAGS.gen_frm_dir)
    print(FLAGS.model_no)

def del_all_flags(FLAGS):
    flags_dict = FLAGS._flags()
    keys_list = [keys for keys in flags_dict]
    for keys in keys_list:
        FLAGS.__delattr__(keys)

#21 and 28
if __name__ == '__main__':
  start_time=time.time()
  tf.app.run()
  print("--- %s seconds ---" % (time.time() - start_time))

