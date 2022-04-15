import tensorflow as tf

flags = tf.app.flags

############################
#    hyper parameters      #
############################

# For separate margin loss
flags.DEFINE_float('m_plus', 0.9, 'the parameter of m plus')
flags.DEFINE_float('m_minus', 0.1, 'the parameter of m minus')
flags.DEFINE_float('lambda_val', 0.5, 'down weight of the loss for absent digit classes')

# for training
flags.DEFINE_integer('batch_size', 8, 'batch size')
flags.DEFINE_integer('batch_size_X', 5, 'batch size')
flags.DEFINE_integer('batch_size_Y', 8, 'batch size')
flags.DEFINE_integer('batch_size_Z', 8, 'batch size')
flags.DEFINE_integer('epoch', 16, 'epoch') # 50 to 1
flags.DEFINE_integer('restore_epoch', 15, 'epoch') # 50 to 1
flags.DEFINE_integer('computer', 0, '0 2060, 1  1080ti,2 2080ti') #
flags.DEFINE_integer('year', 0, '0 2015, 1  2018,2 2019') #
flags.DEFINE_integer('preprocess', 0, '0 None, 1 sitk_N4 , 2 nipype.interfaces.ants_N4') #
flags.DEFINE_boolean('enhance', False, 'image enhance')
flags.DEFINE_boolean('is_training', True, 'trsain or predict phase')
flags.DEFINE_boolean('is_testing', False, 'train or predict phase')
flags.DEFINE_boolean('both_hgg_lgg', False, 'if false, only hgg') # 50 to 1
flags.DEFINE_boolean('summary', True, 'if false, no summary') # 50 to 1
flags.DEFINE_integer('validation_set_num', 30, 'number of validation') # 50 to 1
flags.DEFINE_integer('epoch_to_fuse', 36, 'number of epoch to complete fuse loss') # 50 to 1
flags.DEFINE_integer('resblock_layer_num', 3, 'number of iterations in resblock')
flags.DEFINE_integer('system', 0, '0 window, 1 linux , 2 nipype.interfaces.ants_N4') # 50 to 1
flags.DEFINE_integer('stage', 1, '1 [0,1] , 2 [01234]') # 50 to 1
flags.DEFINE_integer('iter_routing', 3, 'number of iterations in routing algorithm')
flags.DEFINE_boolean('mask_with_y', True, 'use the true label to mask out target capsule or not')

flags.DEFINE_float('stddev', 0.01, 'stddev for W initializer')
flags.DEFINE_float('regularization_scale', 0.392, 'regularization coefficient for reconstruction loss, default to 0.0005*784=0.392')

############################
#   environment setting    #
############################
flags.DEFINE_string('dataset', 'mnist', 'The name of dataset [mnist, fashion-mnist')
flags.DEFINE_string('dimensional', 'X', 'dimensional')
flags.DEFINE_integer('num_threads', 8, 'number of threads of enqueueing examples')
flags.DEFINE_string('logdir', 'logdir', 'logs directory')
flags.DEFINE_integer('train_sum_freq', 100, 'the frequency of saving train summary(step)')
flags.DEFINE_integer('val_sum_freq', 500, 'the frequency of saving valuation summary(step)')
flags.DEFINE_integer('save_freq', 3, 'the frequency of saving model(epoch)')
flags.DEFINE_string('data_results', 'data_results', 'path for saving results')

############################
#   distributed setting    #
############################
flags.DEFINE_integer('num_gpu', 2, 'number of gpus for distributed training')
flags.DEFINE_integer('batch_size_per_gpu', 128, 'batch size on 1 gpu')
flags.DEFINE_integer('thread_per_gpu', 4, 'Number of preprocessing threads per tower.')

cfg = tf.app.flags.FLAGS
# tf.logging.set_verbosity(tf.logging.INFO)
