import os
import time
import config
import numpy as np
from dataReader import Reader
import tensorflow as tf
from model.yolo3_model import yolo


def input_fn(data_dir, subset, batch_size):
    """
    Introduction
    ------------
        构建模型的输入模块
    Parameters
    ----------
        data_dir: 训练集数据路径
        subset: 训练集，验证集
        batch_size: batch大小
    Returns
    -------

    """
    dataReader = Reader(subset, data_dir, config.anchors_path, config.num_classes, input_shape = config.input_shape, max_boxes = config.max_boxes, jitter = config.jitter, hue = config.hue, sat = config.sat, cont = config.cont, bri = config.bri)
    image_batch, bbox_true_13, bbox_true_26, bbox_true_52 = dataReader.make_batch(batch_size)
    return image_batch, [bbox_true_13, bbox_true_26, bbox_true_52]



def model_fn(features, labels, mode):
    """
    Introduction
    ------------
        构建模型的Estimator
    Parameters
    ----------
        features: 数据特征
        mode: 模型训练或者验证
    Returns
    -------
        Estimator: 自定义模型Estimator
    """
    model = yolo(config.norm_epsilon, config.norm_decay, config.anchors_path, config.classes_path, config.pre_train, config.weights_file)
    output = model.yolo_inference(features, config.num_anchors / 3, config.num_classes, config.training)
    loss = model.yolo_loss(output, labels, model.anchors, config.num_classes, config.ignore_thresh)
    tf.summary.scalar('loss', loss)
    boundaries = [500, 1000, 1500]
    staged_lr = [config.learning_rate * x for x in [1, 0.01, 0.001, 0.0001]]
    learning_rate = tf.train.piecewise_constant(tf.train.get_global_step(), boundaries, staged_lr)
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
    if config.pre_train:
        train_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='yolo')
        train_op = optimizer.minimize(loss = loss, global_step = tf.train.get_global_step(), var_list = train_var)
    else:
        train_op = optimizer.minimize(loss = loss, global_step = tf.train.get_global_step())
    tensors_to_log = {'learning_rate': learning_rate, 'loss': loss, 'step' : tf.train.get_global_step()}
    logging_hook = tf.train.LoggingTensorHook(tensors = tensors_to_log, every_n_iter=1)
    return tf.estimator.EstimatorSpec(mode = mode, loss = loss,train_op=train_op, training_hooks = [logging_hook], evaluation_hooks = [logging_hook])


def estimator_train():
    """
    Introduction
    ------------
        使用estimator自定义进行训练
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_index
    tf.logging.set_verbosity(tf.logging.INFO)
    model = tf.estimator.Estimator(model_fn = model_fn, model_dir = config.model_dir)
    train_input_spec = tf.estimator.TrainSpec(input_fn = lambda: input_fn(config.data_dir, 'train', config.train_batch_size), max_steps = config.train_num // config.train_batch_size * config.Epoch)
    val_input_spec = tf.estimator.EvalSpec(input_fn = lambda: input_fn(config.data_dir, 'val', config.val_batch_size))
    tf.estimator.train_and_evaluate(model, train_input_spec, val_input_spec)


def train():
    """
    Introduction
    ------------
        训练模型

    """
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_index
    train_data = Reader('train', config.data_dir, config.anchors_path, config.num_classes, input_shape = config.input_shape, max_boxes = config.max_boxes, jitter = config.jitter, hue = config.hue, sat = config.sat, cont = config.cont, bri = config.bri)
    val_data = Reader('val', config.data_dir, config.anchors_path, config.num_classes, input_shape = config.input_shape, max_boxes = config.max_boxes)
    images_train, bbox_true_13_train, bbox_true_26_train, bbox_true_52_train = train_data.provide(config.train_batch_size)
    images_val, bbox_true_13_val, bbox_true_26_val, bbox_true_52_val = val_data.provide(config.val_batch_size)
    model = yolo(config.norm_epsilon, config.norm_decay, config.anchors_path, config.classes_path, config.pre_train, config.weights_file)
    is_training = tf.placeholder(dtype = tf.bool, shape = [])
    images = tf.placeholder(shape = [None, 416, 416, 3], dtype = tf.float32)
    bbox_true_13 = tf.placeholder(shape = [None, 13, 13, 3, 85], dtype = tf.float32)
    bbox_true_26 = tf.placeholder(shape = [None, 26, 26, 3, 85], dtype = tf.float32)
    bbox_true_52 = tf.placeholder(shape = [None, 52, 52, 3, 85], dtype = tf.float32)
    bbox_true = [bbox_true_13, bbox_true_26, bbox_true_52]
    output = model.yolo_inference(images, config.num_anchors / 3, config.num_classes, is_training)
    loss = model.yolo_loss(output, bbox_true, model.anchors, config.num_classes, config.ignore_thresh)
    tf.summary.scalar('loss', loss)
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(config.learning_rate, global_step, 10000, 0.95, staircase = True)
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
    # 如果读取预训练权重，则冻结darknet53网络的变量
    if config.pre_train:
        train_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='yolo')
        train_op = optimizer.minimize(loss = loss, global_step = global_step, var_list = train_var)
    else:
        train_op = optimizer.minimize(loss = loss, global_step = global_step)
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    # 训练时更新batch norm的参数操作
    extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.Session(config = tf.ConfigProto(log_device_placement = False)) as sess:
        ckpt = tf.train.get_checkpoint_state(config.model_dir)
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            print('restore model', ckpt.model_checkpoint_path)
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            sess.run(init)
        summary_writer = tf.summary.FileWriter('./logs', sess.graph)
        tf.train.start_queue_runners(sess = sess)
        best_loss_valid = 10e6
        for epoch in range(config.Epoch):
            for step in range(int(config.train_num / config.train_batch_size)):
                start_time = time.time()
                train_loss, _, _ = sess.run([loss, train_op, extra_update_ops], {images : images_train.eval(), bbox_true_13 : bbox_true_13_train.eval(), bbox_true_26 : bbox_true_26_train.eval(), bbox_true_52 : bbox_true_52_train.eval(), is_training : True})
                duration = time.time() - start_time
                examples_per_sec = float(duration) / config.train_batch_size
                format_str = ('Epoch {} step {},  train loss = {} ( {} examples/sec; {} ''sec/batch)')
                print(format_str.format(epoch, step, train_loss, examples_per_sec, duration))
                summary_writer.add_summary(summary = tf.Summary(value = [tf.Summary.Value(tag = "train loss", simple_value = train_loss)]), global_step = step)
                summary_writer.flush()
            mean_loss_val = []
            for step in range(int(config.val_num / config.val_batch_size)):
                start_time = time.time()
                val_loss = sess.run(loss, {images: images_val.eval(), bbox_true_13: bbox_true_13_val.eval(), bbox_true_26: bbox_true_26_val.eval(), bbox_true_52: bbox_true_52_val.eval(), is_training: False})
                duration = time.time() - start_time
                examples_per_sec = float(duration) / config.val_batch_size
                format_str = ('Epoch {} step {}, val loss = {} ({} examples/sec; {} ''sec/batch)')
                print(format_str.format(epoch, step, val_loss, examples_per_sec, duration))
                summary_writer.add_summary(summary = tf.Summary(value = [tf.Summary.Value(tag = "val loss", simple_value = val_loss)]), global_step = step)
                summary_writer.flush()
                mean_loss_val.append(val_loss)
            mean_loss_valid = np.mean(mean_loss_val)
            if best_loss_valid > mean_loss_valid:
                best_loss_valid = mean_loss_valid
                checkpoint_path = os.path.join(config.model_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step = epoch)



def dstributed_train(ps_hosts, worker_hosts, job_name, task_index):
    """
    Introduction
    ------------
        分布式训练
    Parameters
    ----------
        ps_hosts: sever的host
        worker_hosts: worker的host
        job_name: 判断是作为ps还是worker
        task_index: 任务index
    """
    ps_hosts = ps_hosts.split(',')
    worker_hosts = worker_hosts.split(',')
    cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})
    server = tf.train.Server(cluster, job_name = job_name, task_index = task_index)
    if job_name == 'ps':
        server.join()
    else:
        with tf.device(tf.train.replica_device_setter(worker_device = "/job:worker/task:%d" % task_index, cluster = cluster)):
            global_step = tf.Variable(0, trainable=False)
            is_training = tf.placeholder(dtype=tf.bool, shape=[])
            train_data = Reader('train', config.data_dir, config.anchors_path, config.num_classes, input_shape = config.input_shape, max_boxes = config.max_boxes, jitter=config.jitter, hue = config.hue, sat = config.sat, cont = config.cont, bri = config.bri)
            val_data = Reader('val', config.data_dir, config.anchors_path, config.num_classes, input_shape = config.input_shape, max_boxes = config.max_boxes)
            images_train, bbox_true_13_train, bbox_true_26_train, bbox_true_52_train = train_data.provide(config.train_batch_size)
            images_val, bbox_true_13_val, bbox_true_26_val, bbox_true_52_val = val_data.provide(config.val_batch_size)
            model = yolo(config.norm_epsilon, config.norm_decay, config.anchors_path, config.classes_path, config.pre_train, config.weights_file)
            images = tf.placeholder(shape = [None, 416, 416, 3], dtype = tf.float32)
            bbox_true_13 = tf.placeholder(shape = [None, 13, 13, 3, 85], dtype = tf.float32)
            bbox_true_26 = tf.placeholder(shape = [None, 26, 26, 3, 85], dtype = tf.float32)
            bbox_true_52 = tf.placeholder(shape = [None, 52, 52, 3, 85], dtype = tf.float32)
            bbox_true = [bbox_true_13, bbox_true_26, bbox_true_52]
            output = model.yolo_inference(images, config.num_anchors / 3, config.num_classes, is_training)
            loss, accuracy, pred_object = model.yolo_loss(output, bbox_true, model.anchors, config.num_classes, config.ignore_thresh)
            tf.summary.scalar('loss', loss)
            learning_rate = tf.train.exponential_decay(config.learning_rate, global_step, 1000, 0.95, staircase = True)
            optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
            # 如果读取预训练权重，则冻结darknet53网络的变量
            if config.pre_train:
                train_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='yolo')
                train_op = optimizer.minimize(loss = loss, global_step = global_step, var_list = train_var)
            else:
                train_op = optimizer.minimize(loss = loss, global_step = global_step)
            init = tf.global_variables_initializer()
            saver = tf.train.Saver()
            # 训练时更新batch norm的参数操作
            extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.Session(config = tf.ConfigProto(log_device_placement = False)) as sess:
                ckpt = tf.train.get_checkpoint_state(config.model_dir)
                if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
                    saver.restore(sess, ckpt.model_checkpoint_path)
                else:
                    sess.run(init)
                summary_writer = tf.summary.FileWriter('./logs', sess.graph)
                tf.train.start_queue_runners(sess =sess)

                for step in range(int(config.Epoch * config.train_num / config.train_batch_size)):
                    loss_value, _, _, pred = sess.run(
                        [loss, train_op, extra_update_ops, pred_object],
                        {images: images_train.eval(), bbox_true_13: bbox_true_13_train.eval(),
                         bbox_true_26: bbox_true_26_train.eval(), bbox_true_52: bbox_true_52_train.eval(),
                         is_training: True})
                    for index in range(3):
                        print("pred: object: {}".format(pred[index][pred[index] > 0]))
                    if step % 2 == 0:
                        start_time = time.time()
                        train_loss = sess.run(loss, {images: images_train.eval(), bbox_true_13: bbox_true_13_train.eval(), bbox_true_26: bbox_true_26_train.eval(), bbox_true_52: bbox_true_52_train.eval(), is_training: True})
                        duration = time.time() - start_time
                        examples_per_sec = float(duration) / config.train_batch_size
                        format_str = ('step {},  train loss = {} ( {} examples/sec; {} ''sec/batch)')
                        print(format_str.format(step, train_loss, examples_per_sec, duration))
                        summary_writer.add_summary(
                            summary=tf.Summary(value=[tf.Summary.Value(tag="train loss", simple_value=train_loss)]),
                            global_step=step)
                        summary_writer.flush()
                    # Save the model checkpoint periodically.
                    if step > 1 and step % 1000 == 0:
                        checkpoint_path = os.path.join(config.model_dir, 'model.ckpt')
                        saver.save(sess, checkpoint_path, global_step=step)
                    # Run validation periodically
                    if step > 1 and step % 20 == 0:
                        start_time = time.time()
                        val_loss = sess.run(loss, {images: images_val.eval(), bbox_true_13: bbox_true_13_val.eval(), bbox_true_26: bbox_true_26_val.eval(), bbox_true_52: bbox_true_52_val.eval(), is_training: False})
                        duration = time.time() - start_time
                        examples_per_sec = float(duration) / config.val_batch_size
                        format_str = ('step {}, val loss = {} ({} examples/sec; {} ''sec/batch)')
                        print(format_str.format(step, val_loss, examples_per_sec, duration))
                        summary_writer.add_summary(summary = tf.Summary(value=[tf.Summary.Value(tag="val loss", simple_value = val_loss)]), global_step = step)
                        summary_writer.flush()

if __name__ == "__main__":
    train()








