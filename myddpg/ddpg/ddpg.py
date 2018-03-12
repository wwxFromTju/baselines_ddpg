from copy import copy
from functools import reduce

import numpy as np
import tensorflow as tf
import tensorflow.contrib as tc

from myddpg.mpi_util.mpi_util import MpiAdam
import myddpg.tf_util.tf_util as U
from myddpg.mpi_util.mpi_util import RunningMeanStd
from mpi4py import MPI


# 对stats做一下中心化
def normalize(x, stats):
    if stats is None:
        return x
    return (x - stats.mean) / stats.std


# 通过记录std与mean，将归一化的stats恢复到原来的scale
def denormalize(x, stats):
    if stats is None:
        return x
    return x * stats.std + stats.mean


# 计算某个axis上的平均标准差
def reduce_std(x, axis=None, keepdims=False):
    return tf.sqrt(reduce_var(x, axis=axis, keepdims=keepdims))


# 计算某个axis上的平均方差
def reduce_var(x, axis=None, keepdims=False):
    m = tf.reduce_mean(x, axis=axis, keep_dims=True)
    devs_squared = tf.square(x - m)
    return tf.reduce_mean(devs_squared, axis=axis, keep_dims=keepdims)


# 构造赋值的op，将成对的(var, target_var)分别构造：直接赋值，polyak averaging
def get_target_updates(vars, target_vars, tau):
    soft_updates = []
    init_updates = []
    assert len(vars) == len(target_vars)
    for var, target_var in zip(vars, target_vars):
        init_updates.append(tf.assign(target_var, var))
        soft_updates.append(tf.assign(target_var, (1. - tau) * target_var + tau * var))
    assert len(init_updates) == len(vars)
    assert len(soft_updates) == len(vars)
    return tf.group(*init_updates), tf.group(*soft_updates)


# 在构造的过程中加入噪声
def get_perturbed_actor_updates(actor, perturbed_actor, param_noise_stddev):
    assert len(actor.vars) == len(perturbed_actor.vars)
    assert len(actor.perturbable_vars) == len(perturbed_actor.perturbable_vars)

    updates = []
    for var, perturbed_var in zip(actor.vars, perturbed_actor.vars):
        if var in actor.perturbable_vars:
            updates.append(tf.assign(perturbed_var, var + tf.random_normal(tf.shape(var), mean=0., stddev=param_noise_stddev)))
        else:
            updates.append(tf.assign(perturbed_var, var))
    assert len(updates) == len(actor.vars)
    return tf.group(*updates)


class DDPG(object):
    def __init__(self, actor, critic, memory, observation_shape, action_shape, param_noise=None, action_noise=None,
        gamma=0.99, tau=0.001, normalize_returns=False, enable_popart=False, normalize_observations=True,
        batch_size=128, observation_range=(-5., 5.), action_range=(-1., 1.), return_range=(-np.inf, np.inf),
        adaptive_param_noise=True, adaptive_param_noise_policy_threshold=.1,
        critic_l2_reg=0., actor_lr=1e-4, critic_lr=1e-3, clip_norm=None, reward_scale=1.):
        # Inputs.
        self.obs0 = tf.placeholder(tf.float32, shape=(None,) + observation_shape, name='obs0')
        self.obs1 = tf.placeholder(tf.float32, shape=(None,) + observation_shape, name='obs1')
        self.terminals1 = tf.placeholder(tf.float32, shape=(None, 1), name='terminals1')
        self.rewards = tf.placeholder(tf.float32, shape=(None, 1), name='rewards')
        self.actions = tf.placeholder(tf.float32, shape=(None,) + action_shape, name='actions')
        self.critic_target = tf.placeholder(tf.float32, shape=(None, 1), name='critic_target')
        self.param_noise_stddev = tf.placeholder(tf.float32, shape=(), name='param_noise_stddev')

        # Parameters.
        self.gamma = gamma
        self.tau = tau
        self.memory = memory
        self.normalize_observations = normalize_observations
        self.normalize_returns = normalize_returns
        self.action_noise = action_noise
        self.param_noise = param_noise
        self.action_range = action_range
        self.return_range = return_range
        self.observation_range = observation_range
        self.critic = critic
        self.actor = actor
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.clip_norm = clip_norm
        self.enable_popart = enable_popart
        self.reward_scale = reward_scale
        self.batch_size = batch_size
        self.stats_sample = None
        self.critic_l2_reg = critic_l2_reg

        # Observation normalization.
        if self.normalize_observations:
            with tf.variable_scope('obs_rms'):
                self.obs_rms = RunningMeanStd(shape=observation_shape)
        else:
            self.obs_rms = None
        normalized_obs0 = tf.clip_by_value(normalize(self.obs0, self.obs_rms),
            self.observation_range[0], self.observation_range[1])
        normalized_obs1 = tf.clip_by_value(normalize(self.obs1, self.obs_rms),
            self.observation_range[0], self.observation_range[1])

        # Return normalization.
        if self.normalize_returns:
            with tf.variable_scope('ret_rms'):
                self.ret_rms = RunningMeanStd()
        else:
            self.ret_rms = None

        # Create target networks.
        target_actor = copy(actor)
        target_actor.name = 'target_actor'
        self.target_actor = target_actor
        target_critic = copy(critic)
        target_critic.name = 'target_critic'
        self.target_critic = target_critic

        # Create networks and core TF parts that are shared across setup parts.
        self.actor_tf = actor(normalized_obs0)
        self.normalized_critic_tf = critic(normalized_obs0, self.actions)
        self.critic_tf = denormalize(tf.clip_by_value(self.normalized_critic_tf, self.return_range[0], self.return_range[1]), self.ret_rms)
        self.normalized_critic_with_actor_tf = critic(normalized_obs0, self.actor_tf, reuse=True)
        self.critic_with_actor_tf = denormalize(tf.clip_by_value(self.normalized_critic_with_actor_tf, self.return_range[0], self.return_range[1]), self.ret_rms)
        Q_obs1 = denormalize(target_critic(normalized_obs1, target_actor(normalized_obs1)), self.ret_rms)
        self.target_Q = self.rewards + (1. - self.terminals1) * gamma * Q_obs1

        # Set up parts.
        if self.param_noise is not None:
            self.setup_param_noise(normalized_obs0)

        self.setup_actor_optimizer()
        self.setup_critic_optimizer()

        if self.normalize_returns and self.enable_popart:
            self.setup_popart()

        self.setup_stats()
        self.setup_target_network_updates()

    def setup_target_network_updates(self):
        # 就是设置一下普通的直接赋值的op和增量更新的op
        actor_init_updates, actor_soft_updates = get_target_updates(self.actor.vars, self.target_actor.vars, self.tau)
        critic_init_updates, critic_soft_updates = get_target_updates(self.critic.vars, self.target_critic.vars, self.tau)

        # actor与criti一起更新
        self.target_init_updates = [actor_init_updates, critic_init_updates]
        self.target_soft_updates = [actor_soft_updates, critic_soft_updates]

    def setup_param_noise(self, normalized_obs0):
        assert self.param_noise is not None

        # 注意一下，没有调用的时候，就是一个类，还有设置和初始化对颖的weights
        param_noise_actor = copy(self.actor)
        param_noise_actor.name = 'param_noise_actor'

        # actor 设置input的op，同时初始化对应的参数
        self.perturbed_actor_tf = param_noise_actor(normalized_obs0)
        # 采用在参数空间用noise的方式来更新
        self.perturb_policy_ops = get_perturbed_actor_updates(self.actor, param_noise_actor, self.param_noise_stddev)

        # Configure separate copy for stddev adoption.
        adaptive_param_noise_actor = copy(self.actor)
        adaptive_param_noise_actor.name = 'adaptive_param_noise_actor'
        adaptive_actor_tf = adaptive_param_noise_actor(normalized_obs0)
        self.perturb_adaptive_policy_ops = get_perturbed_actor_updates(self.actor, adaptive_param_noise_actor, self.param_noise_stddev)
        self.adaptive_policy_distance = tf.sqrt(tf.reduce_mean(tf.square(self.actor_tf - adaptive_actor_tf)))

    def setup_actor_optimizer(self):
        # actor的优化，就是直接最大化当前actor的a在critic下面的Q即可
        # 求导的时候 \nabla_{a}(Q(s, pi(a|s))) == \nabla_{pi}(Q(s, pi(a|s))) * \nabla_{a}(pi(a|s))
        self.actor_loss = -tf.reduce_mean(self.critic_with_actor_tf)
        actor_shapes = [var.get_shape().as_list() for var in self.actor.trainable_vars]
        actor_nb_params = sum([reduce(lambda x, y: x * y, shape) for shape in actor_shapes])

        # flatten之后，用adam计算，将全部进程的grad，reduce一下mean，然后就是用adam的方式更新
        self.actor_grads = U.flatgrad(self.actor_loss, self.actor.trainable_vars, clip_norm=self.clip_norm)
        self.actor_optimizer = MpiAdam(var_list=self.actor.trainable_vars,
            beta1=0.9, beta2=0.999, epsilon=1e-08)

    def setup_critic_optimizer(self):
        # 这边就是如果要做normal的话，就对return做一下normal，如果不用的化，就是ret_rms是none的话，
        normalized_critic_target_tf = tf.clip_by_value(normalize(self.critic_target, self.ret_rms), self.return_range[0], self.return_range[1])
        # 类似DQN的loss
        self.critic_loss = tf.reduce_mean(tf.square(self.normalized_critic_tf - normalized_critic_target_tf))

        # 就是对loss加一下weights参数的二范数，约束一下weights不要太大
        if self.critic_l2_reg > 0.:
            critic_reg_vars = [var for var in self.critic.trainable_vars if 'kernel' in var.name and 'output' not in var.name]

            critic_reg = tc.layers.apply_regularization(
                tc.layers.l2_regularizer(self.critic_l2_reg),
                weights_list=critic_reg_vars
            )
            self.critic_loss += critic_reg

        critic_shapes = [var.get_shape().as_list() for var in self.critic.trainable_vars]
        critic_nb_params = sum([reduce(lambda x, y: x * y, shape) for shape in critic_shapes])

        # 与actor一样，就是用mpi，reduce mean一下所有进程的梯度，然后用adam的方式来更新参数
        self.critic_grads = U.flatgrad(self.critic_loss, self.critic.trainable_vars, clip_norm=self.clip_norm)
        self.critic_optimizer = MpiAdam(var_list=self.critic.trainable_vars,
            beta1=0.9, beta2=0.999, epsilon=1e-08)

    def setup_popart(self):
        # See https://arxiv.org/pdf/1602.07714.pdf for details.
        self.old_std = tf.placeholder(tf.float32, shape=[1], name='old_std')
        new_std = self.ret_rms.std
        self.old_mean = tf.placeholder(tf.float32, shape=[1], name='old_mean')
        new_mean = self.ret_rms.mean

        self.renormalize_Q_outputs_op = []
        for vs in [self.critic.output_vars, self.target_critic.output_vars]:
            assert len(vs) == 2
            M, b = vs
            assert 'kernel' in M.name
            assert 'bias' in b.name
            assert M.get_shape()[-1] == 1
            assert b.get_shape()[-1] == 1
            self.renormalize_Q_outputs_op += [M.assign(M * self.old_std / new_std)]
            self.renormalize_Q_outputs_op += [b.assign((b * self.old_std + self.old_mean - new_mean) / new_std)]

    def setup_stats(self):
        ops = []
        names = []

        if self.normalize_returns:
            ops += [self.ret_rms.mean, self.ret_rms.std]
            names += ['ret_rms_mean', 'ret_rms_std']

        if self.normalize_observations:
            ops += [tf.reduce_mean(self.obs_rms.mean), tf.reduce_mean(self.obs_rms.std)]
            names += ['obs_rms_mean', 'obs_rms_std']

        ops += [tf.reduce_mean(self.critic_tf)]
        names += ['reference_Q_mean']
        ops += [reduce_std(self.critic_tf)]
        names += ['reference_Q_std']

        ops += [tf.reduce_mean(self.critic_with_actor_tf)]
        names += ['reference_actor_Q_mean']
        ops += [reduce_std(self.critic_with_actor_tf)]
        names += ['reference_actor_Q_std']

        ops += [tf.reduce_mean(self.actor_tf)]
        names += ['reference_action_mean']
        ops += [reduce_std(self.actor_tf)]
        names += ['reference_action_std']

        if self.param_noise:
            ops += [tf.reduce_mean(self.perturbed_actor_tf)]
            names += ['reference_perturbed_action_mean']
            ops += [reduce_std(self.perturbed_actor_tf)]
            names += ['reference_perturbed_action_std']

        self.stats_ops = ops
        self.stats_names = names

    def pi(self, obs, apply_noise=True, compute_Q=True):
        # 判断是不是要在actor的参数空间中使用noise -》 探索
        if self.param_noise is not None and apply_noise:
            actor_tf = self.perturbed_actor_tf
        else:
            actor_tf = self.actor_tf

        # 计算s下的（Q值和）action
        feed_dict = {self.obs0: [obs]}
        if compute_Q:
            action, q = self.sess.run([actor_tf, self.critic_with_actor_tf], feed_dict=feed_dict)
        else:
            action = self.sess.run(actor_tf, feed_dict=feed_dict)
            q = None

        # 在action中做noise -》 探索
        action = action.flatten()
        if self.action_noise is not None and apply_noise:
            noise = self.action_noise()
            assert noise.shape == action.shape
            action += noise

        # 截断到合理的范围
        action = np.clip(action, self.action_range[0], self.action_range[1])
        return action, q

    def store_transition(self, obs0, action, reward, obs1, terminal1):
        # scale一下reward的尺度
        reward *= self.reward_scale
        # 存到buffer里
        self.memory.append(obs0, action, reward, obs1, terminal1)

        # 这个比较有意思，就是增量地更新obs的之后用来做normalize的一些参数---》有意思的是：对obs做的预处理
        if self.normalize_observations:
            self.obs_rms.update(np.array([obs0]))

    def train(self):
        # 从buffer抽样
        batch = self.memory.sample(batch_size=self.batch_size)

        if self.normalize_returns and self.enable_popart:
            # 先直接计算target Q value
            old_mean, old_std, target_Q = self.sess.run([self.ret_rms.mean, self.ret_rms.std, self.target_Q], feed_dict={
                self.obs1: batch['obs1'],
                self.rewards: batch['rewards'],
                self.terminals1: batch['terminals1'].astype('float32'),
            })
            # 用新的target value来增量更新参数
            self.ret_rms.update(target_Q.flatten())

            self.sess.run(self.renormalize_Q_outputs_op, feed_dict={
                self.old_std : np.array([old_std]),
                self.old_mean : np.array([old_mean]),
            })
        else:
            # 直接计算target Q value
            target_Q = self.sess.run(self.target_Q, feed_dict={
                self.obs1: batch['obs1'],
                self.rewards: batch['rewards'],
                self.terminals1: batch['terminals1'].astype('float32'),
            })

        ops = [self.actor_grads, self.actor_loss, self.critic_grads, self.critic_loss]
        actor_grads, actor_loss, critic_grads, critic_loss = self.sess.run(ops, feed_dict={
            self.obs0: batch['obs0'],
            self.actions: batch['actions'],
            self.critic_target: target_Q,
        })
        self.actor_optimizer.update(actor_grads, stepsize=self.actor_lr)
        self.critic_optimizer.update(critic_grads, stepsize=self.critic_lr)

        return critic_loss, actor_loss

    def initialize(self, sess):
        self.sess = sess
        # 初始化，注意一下因为是随机初始化，所以每个rank的初始化完后，actor，critic的参数都不一样
        self.sess.run(tf.global_variables_initializer())
        # 将actor, critic的参数都设置为rank==0的进程的参数
        self.actor_optimizer.sync()
        self.critic_optimizer.sync()
        # 设置target network的权重等于online network的参数
        self.sess.run(self.target_init_updates)

    def update_target_net(self):
        # 使用polyak averaging的方式来更新参数
        self.sess.run(self.target_soft_updates)

    def get_stats(self):
        # buffer来抽取<S, A, R, S'>
        if self.stats_sample is None:
            self.stats_sample = self.memory.sample(batch_size=self.batch_size)
        # 利用S, A来计算value，具体的内容要看stats里面的op，在setup_stats中，大致就是预处理，然后计算Q，action
        values = self.sess.run(self.stats_ops, feed_dict={
            self.obs0: self.stats_sample['obs0'],
            self.actions: self.stats_sample['actions'],
        })

        # [:] 就是shallow copy的效果，对于可变的是赋值引用
        names = self.stats_names[:]
        assert len(names) == len(values)
        stats = dict(zip(names, values))

        # 如果需要考虑param noise的效果，那么就将dict解开，然后将两个dict合并在一起
        if self.param_noise is not None:
            stats = {**stats, **self.param_noise.get_stats()}

        return stats

    # 就是在param空间使用noise来帮助更有效的探索
    def adapt_param_noise(self):
        if self.param_noise is None:
            return 0.

        # 就是用另外一个noise的actor来做探索，这边就是用现在的actor的权重加上noise，然后更新到noist的actor上
        self.sess.run(self.perturb_adaptive_policy_ops, feed_dict={
            self.param_noise_stddev: self.param_noise.current_stddev,
        })

        # 通过sample出来的数据，来控制加入noise的程度---》参数空间 ！= 策略空间
        batch = self.memory.sample(batch_size=self.batch_size)
        distance = self.sess.run(self.adaptive_policy_distance, feed_dict={
            self.obs0: batch['obs0'],
            self.param_noise_stddev: self.param_noise.current_stddev,
        })
        # 因为加入的noise是随机，所以计算一下全部进程中的加入noise后actor的distance均值
        mean_distance = MPI.COMM_WORLD.allreduce(distance, op=MPI.SUM) / MPI.COMM_WORLD.Get_size()
        self.param_noise.adapt(mean_distance)
        return mean_distance

    def reset(self):
        # 将actor的噪声reset
        if self.action_noise is not None:
            self.action_noise.reset()

        # 然后按照当前的actor的weights，重新set一个noist的actor
        if self.param_noise is not None:
            self.sess.run(self.perturb_policy_ops, feed_dict={
                self.param_noise_stddev: self.param_noise.current_stddev,
            })
