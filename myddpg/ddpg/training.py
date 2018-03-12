import os
from collections import deque

from myddpg.ddpg.ddpg import DDPG
import myddpg.tf_util.tf_util as U

import numpy as np
import tensorflow as tf
from mpi4py import MPI


def train(env, nb_epochs, nb_epoch_cycles, reward_scale, param_noise, actor, critic,
    normalize_returns, normalize_observations, critic_l2_reg, actor_lr, critic_lr, action_noise,
    popart, gamma, clip_norm, nb_train_steps, nb_rollout_steps, batch_size, memory,
    tau=0.01, param_noise_adaption_interval=50, store_weights=False, env_id=''):

    rank = MPI.COMM_WORLD.Get_rank()
    max_action = env.action_space.high

    agent = DDPG(actor, critic, memory, env.observation_space.shape, env.action_space.shape,
        gamma=gamma, tau=tau, normalize_returns=normalize_returns, normalize_observations=normalize_observations,
        batch_size=batch_size, action_noise=action_noise, param_noise=param_noise, critic_l2_reg=critic_l2_reg,
        actor_lr=actor_lr, critic_lr=critic_lr, enable_popart=popart, clip_norm=clip_norm,
        reward_scale=reward_scale)

    # 用来记录训练中的reward，可以通过reward的变化来判断agent是否学到了
    summary_writer = tf.summary.FileWriter('./log')
    training_episode_reward = tf.Variable(0, dtype=tf.float32)
    training_episode_reward_op = tf.summary.scalar('mean_episode_reward', training_episode_reward)
    training_epoch_reward = tf.Variable(0, dtype=tf.float32)
    training_epoch_reward_op = tf.summary.scalar('mean_epoch_reward', training_epoch_reward)

    sess = U.single_threaded_session()
    sess.__enter__()

    agent.initialize(sess)
    # sess.graph.finalize()

    agent.reset()
    obs = env.reset()

    episode_reward = 0.
    episodes = 0
    t = 0

    epoch_episode_rewards = []
    epoch_episodes = 0
    for epoch in range(nb_epochs):
        for cycle in range(nb_epoch_cycles):
            # rollout 收集数据
            for t_rollout in range(nb_rollout_steps):
                action, q = agent.pi(obs, apply_noise=True, compute_Q=True)
                # 注意一下model里面的激活函数是tanh，所以这边scale一下，到实际中环境中的action的range
                new_obs, r, done, info = env.step(max_action * action)
                t += 1
                episode_reward += r

                # 将transition存储在一起
                agent.store_transition(obs, action, r, new_obs, done)
                obs = new_obs

                # 主要是reset，顺便记录一下summary
                if done:
                    # 记录一下每个episode的平均reward
                    if rank == 0:
                        summary_writer.add_summary(
                            sess.run(training_episode_reward_op, {training_episode_reward: np.mean(episode_reward)}), episodes)
                        summary_writer.add_summary(
                            sess.run(training_epoch_reward_op, {training_epoch_reward: np.mean(epoch_episode_rewards)}),
                            episodes)
                        print('Episode:{}, Reward:{}'.format(episodes, np.mean(episode_reward)))
                        print('Episode:{}, Epoch Reward:{}'.format(episodes, np.mean(epoch_episode_rewards)))

                    epoch_episode_rewards.append(episode_reward)
                    episode_reward = 0.
                    epoch_episodes += 1
                    episodes += 1

                    agent.reset()
                    obs = env.reset()

            # 训练网络
            for t_train in range(nb_train_steps):
                # 调节一下noise的参数
                if memory.nb_entries >= batch_size and t_train % param_noise_adaption_interval == 0:
                    agent.adapt_param_noise()
                # 训练网络
                agent.train()
                # 更新target网络
                agent.update_target_net()


    # 存储NN的权重
    if rank == 0:
        path_pre = './weights/'
        if MPI.COMM_WORLD.Get_rank() == 0 and store_weights:
            if not os.path.exists(path_pre):
                os.mkdir(path_pre)
            if not os.path.exists(path_pre + env_id):
                os.mkdir(path_pre + env_id)
            sess = tf.get_default_session()
            saver = tf.train.Saver()
            saver.save(sess, path_pre + env_id + '/' + env_id + '.cptk')
