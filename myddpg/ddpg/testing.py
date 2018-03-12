import os

from myddpg.ddpg.ddpg import DDPG
import myddpg.tf_util.tf_util as U

import numpy as np
import tensorflow as tf
from mpi4py import MPI


def test(env, nb_epochs, nb_epoch_cycles, render_eval, reward_scale, render, param_noise, actor, critic,
    normalize_returns, normalize_observations, critic_l2_reg, actor_lr, critic_lr, action_noise,
    popart, gamma, clip_norm, nb_train_steps, nb_rollout_steps, nb_eval_steps, batch_size, memory,
    tau=0.01, eval_env=None, param_noise_adaption_interval=50, store_weights=False, env_id=''):
    rank = MPI.COMM_WORLD.Get_rank()

    max_action = env.action_space.high

    agent = DDPG(actor, critic, memory, env.observation_space.shape, env.action_space.shape,
        gamma=gamma, tau=tau, normalize_returns=normalize_returns, normalize_observations=normalize_observations,
        batch_size=batch_size, action_noise=action_noise, param_noise=param_noise, critic_l2_reg=critic_l2_reg,
        actor_lr=actor_lr, critic_lr=critic_lr, enable_popart=popart, clip_norm=clip_norm,
        reward_scale=reward_scale)

    sess = U.single_threaded_session()
    sess.__enter__()

    agent.initialize(sess)

    agent.reset()
    obs = env.reset()

    if rank == 0:
        path_pre = './weights/'
        assert os.path.exists(path_pre) and os.path.exists(path_pre + env_id), 'don\'t exit the dirctory'
        sess = tf.get_default_session()
        saver = tf.train.Saver()
        saver.restore(sess, path_pre + env_id + '/' + env_id + '.cptk')


    for epoch in range(nb_epochs):
        for cycle in range(nb_epoch_cycles):
            for t_rollout in range(nb_rollout_steps):
                # Predict next action.
                action, q = agent.pi(obs, apply_noise=False, compute_Q=True)
                # scale for execution in env (as far as DDPG is concerned, every action is in [-1, 1])
                new_obs, r, done, info = env.step( max_action * action)
                env.render()

                obs = new_obs

                if done:
                    agent.reset()
                    obs = env.reset()



