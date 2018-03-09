agent.initialize(sess)
# sess.graph.finalize()

agent.reset()
obs = env.reset()
if eval_env is not None:
    eval_obs = eval_env.reset()

episode_reward = 0.
episode_step = 0
episodes = 0
t = 0

epoch_episode_rewards = []
epoch_episode_steps = []
epoch_actions = []
epoch_qs = []
epoch_episodes = 0
for epoch in range(nb_epochs):
    for cycle in range(nb_epoch_cycles):
        # Perform rollouts.
        for t_rollout in range(nb_rollout_steps):
            # Predict next action.
            action, q = agent.pi(obs, apply_noise=True, compute_Q=True)
            assert action.shape == env.action_space.shape

            # Execute next action.
            if rank == 0 and render:
                env.render()
            assert max_action.shape == action.shape
            new_obs, r, done, info = env.step(
                max_action * action)  # scale for execution in env (as far as DDPG is concerned, every action is in [-1, 1])
            t += 1
            if rank == 0 and render:
                env.render()
            episode_reward += r
            episode_step += 1

            # Book-keeping.
            epoch_actions.append(action)
            epoch_qs.append(q)
            agent.store_transition(obs, action, r, new_obs, done)
            obs = new_obs

            if done:
                # Episode done.
                epoch_episode_rewards.append(episode_reward)
                episode_rewards_history.append(episode_reward)
                epoch_episode_steps.append(episode_step)
                episode_reward = 0.
                episode_step = 0
                epoch_episodes += 1
                episodes += 1

                agent.reset()
                obs = env.reset()

        # Train.
        epoch_actor_losses = []
        epoch_critic_losses = []
        epoch_adaptive_distances = []
        for t_train in range(nb_train_steps):
            # Adapt param noise, if necessary.
            if memory.nb_entries >= batch_size and t_train % param_noise_adaption_interval == 0:
                distance = agent.adapt_param_noise()
                epoch_adaptive_distances.append(distance)

            cl, al = agent.train()
            epoch_critic_losses.append(cl)
            epoch_actor_losses.append(al)
            agent.update_target_net()

        # Evaluate.
        eval_episode_rewards = []
        eval_qs = []
        if eval_env is not None:
            eval_episode_reward = 0.
            for t_rollout in range(nb_eval_steps):
                eval_action, eval_q = agent.pi(eval_obs, apply_noise=False, compute_Q=True)
                eval_obs, eval_r, eval_done, eval_info = eval_env.step(
                    max_action * eval_action)  # scale for execution in env (as far as DDPG is concerned, every action is in [-1, 1])
                if render_eval:
                    eval_env.render()
                eval_episode_reward += eval_r

                eval_qs.append(eval_q)
                if eval_done:
                    eval_obs = eval_env.reset()
                    eval_episode_rewards.append(eval_episode_reward)
                    eval_episode_rewards_history.append(eval_episode_reward)
                    eval_episode_reward = 0.

    if rank == 0:
        summary_writer.add_summary(
            sess.run(training_reward_op, {training_reward: np.mean(epoch_episode_rewards)}), episodes)