from baselines.common import Dataset, explained_variance, fmt_row, zipsame
from baselines import logger
import baselines.common.tf_util as U
import tensorflow as tf, numpy as np
import time
from baselines.common.mpi_adam import MpiAdam
from baselines.common.mpi_moments import mpi_moments
from mpi4py import MPI
from collections import deque
import itertools
import collections


def result_record():
    global lenbuffer, rewbuffer, iters_so_far, timesteps_so_far, \
        episodes_so_far, tstart
    if len(lenbuffer) == 0:
        mean_lenbuffer = 0
    else:
        mean_lenbuffer = np.mean(lenbuffer)
    if len(rewbuffer) == 0:
        # TODO: Add pong game checking
        mean_rewbuffer = 0
    else:
        mean_rewbuffer = np.mean(rewbuffer)
    logger.record_tabular("EpLenMean", mean_lenbuffer)
    logger.record_tabular("EpRewMean", mean_rewbuffer)
    logger.record_tabular("EpisodesSoFar", episodes_so_far)
    logger.record_tabular("TimestepsSoFar", timesteps_so_far)
    logger.record_tabular("TimeElapsed", time.time() - tstart)
    if MPI.COMM_WORLD.Get_rank() == 0:
        logger.dump_tabular()


def add_vtarg_and_adv(seg, gamma, lam):
    """
    Compute target value using TD(lambda) estimator, and advantage with GAE(lambda)
    """
    new = np.append(seg["new"], 0)  # last element is only used for last vtarg, but we already zeroed it if last new = 1
    vpred = np.append(seg["vpred"], seg["nextvpred"])
    T = len(seg["rew"])
    seg["adv"] = gaelam = np.empty(T, 'float32')
    rew = seg["rew"]
    lastgaelam = 0
    for t in reversed(range(T)):
        nonterminal = 1 - new[t + 1]
        delta = rew[t] + gamma * vpred[t + 1] * nonterminal - vpred[t]
        gaelam[t] = lastgaelam = delta + gamma * lam * nonterminal * lastgaelam
    seg["tdlamret"] = seg["adv"] + seg["vpred"]


def learn(env, policy_fn, *,
          timesteps_per_actorbatch,  # timesteps per actor per update
          clip_param, entcoeff,  # clipping parameter epsilon, entropy coeff
          optim_epochs, optim_stepsize, optim_batchsize,  # optimization hypers
          gamma, lam,  # advantage estimation
          max_timesteps = 0, max_episodes = 0, max_iters = 0, max_seconds = 0,  # time constraint
          callback = None,  # you can do anything in the callback, since it takes locals(), globals()
          adam_epsilon = 1e-5,
          schedule = 'constant'  # annealing for stepsize parameters (epsilon and adam)
          ):
    # Setup losses and stuff
    # ----------------------------------------
    ob_space = env.observation_space
    ac_space = env.action_space
    pi = policy_fn("pi", ob_space, ac_space)  # Construct network for new policy
    td_v_target = tf.placeholder(dtype = tf.float32, shape = [1, 1])  # V target

    lrmult = tf.placeholder(name = 'lrmult', dtype = tf.float32,
                            shape = [])  # learning rate multiplier, updated with schedule

    ob = U.get_placeholder_cached(name = "ob")
    ac = pi.pdtype.sample_placeholder([None])
    adv = tf.placeholder(dtype = tf.float32, shape = [1, 1])

    ent = pi.pd.entropy()

    vf_loss = tf.reduce_mean(tf.square(pi.vpred - td_v_target))
    vf_losses = [vf_loss]
    vf_loss_names = ["vf_loss"]

    pol_loss = tf.reduce_mean(adv * pi.pd.neglogp(ac))
    pol_losses = [pol_loss]
    pol_loss_names = ["pol_loss"]

    var_list = pi.get_trainable_variables()
    vf_var_list = [v for v in var_list if v.name.split("/")[1].startswith(
        "vf")]
    pol_var_list = [v for v in var_list if v.name.split("/")[1].startswith(
        "pol")]

    # vf_var_list = [v for v in var_list if v.name.split("/")[1].startswith(
    #     "vf") and v.name.split("/")[2].startswith(
    #     "final")]
    # pol_var_list = [v for v in var_list if v.name.split("/")[1].startswith(
    #     "pol") and v.name.split("/")[2].startswith(
    #     "final")]

    # Train V function
    vf_lossandgrad = U.function([ob, td_v_target, lrmult],
                                vf_losses + [U.flatgrad(vf_loss, vf_var_list)])
    vf_adam = MpiAdam(vf_var_list, epsilon = adam_epsilon)

    # vf_optimizer = tf.train.AdamOptimizer(learning_rate = lrmult, epsilon = adam_epsilon)
    # vf_train_op = vf_optimizer.minimize(vf_loss, vf_var_list)

    # Train Policy
    pol_lossandgrad = U.function([ob, ac, adv, lrmult],
                                 pol_losses + [U.flatgrad(pol_loss, pol_var_list)])
    pol_adam = MpiAdam(pol_var_list, epsilon = adam_epsilon)

    # pol_optimizer = tf.train.AdamOptimizer(learning_rate = 0.1 * lrmult, epsilon = adam_epsilon)
    # pol_train_op = pol_optimizer.minimize(pol_loss, pol_var_list)

    # Computation
    compute_v_pred = U.function([ob], [pi.vpred])
    # vf_update = U.function([ob, td_v_target], [vf_train_op])
    # pol_update = U.function([ob, ac, adv], [pol_train_op])

    U.initialize()
    vf_adam.sync()
    pol_adam.sync()
    # Prepare for rollouts
    # ----------------------------------------

    global timesteps_so_far, episodes_so_far, iters_so_far, \
        tstart, lenbuffer, rewbuffer, best_fitness
    episodes_so_far = 0
    timesteps_so_far = 0
    iters_so_far = 0
    tstart = time.time()
    lenbuffer = deque(maxlen = 100)  # rolling buffer for episode lengths
    rewbuffer = deque(maxlen = 100)  # rolling buffer for episode rewards
    Transition = collections.namedtuple("Transition", ["ob", "ac", "reward", "next_ob", "done"])

    assert sum([max_iters > 0, max_timesteps > 0, max_episodes > 0,
                max_seconds > 0]) == 1, "Only one time constraint permitted"

    # Step learning, this loop now indicates episodes
    while True:
        if callback: callback(locals(), globals())
        if max_timesteps and timesteps_so_far >= max_timesteps:
            break
        elif max_episodes and episodes_so_far >= max_episodes:
            break
        elif max_iters and iters_so_far >= max_iters:
            break
        elif max_seconds and time.time() - tstart >= max_seconds:
            break

        if schedule == 'constant':
            cur_lrmult = 1.0
        elif schedule == 'linear':
            cur_lrmult = max(1.0 - float(timesteps_so_far) / max_timesteps, 0)
        else:
            raise NotImplementedError

        logger.log("********** Episode %i ************" % episodes_so_far)

        ob = env.reset()
        # episode = []
        cur_ep_ret = 0  # return in current episode
        cur_ep_len = 0  # len of current episode
        ep_rets = []  # returns of completed episodes in this segment
        ep_lens = []  # lengths of ...

        obs = []
        for t in itertools.count():
            ac, vpred = pi.act(stochastic = True, ob = ob)

            obs.append(ob)
            next_ob, rew, done, _ = env.step(ac)
            # episode.append(Transition(ob=ob.reshape((1, ob.shape[0])), ac=ac.reshape((1, ac.shape[0])), reward=rew, next_ob=next_ob.reshape((1, ob.shape[0])), done=done))
            cur_ep_ret += rew
            cur_ep_len += 1
            timesteps_so_far += 1

            # Compute v target and TD
            v_target = rew + gamma * np.array(compute_v_pred(next_ob.reshape((1, ob.shape[0]))))
            adv = v_target - np.array(compute_v_pred(ob.reshape((1, ob.shape[0]))))

            # Update V and Update Policy
            vf_loss, vf_g = vf_lossandgrad(ob.reshape((1, ob.shape[0])), v_target,
                                           optim_stepsize * cur_lrmult)
            vf_adam.update(vf_g, optim_stepsize * cur_lrmult)
            pol_loss, pol_g = pol_lossandgrad(ob.reshape((1, ob.shape[0])), ac.reshape((1, ac.shape[0])), adv,
                                              optim_stepsize * cur_lrmult)
            pol_adam.update(pol_g, optim_stepsize * cur_lrmult * 0.1)
            ob = next_ob
            if timesteps_so_far % 10000 == 0 and timesteps_so_far > 0:
                result_record()
            if done:
                print(
                    "Episode {} - Total reward = {}, Total Steps = {}".format(episodes_so_far, cur_ep_ret, cur_ep_len))
                ep_rets.append(cur_ep_ret)  # returns of completed episodes in this segment
                ep_lens.append(cur_ep_len)  # lengths of ..
                rewbuffer.extend(ep_rets)
                lenbuffer.extend(ep_lens)

                if hasattr(pi, "ob_rms"): pi.ob_rms.update(np.array(obs))  # update running mean/std for normalization
                iters_so_far += 1
                episodes_so_far += 1
                break


def flatten_lists(listoflists):
    return [el for list_ in listoflists for el in list_]
