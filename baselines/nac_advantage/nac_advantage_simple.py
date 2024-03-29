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
from baselines.common.normalizer import Normalizer

def traj_segment_generator(pi, env, horizon, stochastic):
    global timesteps_so_far
    t = 0
    ac = env.action_space.sample()  # not used, just so we have the datatype
    new = True  # marks if we're on first timestep of an episode
    ob = env.reset()

    cur_ep_ret = 0  # return in current episode
    cur_ep_len = 0  # len of current episode
    ep_rets = []  # returns of completed episodes in this segment
    ep_lens = []  # lengths of ...

    # Initialize history arrays
    obs = np.array([ob for _ in range(horizon)])
    rews = np.zeros(horizon, 'float32')
    vpreds = np.zeros(horizon, 'float32')
    news = np.zeros(horizon, 'int32')
    acs = np.array([ac for _ in range(horizon)])
    prevacs = acs.copy()
    ep_num = 0

    while True:
        prevac = ac
        ac, vpred = pi.act(stochastic, ob)
        ac = np.clip(ac, env.action_space.low, env.action_space.high)
        # Slight weirdness here because we need value function at time T
        # before returning segment [0, T-1] so we get the correct
        # terminal value
        if t > 0 and t % horizon == 0 and ep_num >= 5:
            yield {"ob": obs, "rew": rews, "vpred": vpreds, "new": news,
                   "ac": acs, "prevac": prevacs, "nextvpred": vpred * (1 - new),
                   "ep_rets": ep_rets, "ep_lens": ep_lens}
            # Be careful!!! if you change the downstream algorithm to aggregate
            # several of these batches, then be sure to do a deepcopy
            ep_rets = []
            ep_lens = []
            ep_num = 0
            cur_ep_ret = 0
            cur_ep_len = 0
        i = t % horizon
        obs[i] = ob
        vpreds[i] = vpred
        news[i] = new
        acs[i] = ac
        prevacs[i] = prevac
        ac = np.clip(ac, env.action_space.low, env.action_space.high)
        ob, rew, new, _ = env.step(ac)
        # rew = np.clip(rew, -1., 1.)
        rews[i] = rew

        cur_ep_ret += rew
        cur_ep_len += 1
        # timesteps_so_far += 1
        if new:
            ep_num += 1
            ep_rets.append(cur_ep_ret)
            ep_lens.append(cur_ep_len)
            cur_ep_ret = 0
            cur_ep_len = 0
            ob = env.reset()
        t += 1


def result_record():
    global lenbuffer, rewbuffer, iters_so_far, timesteps_so_far, \
        episodes_so_far, tstart

    print(rewbuffer)
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
          shift = 0,
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
    G_t_inv = tf.placeholder(dtype = tf.float32, shape = [None, None])
    alpha = tf.placeholder(dtype = tf.float32, shape = [1])
    ob = U.get_placeholder_cached(name = "ob")
    ac = pi.pdtype.sample_placeholder([])
    adv = tf.placeholder(dtype = tf.float32, shape = [1, 1])
    step = tf.placeholder(dtype = tf.float32, shape = [1])

    vf_loss = tf.reduce_mean(tf.square(pi.vpred - td_v_target))
    vf_losses = [vf_loss]
    vf_loss_names = ["vf_loss"]

    pol_loss = -tf.reduce_mean(adv * pi.pd.logp(ac))
    pol_losses = [pol_loss]
    pol_loss_names = ["pol_loss"]

    var_list = pi.get_trainable_variables()
    vf_var_list = [v for v in var_list if v.name.split("/")[1].startswith(
        "vf")]
    pol_var_list = [v for v in var_list if v.name.split("/")[1].startswith(
        "pol")]

    compatible_feature = U.flatgrad(pi.pd.neglogp(ac), pol_var_list)

    G_t_inv_next = 1/(1-alpha) * (G_t_inv -
                                  alpha * (G_t_inv * compatible_feature)*tf.transpose(G_t_inv * compatible_feature)
                                  / (1 - alpha + alpha * tf.transpose(compatible_feature) * G_t_inv * compatible_feature))

    # Train V function
    vf_lossandgrad = U.function([ob, td_v_target, lrmult],
                                vf_losses + [U.flatgrad(vf_loss, vf_var_list, 20.0)])
    vf_adam = MpiAdam(vf_var_list, epsilon = adam_epsilon)

    # vf_optimizer = tf.train.AdamOptimizer(learning_rate = lrmult, epsilon = adam_epsilon)
    # vf_train_op = vf_optimizer.minimize(vf_loss, vf_var_list)

    # Train Policy
    pol_lossandgrad = U.function([ob, ac, adv, lrmult, td_v_target],
                                 pol_losses + [U.flatgrad(pol_loss, pol_var_list, 20.0)])
    pol_adam = MpiAdam(pol_var_list, epsilon = adam_epsilon)

    # pol_optimizer = tf.train.AdamOptimizer(learning_rate = 0.1 * lrmult, epsilon = adam_epsilon)
    # pol_train_op = pol_optimizer.minimize(pol_loss, pol_var_list)

    # Computation
    compute_v_pred = U.function([ob], [pi.vpred])
    get_pol_weights_num = np.sum([np.prod(v.get_shape().as_list()) for v in pol_var_list])
    get_compatible_feature = U.function([ob, ac], [compatible_feature])
    # vf_update = U.function([ob, td_v_target], [vf_train_op])
    # pol_update = U.function([ob, ac, adv], [pol_train_op])

    U.initialize()
    vf_adam.sync()
    pol_adam.sync()
    # Prepare for rollouts
    # ----------------------------------------

    seg_gen = traj_segment_generator(pi, env, timesteps_per_actorbatch, stochastic=False)

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
    omega_t = np.zeros(get_pol_weights_num)
    normalizer = Normalizer(1)
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

        # logger.log("********** Episode %i ************" % episodes_so_far)

        rac_alpha = optim_stepsize * cur_lrmult
        rac_beta = optim_stepsize * cur_lrmult * 0.05
        #
        # print("rac_alpha=", rac_alpha)
        # print("rac_beta=", rac_beta)
        if timesteps_so_far == 0:
            # result_record()
            seg = seg_gen.__next__()
            lrlocal = (seg["ep_lens"], seg["ep_rets"]) # local values
            listoflrpairs = MPI.COMM_WORLD.allgather(lrlocal) # list of tuples
            lens, rews = map(flatten_lists, zip(*listoflrpairs))
            lenbuffer.extend(lens)
            rewbuffer.extend(rews)
            result_record()

        ob = env.reset()
        # episode = []
        cur_ep_ret = 0  # return in current episode
        cur_ep_len = 0  # len of current episode
        ep_rets = []  # returns of completed episodes in this segment
        ep_lens = []  # lengths of ...

        obs = []
        record = False
        for t in itertools.count():
            ac, vpred = pi.act(stochastic = True, ob = ob)
            origin_ac = ac
            ac = np.clip(ac, ac_space.low, ac_space.high)
            obs.append(ob)
            next_ob, rew, done, _ = env.step(ac)
            if env.spec._env_name == "MountainCarContinuous":
                rew = rew - np.abs(next_ob[0] - env.unwrapped.goal_position)
            ac = origin_ac

            # rew = np.clip(rew, -1., 1.)
            # episode.append(Transition(ob=ob.reshape((1, ob.shape[0])), ac=ac.reshape((1, ac.shape[0])), reward=rew, next_ob=next_ob.reshape((1, ob.shape[0])), done=done))

            original_rew = rew
            if env.spec._env_name != "InvertedPendulumBulletEnv":
                normalizer.update(rew)
                rew = normalizer.normalize(rew)
            cur_ep_ret += (original_rew - shift)
            cur_ep_len += 1
            timesteps_so_far += 1
            # Compute v target and TD
            v_target = rew + gamma * np.array(compute_v_pred(next_ob.reshape((1, ob.shape[0]))))
            adv = v_target - np.array(compute_v_pred(ob.reshape((1, ob.shape[0]))))

            # Update V and Update Policy
            vf_loss, vf_g = vf_lossandgrad(ob.reshape((1, ob.shape[0])), v_target,
                                           rac_alpha)
            vf_adam.update(vf_g, rac_alpha)
            pol_loss, pol_g = pol_lossandgrad(ob.reshape((1, ob.shape[0])), ac, adv,
                                              rac_beta)
            compatible_feature = np.array(
                get_compatible_feature(ob.reshape((1, ob.shape[0])), ac))
            compatible_feature_product = compatible_feature * compatible_feature.T
            omega_t = (np.eye(compatible_feature_product.shape[0]) - 0.1 * rac_alpha * compatible_feature_product).dot(
                omega_t) \
                      + 0.1 * rac_alpha * pol_g

            pol_adam.update(omega_t, rac_beta)
            ob = next_ob
            if timesteps_so_far % 10000 == 0:
                record = True
            if done:
                # print(
                #     "Episode {} - Total reward = {}, Total Steps = {}".format(episodes_so_far, cur_ep_ret, cur_ep_len))
                # ep_rets.append(cur_ep_ret)  # returns of completed episodes in this segment
                # ep_lens.append(cur_ep_len)  # lengths of ..

                # lenbuffer.append(cur_ep_len)
                # rewbuffer.append(cur_ep_ret)

                if hasattr(pi, "ob_rms"): pi.ob_rms.update(np.array(obs))  # update running mean/std for normalization
                iters_so_far += 1
                episodes_so_far += 1
                ob = env.reset()
                if record:
                    seg = seg_gen.__next__()
                    lrlocal = (seg["ep_lens"], seg["ep_rets"])  # local values
                    listoflrpairs = MPI.COMM_WORLD.allgather(lrlocal)  # list of tuples
                    lens, rews = map(flatten_lists, zip(*listoflrpairs))
                    lenbuffer.extend(lens)
                    rewbuffer.extend(rews)
                    result_record()
                    record = False
                break


def flatten_lists(listoflists):
    return [el for list_ in listoflists for el in list_]
