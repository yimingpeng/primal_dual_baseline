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


def traj_segment_generator(pi, env, horizon,
                                     vf_lossandgrad,
                                     vf_adam,
                                     pol_lossandgrad,
                                     pol_adam,
                                     compute_v_pred,
                                     gamma,
                                     cur_lrmult,
                                     optim_stepsize,
                                     rho,
                                     update_step_threshold,
                                     stochastic):
    global timesteps_so_far
    t = 0
    ac = env.action_space.sample() # not used, just so we have the datatype
    new = True # marks if we're on first timestep of an episode
    ob = env.reset()

    cur_ep_ret = 0 # return in current episode
    cur_ep_len = 0 # len of current episode
    ep_rets = [] # returns of completed episodes in this segment
    ep_lens = [] # lengths of ...

    # Initialize history arrays
    obs = np.array([ob for _ in range(horizon)])
    rews = np.zeros(horizon, 'float32')
    vpreds = np.zeros(horizon, 'float32')
    news = np.zeros(horizon, 'int32')
    acs = np.array([ac for _ in range(horizon)])
    prevacs = acs.copy()

    pol_gradients = []
    t_0 = 0
    while True:
        if timesteps_so_far % 10000 == 0 and timesteps_so_far > 0:
            result_record()
        prevac = ac
        ac, vpred = pi.act(stochastic, ob)
        # Slight weirdness here because we need value function at time T
        # before returning segment [0, T-1] so we get the correct
        # terminal value
        if t > 0 and t % horizon == 0:
            yield {"ob" : obs, "rew" : rews, "vpred" : vpreds, "new" : news,
                    "ac" : acs, "prevac" : prevacs, "nextvpred": vpred * (1 - new),
                    "ep_rets" : ep_rets, "ep_lens" : ep_lens}
            # Be careful!!! if you change the downstream algorithm to aggregate
            # several of these batches, then be sure to do a deepcopy
            ep_rets = []
            ep_lens = []
        i = t % horizon
        obs[i] = ob
        vpreds[i] = vpred
        news[i] = new
        acs[i] = ac
        prevacs[i] = prevac
        if env.spec._env_name == "LunarLanderContinuous":
            ac = np.clip(ac, -1.0, 1.0)
        next_ob, rew, new, _ = env.step(ac)

        # Compute v target and TD
        v_target = rew + gamma * np.array(compute_v_pred(next_ob.reshape((1, ob.shape[0]))))
        adv = v_target - np.array(compute_v_pred(ob.reshape((1, ob.shape[0]))))

        # Update V and Update Policy
        vf_loss, vf_g = vf_lossandgrad(ob.reshape((1, ob.shape[0])), v_target,
                                       cur_lrmult)
        vf_adam.update(vf_g, optim_stepsize * cur_lrmult)
        pol_loss, pol_g = pol_lossandgrad(ob.reshape((1, ob.shape[0])), ac.reshape((1, ac.shape[0])), adv,
                                          cur_lrmult)
        pol_gradients.append(pol_g)

        if t % update_step_threshold == 0 and t > 0:
            scaling_factor = [rho ** (t - i) for i in range(t_0, t)]
            coef = t/np.sum(scaling_factor)
            sum_weighted_pol_gradients = np.sum([scaling_factor[i] * pol_gradients[i] for i in range(len(scaling_factor))], axis = 0)
            pol_adam.update(coef*sum_weighted_pol_gradients, optim_stepsize * 0.1 * cur_lrmult)
            pol_gradients = []
            t_0 = t

        rews[i] = rew

        cur_ep_ret += rew
        cur_ep_len += 1
        timesteps_so_far+=1
        ob = next_ob
        if new:
            # Episode End Update
            scaling_factor = [rho ** (t - i) for i in range(t_0, t)]
            coef = t/np.sum(scaling_factor)
            sum_weighted_pol_gradients = np.sum([scaling_factor[i] * pol_gradients[i] for i in range(len(scaling_factor))], axis = 0)
            pol_adam.update(coef*sum_weighted_pol_gradients, optim_stepsize * 0.1 * cur_lrmult)
            pol_gradients = []
            t_0 = t

            print(
                "Episode {} - Total reward = {}, Total Steps = {}".format(episodes_so_far, cur_ep_ret, cur_ep_len))
            ep_rets.append(cur_ep_ret)
            ep_lens.append(cur_ep_len)
            cur_ep_ret = 0
            cur_ep_len = 0
            ob = env.reset()
        t += 1

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
    new = np.append(seg["new"], 0) # last element is only used for last vtarg, but we already zeroed it if last new = 1
    vpred = np.append(seg["vpred"], seg["nextvpred"])
    T = len(seg["rew"])
    seg["adv"] = gaelam = np.empty(T, 'float32')
    rew = seg["rew"]
    lastgaelam = 0
    for t in reversed(range(T)):
        nonterminal = 1-new[t+1]
        delta = rew[t] + gamma * vpred[t+1] * nonterminal - vpred[t]
        gaelam[t] = lastgaelam = delta + gamma * lam * nonterminal * lastgaelam
    seg["tdlamret"] = seg["adv"] + seg["vpred"]

def learn(env, policy_fn, *,
        timesteps_per_actorbatch, # timesteps per actor per update
        clip_param, entcoeff, # clipping parameter epsilon, entropy coeff
        optim_epochs, optim_stepsize, optim_batchsize,# optimization hypers
        gamma, lam, # advantage estimation
        max_timesteps=0, max_episodes=0, max_iters=0, max_seconds=0,  # time constraint
        callback=None, # you can do anything in the callback, since it takes locals(), globals()
        adam_epsilon=1e-5,
        rho = 0.95,  # Gradient weighting factor
        update_step_threshold = 100, # Updating step threshold
        schedule='constant' # annealing for stepsize parameters (epsilon and adam)
        ):
    # Setup losses and stuff
    # ----------------------------------------
    ob_space = env.observation_space
    ac_space = env.action_space
    pi = policy_fn("pi", ob_space, ac_space) # Construct network for new policy
    oldpi = policy_fn("oldpi", ob_space, ac_space) # Network for old policy
    atarg = tf.placeholder(dtype=tf.float32, shape=[None]) # Target advantage function (if applicable)
    ret = tf.placeholder(dtype=tf.float32, shape=[None]) # Empirical return

    td_v_target = tf.placeholder(dtype = tf.float32, shape = [1, 1])  # V target for RAC

    lrmult = tf.placeholder(name='lrmult', dtype=tf.float32, shape=[]) # learning rate multiplier, updated with schedule
    adv = tf.placeholder(dtype = tf.float32, shape = [1, 1]) # Advantage function for RAC

    clip_param = clip_param * lrmult # Annealed cliping parameter epislon

    ob = U.get_placeholder_cached(name="ob")
    ac = pi.pdtype.sample_placeholder([None])

    kloldnew = oldpi.pd.kl(pi.pd)
    ent = pi.pd.entropy()
    meankl = tf.reduce_mean(kloldnew)
    meanent = tf.reduce_mean(ent)
    pol_entpen = (-entcoeff) * meanent

    ratio = tf.exp(pi.pd.logp(ac) - oldpi.pd.logp(ac)) # pnew / pold
    surr1 = ratio * atarg # surrogate from conservative policy iteration
    surr2 = tf.clip_by_value(ratio, 1.0 - clip_param, 1.0 + clip_param) * atarg #
    pol_surr = - tf.reduce_mean(tf.minimum(surr1, surr2)) # PPO's pessimistic surrogate (L^CLIP)
    vf_loss = tf.reduce_mean(tf.square(pi.vpred - ret))
    total_loss = pol_surr + pol_entpen + vf_loss
    losses = [pol_surr, pol_entpen, vf_loss, meankl, meanent]
    loss_names = ["pol_surr", "pol_entpen", "vf_loss", "kl", "ent"]

    pol_rac_loss = tf.reduce_mean(adv * pi.pd.neglogp(ac))
    pol_rac_losses = [pol_rac_loss]
    pol_rac_loss_names = ["pol_rac_loss"]

    vf_rac_loss = tf.reduce_mean(tf.square(pi.vpred - td_v_target))
    vf_rac_losses = [vf_rac_loss]
    vf_rac_loss_names = ["vf_rac_loss"]

    var_list = pi.get_trainable_variables()

    vf_final_var_list = [v for v in var_list if v.name.split("/")[1].startswith(
        "vf") and v.name.split("/")[2].startswith(
        "final")]
    pol_final_var_list = [v for v in var_list if v.name.split("/")[1].startswith(
        "pol") and v.name.split("/")[2].startswith(
        "final")]

    # Train V function
    vf_lossandgrad = U.function([ob, td_v_target, lrmult],
                                vf_rac_losses + [U.flatgrad(vf_rac_loss, vf_final_var_list)])
    vf_adam = MpiAdam(vf_final_var_list, epsilon = adam_epsilon)

    # Train Policy
    pol_lossandgrad = U.function([ob, ac, adv, lrmult],
                                 pol_rac_losses + [U.flatgrad(pol_rac_loss, pol_final_var_list)])
    pol_adam = MpiAdam(pol_final_var_list, epsilon = adam_epsilon)

    lossandgrad = U.function([ob, ac, atarg, ret, lrmult], losses + [U.flatgrad(total_loss, var_list)])
    adam = MpiAdam(var_list, epsilon=adam_epsilon)

    assign_old_eq_new = U.function([],[], updates=[tf.assign(oldv, newv)
        for (oldv, newv) in zipsame(oldpi.get_variables(), pi.get_variables())])
    compute_losses = U.function([ob, ac, atarg, ret, lrmult], losses)

    compute_v_pred = U.function([ob], [pi.vpred])

    U.initialize()
    adam.sync()
    pol_adam.sync()
    vf_adam.sync()

    global timesteps_so_far, episodes_so_far, iters_so_far, \
        tstart, lenbuffer, rewbuffer,best_fitness
    episodes_so_far = 0
    timesteps_so_far = 0
    iters_so_far = 0
    tstart = time.time()
    lenbuffer = deque(maxlen=100) # rolling buffer for episode lengths
    rewbuffer = deque(maxlen=100) # rolling buffer for episode rewards

    assert sum([max_iters>0, max_timesteps>0, max_episodes>0, max_seconds>0])==1, "Only one time constraint permitted"

    seg = None
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
            cur_lrmult =  max(1.0 - float(timesteps_so_far) / (max_timesteps/2), 0)
        else:
            raise NotImplementedError

        logger.log("********** Iteration %i ************"%iters_so_far)

        t = 0
        ac = env.action_space.sample() # not used, just so we have the datatype
        new = True # marks if we're on first timestep of an episode
        ob = env.reset()

        cur_ep_ret = 0 # return in current episode
        cur_ep_len = 0 # len of current episode
        ep_rets = [] # returns of completed episodes in this segment
        ep_lens = [] # lengths of ...
        horizon = timesteps_per_actorbatch

        # Initialize history arrays
        obs = np.array([ob for _ in range(horizon)])
        rews = np.zeros(horizon, 'float32')
        vpreds = np.zeros(horizon, 'float32')
        news = np.zeros(horizon, 'int32')
        acs = np.array([ac for _ in range(horizon)])
        prevacs = acs.copy()

        rac_alpha = optim_stepsize * cur_lrmult
        rac_beta = optim_stepsize * cur_lrmult * 0.1

        pol_gradients = []
        t_0 = 0

        for t in itertools.count():
            if timesteps_so_far % 10000 == 0 and timesteps_so_far > 0:
                result_record()
            prevac = ac
            ac, vpred = pi.act(stochastic=True, ob=ob)
            # Slight weirdness here because we need value function at time T
            # before returning segment [0, T-1] so we get the correct
            # terminal value
            if t > 0 and t % horizon == 0:
                seg = {"ob" : obs, "rew" : rews, "vpred" : vpreds, "new" : news,
                    "ac" : acs, "prevac" : prevacs, "nextvpred": vpred * (1 - new),
                    "ep_rets" : ep_rets, "ep_lens" : ep_lens}
                ep_rets = []
                ep_lens = []
                break
            i = t % horizon
            obs[i] = ob
            vpreds[i] = vpred
            news[i] = new
            acs[i] = ac
            prevacs[i] = prevac
            if env.spec._env_name == "LunarLanderContinuous":
                ac = np.clip(ac, -1.0, 1.0)
            next_ob, rew, new, _ = env.step(ac)

            # Compute v target and TD
            v_target = rew + gamma * np.array(compute_v_pred(next_ob.reshape((1, ob.shape[0]))))
            adv = v_target - np.array(compute_v_pred(ob.reshape((1, ob.shape[0]))))

            # Update V and Update Policy
            vf_loss, vf_g = vf_lossandgrad(ob.reshape((1, ob.shape[0])), v_target,
                                           rac_alpha)
            vf_adam.update(vf_g, rac_alpha)
            pol_loss, pol_g = pol_lossandgrad(ob.reshape((1, ob.shape[0])), ac.reshape((1, ac.shape[0])), adv,
                                              rac_beta)
            pol_gradients.append(pol_g)

            if t % update_step_threshold == 0 and t > 0:
                scaling_factor = [rho ** (t - i) for i in range(t_0, t)]
                coef = t/np.sum(scaling_factor)
                sum_weighted_pol_gradients = np.sum([scaling_factor[i] * pol_gradients[i] for i in range(len(scaling_factor))], axis = 0)
                pol_adam.update(coef*sum_weighted_pol_gradients, rac_beta)
                pol_gradients = []
                t_0 = t

            rews[i] = rew

            cur_ep_ret += rew
            cur_ep_len += 1
            timesteps_so_far+=1
            ob = next_ob
            if new:
                # Episode End Update
                scaling_factor = [rho ** (t - i) for i in range(t_0, t)]
                coef = t/np.sum(scaling_factor)
                sum_weighted_pol_gradients = np.sum([scaling_factor[i] * pol_gradients[i] for i in range(len(scaling_factor))], axis = 0)
                pol_adam.update(coef*sum_weighted_pol_gradients, rac_beta)
                pol_gradients = []
                t_0 = t
                print(
                    "Episode {} - Total reward = {}, Total Steps = {}".format(episodes_so_far, cur_ep_ret, cur_ep_len))
                ep_rets.append(cur_ep_ret)
                ep_lens.append(cur_ep_len)
                rewbuffer.extend(ep_rets)
                lenbuffer.extend(ep_lens)
                cur_ep_ret = 0
                cur_ep_len = 0
                ob = env.reset()
                episodes_so_far += 1
            t += 1

        add_vtarg_and_adv(seg, gamma, lam)

        # ob, ac, atarg, ret, td1ret = map(np.concatenate, (obs, acs, atargs, rets, td1rets))
        ob, ac, atarg, tdlamret = seg["ob"], seg["ac"], seg["adv"], seg["tdlamret"]
        vpredbefore = seg["vpred"] # predicted value function before udpate
        atarg = (atarg - atarg.mean()) / atarg.std() # standardized advantage function estimate
        d = Dataset(dict(ob=ob, ac=ac, atarg=atarg, vtarg=tdlamret), shuffle=not pi.recurrent)
        optim_batchsize = optim_batchsize or ob.shape[0]

        if hasattr(pi, "ob_rms"): pi.ob_rms.update(ob) # update running mean/std for policy

        assign_old_eq_new() # set old parameter values to new parameter values
        # logger.log("Optimizing...")
        # logger.log(fmt_row(13, loss_names))
        # Here we do a bunch of optimization epochs over the data
        for _ in range(optim_epochs):
            losses = [] # list of tuples, each of which gives the loss for a minibatch
            for batch in d.iterate_once(optim_batchsize):
                *newlosses, g = lossandgrad(batch["ob"], batch["ac"], batch["atarg"], batch["vtarg"], cur_lrmult)
                adam.update(g, optim_stepsize * cur_lrmult)
                losses.append(newlosses)
            # logger.log(fmt_row(13, np.mean(losses, axis=0)))
        # logger.log("Current Iteration Training Performance:" + str(np.mean(seg["ep_rets"])))
        iters_so_far += 1

def flatten_lists(listoflists):
    return [el for list_ in listoflists for el in list_]
