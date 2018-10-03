#!/usr/bin/env python3
import inspect
import os
import sys

sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.abspath(os.path.join(os.getcwd(), os.pardir)), os.pardir)))
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0,parentdir)
# very important, don't remove, otherwise pybullet cannot run (reasons are unknown)
import pybullet_envs
from baselines.common.cmd_util import pybullet_arg_parser, make_pybullet_env
from baselines.common import tf_util as U
from baselines import logger


def train(env_id, num_timesteps, seed):
    from baselines.ppo_dual_nac_advantage import mlp_policy, pposgd_simple
    U.make_session(num_cpu=1).__enter__()
    def policy_fn(name, ob_space, ac_space):
        return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
            hid_size=64, num_hid_layers=2)

    env = make_pybullet_env(env_id, seed)
    pposgd_simple.learn(env,policy_fn,
            max_timesteps=num_timesteps,
            timesteps_per_actorbatch=2048,
            clip_param=0.2, entcoeff=0.0,
            optim_epochs=10, optim_stepsize=3e-4, optim_batchsize=64,
            gamma=0.99, lam=0.95,
            rho = 0.95,  # Gradient weighting factor
            update_step_threshold = 25, # Updating step threshold
                        schedule='linear'
        )
    env.close()

def main():
    args = pybullet_arg_parser().parse_args()
    logger.configure(
                     format_strs=['stdout', 'log', 'csv'], log_suffix = "PPO_Dual_NAC_Advantage-"+args.env)
    logger.log("Algorithm: PPO_Dual_NAC_Advantage-"+args.env)
    train(args.env, num_timesteps=args.num_timesteps, seed=args.seed)

if __name__ == '__main__':
    main()