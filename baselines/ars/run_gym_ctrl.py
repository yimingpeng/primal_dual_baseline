#!/usr/bin/env python3
# Add the current folder to PYTHONPATH by Yiming
import os
import sys

sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.abspath(os.path.join(os.getcwd(),os.pardir)), os.pardir)))

from baselines.common.cmd_util import gym_ctrl_arg_parser, make_gym_control_env
from baselines.common import tf_util as U, set_global_seeds
from baselines import logger

def train(env_id, num_timesteps, seed):
    from baselines.ars import ars
    main_loop_size = 1000
    horizon = 1000
    step_size = 0.005
    noise = 0.03
    hp = ars.Hp(main_loop_size, horizon, num_timesteps, step_size, noise)
    set_global_seeds(seed)
    env = make_gym_control_env(env_id, seed)
    # env = wrappers.Monitor(env, monitor_dir, force=True)
    num_inputs = env.observation_space.shape[0]
    num_outputs = env.action_space.shape[0]
    policy = ars.Policy(num_inputs, num_outputs, hp)
    normalizer = ars.Ob_Normalizer(num_inputs)
    ars.train(env, policy, normalizer, hp)
    env.close()

def main():
    args = gym_ctrl_arg_parser().parse_args()
    logger.configure(format_strs=['stdout', 'log', 'csv'], log_suffix = "ARS-"+args.env)
    logger.log("Algorithm: ARS-"+args.env)
    import random
    args.seed += random.randint(0, 2**32 - 1)
    logger.log("Algorithm: SEED-"+str(args.seed))
    train(args.env, num_timesteps=args.num_timesteps, seed=args.seed)

if __name__ == '__main__':
    main()
