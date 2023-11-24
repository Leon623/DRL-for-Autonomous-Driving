from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common import make_vec_env
from stable_baselines import PPO2
# !/usr/bin/env python

# Copyright (c) 2019: Jianyu Chen (jianyuchen@berkeley.edu).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import gym
import gym_carla
import carla
import numpy as np;


def main():
    # parameters for the gym_carla environment
    params = {
        'number_of_vehicles': 1,
        'number_of_walkers': 0,
        'display_size': 256,  # screen size of bird-eye render
        'max_past_step': 1,  # the number of past steps to draw
        'dt': 0.1,  # time interval between two frames
        'discrete': False,  # whether to use discrete control space
        'discrete_acc': [-2.0, 0.0, 2.0],  # discrete value of accelerations
        'discrete_steer': [-0.3, 0.0, 0.3],  # discrete value of steering angles
        'continuous_accel_range': [0, 3.0],  # continuous acceleration range
        'continuous_steer_range': [-0.8, 0.8],  # continuous steering angle range
        'ego_vehicle_filter': 'vehicle.lincoln*',  # filter for defining ego vehicle
        'port': 2000,  # connection port
        'town': 'Town03',  # which town to simulate
        'task_mode': 'random',  # mode of the task, [random, roundabout (only for Town03)]
        'max_time_episode': 1000,  # maximum timesteps per episode
        'max_waypt': 12,  # maximum number of waypoints
        'obs_range': 32,  # observation range (meter) bilo je 32
        'lidar_bin': 0.125,  # bin size of lidar sensor (meter) # bilo je 0.125
        'd_behind': 12,  # distance behind the ego vehicle (meter)
        'out_lane_thres': 2.0,  # threshold for out of lane
        'desired_speed': 8,  # desired speed (m/s)
        'max_ego_spawn_times': 200,  # maximum times to spawn ego vehicle
        'display_route': True,  # whether to render the desired route
        'pixor_size': 64,  # size of the pixor labels
        'pixor': False,  # whether to output PIXOR observation
    }

    # Set gym-carla environment
    env = gym.make('carla-v0', params=params)
    env = gym.wrappers.FlattenDictWrapper(env, dict_keys=['state'])


    model = PPO2(MlpPolicy, env, verbose=1,tensorboard_log="./tensor_usporedba/");
    model.learn(total_timesteps=25000,log_interval=10);

    model.save("modeli/PPO2")
    # model.save("td3_poliranje")
    # model = TD3.load("td3_poliranje",env=env,tensorboard_log="./td3_tensorboard/");
    # model.learn(total_timesteps=10000,log_interval=10);

    # model = TD3.load("td3_mountain");
    # model.set_env(env);
    # model.learn(total_timesteps=50000,log_interval=10);
    model = DDPG.load("modeli/PPO2");

    obs = env.reset()
    print("done");
    while True:
        action, _states = model.predict(obs);
        # print(action)
        obs, r, done, info = env.step(action);
        # env.render();
        if (done):
            obs = env.reset();

    env.close();

    # if done:
    # obs = env.reset()


if __name__ == '__main__':
    main()

