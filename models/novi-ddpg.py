#!/usr/bin/env python

import gym
import gym_carla
import carla
import numpy as np
from stable_baselines3 import TD3
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import SAC
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3 import A2C
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3 import DDPG
import torch as th
import torch.nn as nn
from stable_baselines3.common.preprocessing import is_image_space


class CustomCombinedExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict):
        # We do not know features-dim here before going over all the items,
        # so put something dummy for now. PyTorch requires calling
        # nn.Module.__init__ before adding modules
        super(CustomCombinedExtractor, self).__init__(observation_space, features_dim=1)

        extractors = {}

        total_concat_size = 0
        # We need to know size of the output of this extractor,
        # so go over all the spaces and compute output feature sizes
        for key, subspace in observation_space.spaces.items():
            if key == "camera":
                # We will just downsample one channel of the image by 4x4 and flatten.
                # Assume the image is single-channel (subspace.shape[0] == 0)
                extractors[key] = nn.Sequential(nn.MaxPool2d(4), nn.Flatten())
                total_concat_size += subspace.shape[1] // 4 * subspace.shape[2] // 4
            elif key == "state":
                # Run through a simple MLP
                extractors[key] = nn.Linear(0, 16)
                total_concat_size += 16

        self.extractors = nn.ModuleDict(extractors)

        # Update the features dim manually
        self._features_dim = total_concat_size

    def forward(self, observations) -> th.Tensor:
        encoded_tensor_list = []

        # self.extractors contain nn.Modules that do all the processing.
        for key, extractor in self.extractors.items():
            encoded_tensor_list.append(extractor(observations[key]))
        # Return a (B, self._features_dim) PyTorch tensor, where B is batch dimension.
        return th.cat(encoded_tensor_list, dim=1)

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
    'continuous_accel_range': [1,2],  # continuous acceleration range
    'continuous_steer_range': [-0.3, 0.3],  # continuous steering angle range
    'ego_vehicle_filter': 'vehicle.lincoln*',  # filter for defining ego vehicle
    'port': 2000,  # connection port
    'town': 'Town03',  # which town to simulate
    'task_mode': 'random',  # mode of the task, [random, roundabout (only for Town03)]
    'max_time_episode': 1000,  # maximum timesteps per episode
    'max_waypt': 12,  # maximum number of waypoints
    'obs_range': 32,  # observation range (meter) bilo je 32
    'lidar_bin': 0.5,  # bin size of lidar sensor (meter) # bilo je 0.125
    'd_behind': 12,  # distance behind the ego vehicle (meter)
    'out_lane_thres': 2.0,  # threshold for out of lane
    'desired_speed': 8,  # desired speed (m/s)
    'max_ego_spawn_times': 200,  # maximum times to spawn ego vehicle
    'display_route': True,  # whether to render the desired route
    'pixor_size': 64,  # size of the pixor labels
    'pixor': False,  # whether to output PIXOR observation
  }

  policy_kwargs = dict(
    features_extractor_class=CustomCombinedExtractor,
  )

  env = gym.make('carla-v0', params=params)
  print(env.observation_space);
  print(is_image_space(env.observation_space.spaces['camera']))
  print(is_image_space(env.observation_space.spaces['lidar']))


  n_actions = env.action_space.shape[-1]
  action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

  model = DDPG("MultiInputPolicy", env, verbose=1,buffer_size=10000, tensorboard_log="./zavrsni-usporedbe/",action_noise = action_noise);
  model.learn(total_timesteps=25000,log_interval=1)

  model.save("modeli/DDPG")
  model = DDPG.load("modeli/DDPG",env=env,verbose=1,buffer_size=10000, tensorboard_log="./zavrsni-modeli/",action_noise = action_noise);
  # model.save("td3_poliranje")
  # model = TD3.load("td3_poliranje",env=env,tensorboard_log="./td3_tensorboard/");
  # model.learn(total_timesteps=10000,log_interval=10);

  # model = TD3.load("td3_mountain");
  # model.set_env(env);
  # model.learn(total_timesteps=50000,log_interval=10);

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
