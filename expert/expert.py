"""
这份代码用于训练专家模型 收集专家样本
"""
from stable_baselines3.common.base_class import BaseAlgorithm
import gym
from stable_baselines3 import SAC,PPO
from stable_baselines3.sac import MlpPolicy
from stable_baselines3.ppo import MlpPolicy
from typing import *
from test_model.test_model import test_model
import pickle


class Expert:

    @staticmethod
    def train_expert(config: Dict) -> None:
        """
        训练专家模型, 支持两种baseline算法训练专家模型
        :param config: 训练参数
        :return:
        """
        model = None
        env: gym.Env = config['env']
        if config['model'] == 'SAC':
            model = SAC(MlpPolicy, env, verbose=config['verbose'], seed=config['seed'])
        elif config['model'] == 'PPO':
            model = PPO(MlpPolicy, env, verbose=config['verbose'], seed=config['seed'])
        else:
            print('尚未实现其他算法')
            exit(1)
        # 学习前测试一下策略模型
        init_score = test_model(env, model)
        print(f'训练前测试模型的扥分是 {init_score}')
        model.learn(total_timesteps=config['total_time_steps'])
        train_score = test_model(env, model)
        print(f'训练后的专家得分是 {train_score}')
        if config['save']:
            model.save(config['path'])

    @staticmethod
    def collect_experience(config: Dict):
        """
        收集专家样本
        :param config:
        :return:
        """
        model: BaseAlgorithm = config['model']
        env: gym.Env = config['env']
        env.seed(config['seed'])
        obs = env.reset()
        step = 0
        experience: list = []
        while step < config['collect_nums']:
            action, _ = model.predict(obs)
            next_obs, _, done, _ = env.step(action)
            step = step + 1
            experience.append((obs, action))
            if not done:
                obs = next_obs
            else:
                obs = env.reset()
        with open(config['save_path']) as f:
            pickle.dump(experience, f)


