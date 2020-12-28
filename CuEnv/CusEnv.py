import gym
from Estimator.estimate import *
from typing import *
# import random


class CustomEnv(gym.Env):

    def __init__(self, env: gym.Env, es1: Estimator, es2: Estimator, es3: Estimator,
                 alpha=0.5, beta=0.5, clip_min=None, clip_max=None, default_max=1e5):
        """
        主要用来外包一个环境
        用以做一个封装环境来进行一定来，主要用来进行反向强化学习
        :param env: gym的原生环境
        :param es1: 概率模型估计器1，用以估计\pi_e(s, a)
        :param es2: 概率模型估计器2 用以估计\pi_e(s)
        :param es3: 概率模型估计器3 用以估计\pi(s)
        :param clip_min: 用以做奖励的reward_shape
        :param clip_max: 用以做奖励的reward_shape
        """
        super(CustomEnv, self).__init__()
        # 基础属性与封装前的属性一直
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self.metadata = env.metadata
        # 需要记录的数据有
        self.env = env
        self.current_obs = None
        # 状态动作估计器
        self.es1: Estimator = es1
        # 状态估计器
        self.es2: Estimator = es2
        # 时间
        self.es3: Estimator = es3
        self.it = 0
        # r的两个截断范围
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.default_max = default_max
        self.alpha = alpha
        self.beta = beta
        # random.seed(0)
        # just for debug
        self.nas = 0

    def reset(self):
        obs = self.env.reset()
        self.current_obs = obs
        return obs

    def step(self, action):
        next_obs, t_r, done, info = self.env.step(action)
        self.it = self.it + 1
        # 由于反向强化学习，只需要设计r
        # 将真实的r反馈出去
        info['t_r'] = t_r
        # todo: 环境的规一化的问题，主要是观测到的变量的规一化的问题
        if isinstance(self.es1, SVMEstimator):
            p_s_a: float = self.es1.predict([self.current_obs, action]) * self.es3.predict([self.current_obs])
        else:
            p_s_a: float = self.es1.predict([self.current_obs, action])
        p_s: float = self.alpha * self.es2.predict([self.current_obs]) + self.beta * self.es3.predict([self.current_obs])
        # 这里相除可能会得到一个除0的warning, 最主要的原因就是这个p_s可能过小, 为了解决这个问题，我们需要做一个异常处理机制
        if p_s > 0:
            d_r = p_s_a / p_s
        elif p_s == 0 and p_s_a > 0:
            d_r = self.default_max
        else:
            d_r = 10
            self.nas = self.nas + 1
        if self.clip_max is not None:
            d_r = min(self.clip_max, d_r)
        if self.clip_min is not None:
            d_r = max(self.clip_min, d_r)
        self.current_obs = next_obs
        return next_obs, d_r, done, info

    def seed(self, seed=None):
        self.env.seed(seed)

    def render(self, mode='human'):
        pass


"""
    def train_es1(self, data: List[List]):
        #  es1的输入是[[s_1, a_1], [s_2, a_2] ...]等等
        self.es1.train(data)

    def train_es2(self, data: List[np.ndarray]):
        # es2的输入是[s_1, s_2 ...]
        self.es2.train(data)

    def train_es3(self, data: List[np.ndarray]):
        self.es3.train(data)

"""