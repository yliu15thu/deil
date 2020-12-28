
"""实现deil算法"""
import gym
# 这个只是为了导入更加合适的模型
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.ppo2 import PPO2
from CuEnv.CusEnv import CustomEnv
from test_model.test_model import test_model
# 强化学习的算法
from stable_baselines3.sac.policies import MlpPolicy as sMlp
from stable_baselines3.sac import SAC
from stable_baselines3.ppo.policies import MlpPolicy as pMlp
from stable_baselines3.ppo import PPO

from Estimator.estimate import *
# 这个主要是为了导入更加合理的环境
from pybullet_envs.bullet import *
from matplotlib import pyplot as plt
import pickle 

def expert_collect(env_name: str, epochs: int, seed=0, sub_sample=False, sample_freq=None) -> List:
    res_list = []
    env: gym.Env = gym.make(env_name)
    model_path: str = './expert_model/' + env_name + '.pkl'
    model: PPO2 = PPO2.load(model_path, seed=0)
    expert_score = test_model(env, model)
    print(f'专家得分 {expert_score}')
    env.seed(seed)
    for _ in range(epochs):
        obs = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs)
            next_obs, r, done, _ = env.step(action)
            res_list.append([obs, action])
            obs = next_obs
    if sub_sample:
        sample_index = list(range(0, len(res_list), sample_freq))
        return [res_list[item] for item in sample_index]
    return res_list


class DEIL:

    def __init__(self, config):
        self.config = config

    @staticmethod
    def inter_act(env:gym.Env, model, nums: int, seed=None) -> List[List]:
        """
               :param env: 交互的环境
               :param model: 交互的模型
               :param nums: 交互的步的数量
               :param seed: 交互的种子
               :return: List[obs, action, d_r, t_r]
        """
        if seed is not None:
            env.seed(seed)
        step = 0
        obs = env.reset()
        res = []
        while step < nums:
            action, _ = model.predict(obs)
            # 这里采样的时候也能够隔点采样，能够使得更加平滑
            next_obs, d_r, done, info = env.step(action)
            step = step + 1
            t_r = info['t_r']
            res.append([obs, action, d_r, t_r])
            if done:
                obs = env.reset()
            else:
                obs = next_obs
        return res

    def train(self, expert_data: List[List]):
        origin_env: gym.Env = gym.make(self.config['env_name'])
        if self.config['es1_type'] == 'svm':
            es1 = SVMEstimator()
        elif self.config['es1_type'] == 'gauss':
            es1 = GaussEstimator()
        else:
            print('尚未实现其他估计器')
            exit(2)
            return
        es2 = GaussEstimator()
        es3 = GaussEstimator()
        cus_env: CustomEnv = CustomEnv(origin_env, es1, es2, es3, alpha=self.config['alpha'], beta=self.config['beta'])
        # 准备好两个容器
        # 开始训练之前，我们先训练好我们定制化环境的估计器
        es1.train(expert_data)
        len_expert: int = len(expert_data)
        expert_state: List[np.ndarray] = [item[0] for item in expert_data]
        es2.train(expert_state)
        es3.train(expert_state)
        seed = self.config['seed']
        train_epochs = self.config['train_epochs']
        epoch_steps = self.config['epoch_steps']
        interact_steps = len_expert
        if self.config['model'] == 'ppo':
            # 这里需要有PPO参数
            # todo: 优雅的传入强化学习形式参数，目前的话，还是只能够修改脚本
            model: PPO = PPO(pMlp, env=cus_env, seed=0, verbose=0)
        elif self.config['model'] == 'SAC':
            model: SAC = SAC(sMlp, env=cus_env, seed=seed, verbose=0, device='cuda', learning_rate=0.0001)
        else:
            print('尚未实现其他算法')
            exit(2)
            return
        # 测试环境 model
        f = open(self.config['save_path'], 'wb')
        data = []
        init_score = test_model(origin_env, model)
        data.append((init_score, 0))
        f2 = open('./reward/' + self.config['save_path'][9:], 'wb')
        r_info = []
        print(f'最初模型得分:{init_score}')
        for i in range(train_epochs):
            # 当前策略与环境交互，学出真正状态分布
            interact_steps = self.config['interact_steps']
            interact_data = self.inter_act(cus_env, model, interact_steps)
            interact_state:list = [item[0] for item in interact_data]
            d_r_s: List = [item[2] for item in interact_data]
            t_r_s: List = [item[3] for item in interact_data]
            r_info.append((d_r_s, t_r_s))
            # print(np.corrcoef(d_r_s, t_r_s))
            # maybe 我们能够更加适合极限的去做这个问题，当我们在try的时候，这些try的样本也是可以进行利用和收集的
            # cus_env.train_es2(sample_state + expert_state)
            es3.train(interact_state)
            # 学习一段时间
            model.learn(epoch_steps)
            # 这个时候需要更新环境，同时我们要清空之前的学习经验，因为我们会有新的估计函数
            epoch_score = test_model(origin_env, model)
            data.append((epoch_score, (i + 1) * epoch_steps))
            print(f'目前的epoch是 {i}, 目前的得分是 {epoch_score}')
            if isinstance(model, SAC):
                # 这里我们也许可以不用把之前的经验全部忘掉， 我们可以把之前所有的奖励进行一个修改
                # 这也许是更加适合的方案，这一步可以不浪费以前的探索， 而且这样做有很大的弊端
                model.replay_buffer.reset()
        pickle.dump(data, f)
        f.close()
        pickle.dump(r_info, f2)
        f2.close()        
