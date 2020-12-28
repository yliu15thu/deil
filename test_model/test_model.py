"""
测试模型的好坏
"""

import gym


def test_model(env: gym.Env, model, seed=10, epoch=10):
    env.seed(seed)
    r_s = []
    for _ in range(epoch):
        obs = env.reset()
        done = False
        reward = 0
        while not done:
            action, _ = model.predict(obs)
            next_obs, r, done, _ = env.step(action)
            reward = reward + r
            obs = next_obs
        r_s.append(reward)
    return sum(r_s) / epoch