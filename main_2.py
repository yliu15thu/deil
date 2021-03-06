from DEIL.deil import *
import os
import tensorflow as tf
import warnings
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# warnings.filterwarnings('ignore', module='TensorFlow')
# warnings.filterwarnings('ignore', module='gym')
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


if __name__ == '__main__':
    config = {
        'env_name': 'Acrobot-v1',
        'collect_epochs': 5,
        'collect_seed': 9,
        'es1_type': 'svm',
        'seed': 0,
        'train_epochs': 100,
        'epoch_steps': 1000,
        'interact_steps': 3000,
        'model': 'ppo',
        'sub_sample': False,
        'save_path': './result/Acrobot_epoch5.pkl',
        'alpha': 0.5,
        'beta': 0.5
    }
    expert_data = expert_collect(config['env_name'], config['collect_epochs'], seed=config['collect_seed'])
    print(len(expert_data))
    deil_model = DEIL(config)
    deil_model.train(expert_data)
