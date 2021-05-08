from simple_worm.trainer import Trainer


def train_inv(
        optim_e0=False,
        optim_abg=False,
        N=4,
        T=0.2,
        dt=0.1,
        lr=0.5,
        n_steps=1,
        inverse_opt_max_iter=1,
        save_videos=False,
        save_plots=False
):
    print('\n==== Test Inverse Trainer ===')
    print(f'optim_e0={optim_e0}, optim_abg={optim_abg}\n')

    trainer = Trainer(
        N=N,
        T=T,
        dt=dt,
        optim_e0=optim_e0,
        optim_abg=optim_abg,
        target_params={'alpha_pref_freq': 1, 'beta_pref_freq': 0.5},
        lr=lr,
        reg_weights={},
        inverse_opt_max_iter=inverse_opt_max_iter,
        save_videos=save_videos,
        save_plots=save_plots,
    )
    trainer.train(n_steps)
    print('done')


default_args = {
    'optim_e0': False,
    'optim_abg': False,
    'N': 20,
    'dt': 0.1,
    'T': 0.3,
    'lr': 1e-3,
    'n_steps': 1,
    'inverse_opt_max_iter': 1,
    'save_videos': False,
    'save_plots': False,
}


def test_inv_trainer_e0():
    train_inv(**{**default_args, 'optim_e0': True})


def test_inv_trainer_abg():
    train_inv(**{**default_args, 'optim_abg': True})


def test_inv_trainer_both():
    train_inv(**{**default_args, 'optim_e0': True, 'optim_abg': True})


def test_inv_trainer_neither():
    train_inv(**default_args)


if __name__ == '__main__':
    test_inv_trainer_neither()
    test_inv_trainer_e0()
    test_inv_trainer_abg()
    test_inv_trainer_both()
