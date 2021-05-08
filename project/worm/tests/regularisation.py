from simple_worm.trainer import Trainer


def train_inv_reg(
        optim_e0=False,
        optim_abg=False,
        reg_weights={},
        N=4,
        T=0.2,
        dt=0.1,
        lr=0.5,
        n_steps=1,
        inverse_opt_max_iter=1,
        save_videos=False,
        save_plots=False

):
    print('\n==== Test Regularisation ===')
    print(f'reg_weights={reg_weights}\n')

    trainer = Trainer(
        N=N,
        T=T,
        dt=dt,
        optim_e0=optim_e0,
        optim_abg=optim_abg,
        target_params={'alpha_pref_freq': 1, 'beta_pref_freq': 0.5},
        lr=lr,
        reg_weights=reg_weights,
        inverse_opt_max_iter=inverse_opt_max_iter,
        save_videos=save_videos,
        save_plots=save_plots,
    )
    trainer.train(n_steps)
    print('done')


default_args = {
    'optim_e0': True,
    'optim_abg': True,
    'reg_weights': {},
    'N': 20,
    'dt': 0.1,
    'T': 0.3,
    'lr': 1e-3,
    'n_steps': 1,
    'inverse_opt_max_iter': 1,
    'save_videos': False,
    'save_plots': False,
}


def test_inv_reg_L2():
    rw = {
             'L2': {
                 'alpha': 1e-7,
                 'beta': 1e-7,
                 'gamma': 1e-4,
             }
         }
    train_inv_reg(**{**default_args, 'reg_weights': rw})


def test_inv_reg_grad_t():
    rw = {
             'grad_t': {
                 'alpha': 1e-8,
                 'beta': 1e-8,
                 'gamma': 1e-7,
             },
         }
    train_inv_reg(**{**default_args, 'reg_weights': rw})


def test_inv_reg_grad_x():
    rw = {
             'grad_x': {
                 'alpha': 1e-8,
                 'beta': 1e-8,
             }
         }
    train_inv_reg(**{**default_args, 'reg_weights': rw})


def test_inv_reg_all():
    rw = {
             'L2': {
                 'alpha': 1e-7,
                 'beta': 1e-7,
                 'gamma': 1e-4,
             },
             'grad_t': {
                 'alpha': 1e-8,
                 'beta': 1e-8,
                 'gamma': 1e-7,
             },
             'grad_x': {
                 'alpha': 1e-8,
                 'beta': 1e-8,
             }
         }
    train_inv_reg(**{**default_args, 'reg_weights': rw})


if __name__ == '__main__':
    test_inv_reg_L2()
    test_inv_reg_grad_t()
    test_inv_reg_grad_x()
    test_inv_reg_all()
