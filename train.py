from src.dataset import get_dataset
from src.ebm import CD_EBM
from torch.utils.data import DataLoader

if __name__ == '__main__':

    train_dataset, _ = get_dataset()
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=128)

    SGLD_KWARGS=dict(a=0.02, b=0.0001, gamma=None)
    # No classifier loss here, since alpha == 0.0.
    ebm = CD_EBM(
        x_dim=2, 
        gen_kwargs=dict(n_h=256),
        stochastic=True,
        #gen_kwargs={},
        sgld_kwargs=SGLD_KWARGS,
        sgld_n_steps=100,
        std=0.100
    )
    ebm.train(train_loader, n_epochs=100, use_tqdm=False)
