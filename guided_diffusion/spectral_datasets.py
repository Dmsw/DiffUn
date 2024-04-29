from mpi4py import MPI
import numpy as np
from torch.utils.data import DataLoader, Dataset

def load_spectral_data(
        *,
        data_dir,
        batch_size,
        deterministic=False,
        pad=False,
):
    if not data_dir:
        raise ValueError("unspecified data directory")
    dataset = SpectralDataset(
        data=np.load(data_dir)["A"].astype(np.float32),
        shard=MPI.COMM_WORLD.Get_rank(),
        num_shards=MPI.COMM_WORLD.Get_size(),
        pad=pad,
    )
    if not deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=False
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=False
        )
    while True:
        yield from loader


class SpectralDataset(Dataset):
    def __init__(self,
                 data,
                 shard=0,
                 num_shards=1,
                 pad=False,
                 ):
        super().__init__()
        self.data = data[shard:][::num_shards]
        self.pad = pad

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        spectral = self.data[idx][None]
        spectral = spectral * 2 - 1
        return {"input": spectral}, {}
