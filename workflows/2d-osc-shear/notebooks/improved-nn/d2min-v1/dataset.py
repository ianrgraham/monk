# from torch.utils.data import Dataset
# import pathlib

# class GlassDataset(Dataset):
#     def __init__(self, path: str):
#         # get full path
#         path = pathlib.Path(path)
#         self._path = path.resolve()
#         self._files = list(path.glob("*/*"))
#         self._len = len(self._files)

#     def __len__(self):
#         return self._len
    
#     def __getitem__(self, idx):
#         return self._files[idx]