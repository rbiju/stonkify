from torch.utils.data import Dataset, DataLoader

from stonkify.news import NewsRetriever


class NewsDataset(Dataset):
    def __init__(self, retriever: NewsRetriever):
        self.retriever = retriever

    def __len__(self):
        pass

    def __getitem__(self, item):
        pass
