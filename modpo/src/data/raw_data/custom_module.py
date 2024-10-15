from src.data.raw_data import RawDatasetPreprocessor
from datasets import load_dataset

class CustomRDP(RawDatasetPreprocessor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def get_data(self):
        dataset = load_dataset("marulyanova/PKU-SafeRLHF-10K-Modified")
        return dataset
    
    def preprocess(self, dataset):
        return {
            "prompt": dataset["prompt"],
            "response_0": dataset["response_0"],
            "response_1": dataset["response_1"],
            "response_2": dataset["response_2"],
            "xcomet_best_response_id": dataset["xcomet_best_response_id"],
            "kiwi_best_response_id": dataset["kiwi_best_response_id"],
            "fluency_best_response_id": dataset["fluency_best_response_id"],
        }

    def get_preference_dataset(self, split):
        dataset = self.get_data()
        processed_data = self.preprocess(dataset[split])
        return processed_data
