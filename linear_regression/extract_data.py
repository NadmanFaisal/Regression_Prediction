import pandas as pd
import numpy as np

class extract_data:
    def __init__(self):
        pass

    def convert_to_array(self, path):
        df = pd.read_csv(path)
        df = df.select_dtypes(include='number')
        dp = df.to_numpy()
        return dp
