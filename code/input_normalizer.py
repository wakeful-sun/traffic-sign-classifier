import pandas as pd
import numpy as np
import cv2


class InputNormalizer:

    def __init__(self):
        """
        Normalizer
        """

    @staticmethod
    def _shuffle(data):
        return data.sample(frac=1).reset_index(drop=True)

    def normalize_by_amount(self, labels, images):
        data = list(zip(labels, images))
        df = pd.DataFrame(data=data, columns=["category", "sign_image"])
        shuffled_df = self._shuffle(df)

        df_categories = shuffled_df.groupby(["category"])
        min_category_amount = int(df_categories.count().min())

        amount_normalized_categories = list()

        for category, category_items in df_categories:
            amount_normalized_categories.append(category_items.head(min_category_amount))

        amount_normalized_df = pd.concat(amount_normalized_categories)
        amount_normalized_shuffled_df = self._shuffle(amount_normalized_df)

        return np.array(list(amount_normalized_shuffled_df["category"])), np.array(list(amount_normalized_shuffled_df["sign_image"]))
