from utils import loading_data

# Loading datasets
train = loading_data.load_to_df_from_csv("../data/train.csv")
test = loading_data.load_to_df_from_csv("../data/test.csv")