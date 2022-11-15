import sys
sys.path.append("..")

from utils.loading_data  import load_to_df_from_csv
from utils.plotting import draw_histogram

# Loading datasets
test = load_to_df_from_csv("../data/test.csv")
train = load_to_df_from_csv("../data/train.csv")

# Histogram of movies per year
draw_histogram(train.Year, "Year", "Movies", "Movies Across Years")