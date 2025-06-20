import kagglehub
from kagglehub import KaggleDatasetAdapter
import os
os.environ["KAGGLE_CONFIG_DIR"] = "/path/to/your/.kaggle"


# Load a DataFrame with a specific version of a CSV
df = kagglehub.dataset_load(
    KaggleDatasetAdapter.PANDAS,
    "unsdsn/world-happiness/versions/1",
    "2016.csv",
)

# Load a DataFrame with specific columns from a parquet file
df = kagglehub.dataset_load(
    KaggleDatasetAdapter.PANDAS,
    "robikscube/textocr-text-extraction-from-images-dataset",
    "annot.parquet",
    pandas_kwargs={"columns": ["image_id", "bbox", "points", "area"]}
)

# Load a dictionary of DataFrames from an Excel file where the keys are sheet names 
# and the values are DataFrames for each sheet's data. NOTE: As written, this requires 
# installing the default openpyxl engine.
df_dict = kagglehub.dataset_load(
    KaggleDatasetAdapter.PANDAS,
    "theworldbank/education-statistics",
    "edstats-excel-zip-72-mb-/EdStatsEXCEL.xlsx",
    pandas_kwargs={"sheet_name": None},
)

# Load a DataFrame using an XML file (with the natively available etree parser)
df = dataset_load(
    KaggleDatasetAdapter.PANDAS,
    "parulpandey/covid19-clinical-trials-dataset",
    "COVID-19 CLinical trials studies/COVID-19 CLinical trials studies/NCT00571389.xml",
    pandas_kwargs={"parser": "etree"},
)

# Load a DataFrame by executing a SQL query against a SQLite DB
df = kagglehub.dataset_load(
    KaggleDatasetAdapter.PANDAS,
    "wyattowalsh/basketball",
    "nba.sqlite",
    sql_query="SELECT person_id, player_name FROM draft_history",
)