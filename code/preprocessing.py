import polars as pl
import sys
import os

def create_config(test_ratio_pct: float, start_date: int) -> object:
    class CONFIG:
        target_col = "responder_6"
        lag_cols_original = ["date_id", "symbol_id"] + [f"responder_{idx}" for idx in range(9)]
        lag_cols_rename = { f"responder_{idx}" : f"responder_{idx}_lag_1" for idx in range(9)}
        test_ratio = test_ratio_pct
        start_dt = start_date

    return CONFIG()

def scan_training_data(path: str, CONFIG: object) -> pl.LazyFrame:
    train = pl.scan_parquet(
        path,
        ).select(
            pl.int_range(pl.len(), dtype=pl.UInt32).alias("id"),
            pl.all(),
        ).with_columns(
            (pl.col(CONFIG.target_col)*2).cast(pl.Int32).alias("label"),
        ).filter(
            pl.col("date_id").gt(CONFIG.start_dt)
        ).drop([f"responder_{i}" for i in range(9)])   
    return train

def create_lags(train: pl.LazyFrame, CONFIG: object) -> pl.LazyFrame:
    lags = train.select(pl.col(CONFIG.lag_cols_original))
    lags = lags.rename(CONFIG.lag_cols_rename)
    lags = lags.with_columns(
        date_id = pl.col('date_id') + 1,  # lagged by 1 day
        )
    lags = lags.group_by(["date_id", "symbol_id"], maintain_order=True).last()  # pick up last record of previous date
    return lags

def merge_train_lags(train: pl.LazyFrame, lags: pl.LazyFrame) -> pl.LazyFrame:
    train = train.join(lags, on=["date_id", "symbol_id"],  how="left")
    return train

def create_training_test_sets(train: pl.LazyFrame, CONFIG: object) -> pl.LazyFrame:
    len_train   = train.select(pl.col("date_id")).collect().shape[0]
    test_records = int(len_train * CONFIG.test_ratio)
    len_ofl_mdl = len_train - test_records
    last_tr_dt  = train.select(pl.col("date_id")).collect().row(len_ofl_mdl)[0]

    print(f"\n len_train = {len_train-test_records}")
    print(f" len_test = {test_records}")

    training_data = train.filter(pl.col("date_id").le(last_tr_dt))
    test_data  = train.filter(pl.col("date_id").gt(last_tr_dt))

    return training_data, test_data

def save_preprocessed_data(training_data: pl.LazyFrame, test_data: pl.LazyFrame) -> None:
    if not os.path.isdir("../preprocessed_data"):
        os.mkdir("../preprocessed_data")
    training_data.collect().write_parquet("../preprocessed_data/training.parquet")
    test_data.collect().write_parquet("../preprocessed_data/test.parquet")

if __name__ == "__main__":
    sys.path.append(".")

    CONFIG = create_config(test_ratio_pct=0.1, start_date= 1550) # for approx 5.5M rows

    train = scan_training_data(path = "../raw_data/train_parquet/", CONFIG = CONFIG)
    
    #lags = create_lags(train, CONFIG)

    #train = merge_train_lags(train, lags)

    training_data, test_data = create_training_test_sets(train, CONFIG)

    save_preprocessed_data(training_data, test_data)

    print("\n---> Preprocessing done\n")
    print("\n---> Training and Test data saved in preprocessed_data/\n")