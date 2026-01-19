from data_prepare import load_lamah_daily

DATA_ROOT = "/home/lisiwu/jxwork/1-gnn-lstm/dataset"

precip_df, temp_df, soil_df, runoff_df, static_df = load_lamah_daily(DATA_ROOT)

print("precip:", precip_df.shape)
print("temp:  ", temp_df.shape)
print("soil:  ", soil_df.shape)
print("runoff:", runoff_df.shape)
print("static:", static_df.shape)

print("date range:", precip_df.index.min(), precip_df.index.max())
print("first basins:", static_df.index[:5])
