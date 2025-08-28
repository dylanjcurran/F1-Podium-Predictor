import pandas as pd

races = pd.read_csv("../data/original_data/races.csv")
results = pd.read_csv("../data/original_data/results.csv")
qualifying = pd.read_csv("../data/original_data/qualifying.csv", low_memory=False)
drivers = pd.read_csv("../data/original_data/drivers.csv")
constructors = pd.read_csv("../data/original_data/constructors.csv")
circuits = pd.read_csv("../data/original_data/circuits.csv")
# pit_stops = pd.read_csv("../data/original_data/pit_stops.csv")
# lap_times = pd.read_csv("../data/original_data/lap_times.csv")

# ---- Keep Essential Collumns
races_small = races[["raceId","year","round","circuitId","name","date"]]
circuits_small = circuits[["circuitId","name","location","country"]].rename(
    columns={"name":"circuit_name"}
)

#Results has the outcome per driver per race
results_small = results[[
    "resultId","raceId","driverId","constructorId",
    "grid","position","positionText","positionOrder","points","statusId"
]]

# qualifying has starting performance (sometimes missing for some years)
# keep best available quali position per driver/race
qpos = (qualifying
        .sort_values(["raceId","driverId","position"])
        .groupby(["raceId","driverId"], as_index=False)
        .first()[["raceId","driverId","position"]]
        .rename(columns={"position":"qpos"}))

#Joins
df = (results_small
      .merge(races_small, on="raceId", how="left")
      .merge(circuits_small, on="circuitId", how="left")
      .merge(qpos, on=["raceId","driverId"], how="left")
      .merge(drivers[["driverId","driverRef","code","nationality"]], on="driverId", how="left")
      .merge(constructors[["constructorId","name"]].rename(columns={"name":"constructor_name"}),
             on="constructorId", how="left")
)

#Numeric Cleanups
for col in ["grid","position","positionOrder","points","qpos","year","round"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

#Keep rows with a valid finished order info (positionOrder present)
df = df.dropna(subset=["positionOrder","grid","year"]).reset_index(drop=True)

#Export to Data
df.to_csv("../data/derived_data/final_dataset.csv")
