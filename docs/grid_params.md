## ⚙️ Config: Param Grid Run

| Parameter        | Type    | Default    | Description                                         | Valid Values / Notes                                   | Used In                  |
|------------------|---------|------------|-----------------------------------------------------|--------------------------------------------------------|--------------------------|
| base_run_name    | str     | param_grid | Prefix for output folders and result naming         | Any string                                             | grid_edge_research_pipeline |
| res_save_path    | str     | —          | Directory or file path to save run results          | Any valid path                                         | grid_edge_research_pipeline |
| to_train_test    | bool    | True       | Enable standard train/test validation               | True / False                                           | validation pipeline         |
| to_wfa           | bool    | True       | Enable Walk Forward Analysis validation             | True / False                                           | validation pipeline         |
| to_bootstrap     | bool    | True       | Enable bootstrap resampling test                    | True / False                                           | validation pipeline         |
| to_null_fdr      | bool    | True       | Enable null distribution + FDR correction           | True / False                                           | validation pipeline         |
| default_params   | str     | —          | Path to default parameter YAML                      | Any .yaml path                                         | param loader                |
| custom_params    | str     | —          | Path to custom parameter YAML                       | Any .yaml path                                         | param loader                |
| feature_path     | str     | —          | Path to feature data file (.csv or .parquet)        | Valid .csv or .parquet file                            | feature loader              |
| hloc_path        | str     | —          | Path to HLOC (price) data file                      | Valid .csv or .parquet file                            | feature loader              |
| res_filetype     | str     | csv        | Output file format for results                      | "csv" or "parquet"                                     | result writer               |
| verbose          | bool    | True       | Print progress and logs to console                  | True / False. Only effective if n_jobs=1               | all modules (if serial)     |
| n_jobs           | int     | 1          | Number of parallel jobs to launch                   | Any integer ≥ 1                                        | grid orchestrator           |
| param_space      | dict    | —          | Parameters to grid search and override              | Key = valid param, Value = list of valid values        | grid orchestrator           |
