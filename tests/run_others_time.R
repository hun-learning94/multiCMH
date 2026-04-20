#########################################################################################################
## Test Simulations using others (R)
#########################################################################################################
source("tests/test_others.R")
METHODS = c("RCIT", "GCM")

library(jsonlite)
settings = fromJSON("tests/settings_time.json")
output_dir <- settings$output_dir
print(output_dir)
alp = settings$alp
nsim = settings$nsim

null_p = settings$null_p
null_N = settings$null_N
null_n = settings$null_n
null_P = settings$null_P
roc_p = null_p
roc_N = null_N
roc_n = null_n
roc_P = null_P

print(paste("NULL parameters:", null_p, paste(null_N, collapse = ","), null_n, paste(null_P, collapse = ",")))

run_null = settings$run_null
run_roc = settings$run_roc
print(paste0('run_null ', run_null, ', run_roc ', run_roc))



#########################################################################################################
if(run_null){
  simtype = 'null'
  
  N = null_N
  p = null_p
  for (method in METHODS) {
    for(n in N){
      cat("\n", str_glue("{method} testing n = {n}, p = {p}"), "\n")
      test_others(output_dir, simtype, n, p, nsim, method, alp)
    }
  }
}

