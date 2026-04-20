library(data.table)
library(stringr)
library(GeneralisedCovarianceMeasure)
library(cdcsis)
library(weightedGCM)
# library(CondIndTests)
library(RCIT)

test_others <- function(output_dir, simtype, n, p, nsim, method, alp = 0.05) {
  
  prefix <- str_glue("{simtype}_n{n}p{p}")
  
  # Initialize data tables to store results for this method
  Pvals <- data.table()
  Times <- data.table()
  
  for (sim in 0:(nsim - 1)) {
    # Define filenames for the current simulation
    filename_pvals_sim <- file.path(output_dir, str_glue("{prefix}_{method}_pvals_sim{sim}.csv"))
    filename_times_sim <- file.path(output_dir, str_glue("{prefix}_{method}_times_sim{sim}.csv"))
    
    # Check if results for this simulation already exist
    if (file.exists(filename_pvals_sim) && file.exists(filename_times_sim)) {
      tryCatch({
        # Load the existing results to maintain the cumulative list
        pval <- fread(filename_pvals_sim, header = FALSE)[1, 1]
        time_val <- fread(filename_times_sim, header = FALSE)[1, 1]
        Pvals <- rbindlist(list(Pvals, data.table(pval = pval)), fill = TRUE)
        Times <- rbindlist(list(Times, data.table(time = time_val)), fill = TRUE)
        message(str_glue("Results for sim {sim} with method {method} already exist."))
        next # Skip to the next simulation
      }, error = function(e) {
      })
    }
    
    if (sim %% 10 == 0) cat(str_glue("{method} sim = {sim} "))
    
    # Read in data
    filename <- file.path(output_dir, str_glue("{prefix}_sim{sim}.csv"))
    
    tryCatch({
      if (!file.exists(filename)) {
        stop(str_glue("File not found: {filename}"))
      }
      df <- fread(filename)
      x <- df$x
      y <- df$y
      Z <- as.matrix(df[, !c("x", "y"), with = FALSE])
      
      tic <- proc.time()
      
      # Different methods
      pval <- NA_real_
      if (method == "GCM") {
        res <- gcm.test(x, y, Z)
        pval <- res$p
      } else if (method == "CDIT") {
        res <- cdcov.test(x, y, Z)
        pval <- res$p.value
      } else if (method == "wGCM") {
        res <- wgcm.est(x, y, Z, 0.3, "gam")
        pval <- res
      } else if (method == "KCIT") {
        res = CondIndTest(x, y, Z, method = "KCI")
        pval = res$pvalue
      } else if (method == "RCIT") {
        res = RCIT(x, y, Z)
        pval = res$p
      }
      
      toc <- proc.time()
      Pvals <- rbindlist(list(Pvals, data.table(pval = pval)), fill = TRUE)
      Times <- rbindlist(list(Times, data.table(time = sum((toc - tic)[1:2]))), fill = TRUE)
      if (grepl("time", output_dir)) {
        cat(sprintf("sim %s elapsed %f\n", sim, sum((toc - tic)[1:2])))
      }

      # Save results for the single simulation
      fwrite(data.table(pval), filename_pvals_sim, col.names = FALSE)
      fwrite(data.table(sum((toc - tic)[1:2])), filename_times_sim, col.names = FALSE)
      
    }, error = function(e) {
      message(str_glue("An error occurred during simulation {sim}: {e$message}"))
      Pvals <- rbindlist(list(Pvals, data.table(pval = NA_real_)), fill = TRUE)
      Times <- rbindlist(list(Times, data.table(time = NA_real_)), fill = TRUE)
    })
    
    # Checkpoint: save cumulative results every 10th simulation (9, 19, 29, ...)
    if (sim %% 10 == 9) {
      message(str_glue("Saving cumulative results up to sim {sim}..."))
      filename_pvals_full <- file.path(output_dir, str_glue("{prefix}_{method}_pvals.csv"))
      filename_times_full <- file.path(output_dir, str_glue("{prefix}_{method}_times.csv"))
      
      fwrite(Pvals, filename_pvals_full, col.names = FALSE)
      fwrite(Times, filename_times_full, col.names = FALSE)
    }
  }
  
  # Final save of all results after the loop completes
  filename_pvals_full <- file.path(output_dir, str_glue("{prefix}_{method}_pvals.csv"))
  filename_times_full <- file.path(output_dir, str_glue("{prefix}_{method}_times.csv"))
  fwrite(Pvals, filename_pvals_full, col.names = FALSE)
  fwrite(Times, filename_times_full, col.names = FALSE)
  
  return(invisible())
}