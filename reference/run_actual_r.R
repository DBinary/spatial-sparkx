# Run unmodified SPARK-X R source with its original sparse C++ reductions.
args <- commandArgs(trailingOnly = TRUE)
root <- normalizePath(args[1])
input <- normalizePath(args[2])
output <- args[3]
dir.create(output, recursive = TRUE, showWarnings = FALSE)
suppressPackageStartupMessages(library(Matrix))
suppressPackageStartupMessages(library(parallel))
suppressPackageStartupMessages(library(CompQuadForm))
Rcpp::sourceCpp(file.path(root, ".cache/r_helpers.cpp"),
                cacheDir = file.path(root, ".cache/Rcpp"), showOutput = FALSE)
source(file.path(root, "reference/upstream/R/sparkx.R"))
manifest <- read.delim(file.path(input, "manifest.tsv"), stringsAsFactors = FALSE)
writeLines(capture.output(sessionInfo()), file.path(output, "sessionInfo.txt"))
options(digits = 17)
statuses <- list()
counter <- 0
for (mode in c("default", "analytic_equivalent")) {
    if (mode == "default") {
        davies <- CompQuadForm::davies
    } else {
        davies <- function(q, lambda, ...) {
            # Independent R-side control for the exact projection-kernel law.
            # The official sparkx.R file itself is not edited.
            scale <- mean(lambda)
            if (scale > 0) stopifnot(max(abs(lambda / scale - 1)) < 1e-10)
            list(Qq = pchisq(q / scale, df = length(lambda), lower.tail = FALSE))
        }
    }
    for (i in seq_len(nrow(manifest))) {
        name <- manifest$case[i]
        folder <- file.path(input, name)
        counts <- as.matrix(read.csv(file.path(folder, "counts.csv"), row.names = 1, check.names = FALSE))
        coords <- as.matrix(read.csv(file.path(folder, "coordinates.csv"), row.names = 1, check.names = FALSE))
        design_path <- file.path(folder, "covariates.csv")
        design <- if (file.exists(design_path)) as.matrix(read.csv(design_path, row.names = 1, check.names = FALSE)) else NULL
        if (manifest$sparse[i]) counts <- as(counts, "sparseMatrix")
        destination <- file.path(output, mode, name)
        dir.create(destination, recursive = TRUE, showWarnings = FALSE)
        error <- ""
        messages <- character()
        result <- tryCatch(
            withCallingHandlers(
                sparkx(counts, coords, X_in = design, numCores = 1,
                       option = manifest$option[i], verbose = FALSE),
                warning = function(w) { messages <<- c(messages, conditionMessage(w)); invokeRestart("muffleWarning") }
            ), error = function(e) { error <<- conditionMessage(e); NULL }
        )
        if (!is.null(result)) {
            write.csv(result$stats, file.path(destination, "stats.csv"))
            write.csv(result$res_stest, file.path(destination, "pvalues.csv"))
            write.csv(result$res_mtest, file.path(destination, "combined.csv"))
        }
        counter <- counter + 1
        statuses[[counter]] <- data.frame(case = name, mode = mode, success = !is.null(result),
                                          error = error, warnings = paste(unique(messages), collapse = " | "))
        cat("REFERENCE", mode, name, if (is.null(result)) error else "OK", "\n")
    }
}
write.csv(do.call(rbind, statuses), file.path(output, "statuses.csv"), row.names = FALSE)

# Also validate the optional normalization helper with actual R nls/lm.
vst_counts <- as.matrix(read.csv(file.path(input, "vst_counts.csv"), row.names = 1, check.names = FALSE))
vst_result <- NormalizeVSTfast(as(vst_counts, "sparseMatrix"))
write.csv(vst_result, file.path(output, "vst.csv"))
