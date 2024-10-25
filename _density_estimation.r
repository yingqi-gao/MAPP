# Load the necessary packages
if (!requireNamespace("densityFPCA", quietly = TRUE)) {
  install.packages("densityFPCA")
}
library(densityFPCA)

if (!requireNamespace("tidyverse", quietly = TRUE)) {
  install.packages("tidyverse")
}
library(tidyverse)



#' Title: eCDF in R
#'
#' Description:
#' This function constructs the Cumulative Distribution Function(s) (CDF(s))
#' using the empirical CDF method for given numerical dataset(s).
#' Each is an interpolating function that can be evaluated at arbitrary points.
#'
#' @param samples A matrix or list of samples of observations, where each row of
#'                the matrix or element of the list is a sample for which the
#'                CDF will be computed.
#' @return A list of Cumulative Distribution Functions (CDFs), each of which is
#'         a function to be evaluated at any given point, with a length equal to
#'         the number of rows or elements in `samples`.
#' @example
#' # Generate a random matrix of samples
#' samples <- matrix(c(runif(100, -1, 1), rnorm(100)), nrow = 2, byrow = TRUE)
#'
#' # Run the function
#' results <- ecdf_r(samples)
#'
#' results[[1]](0.32)
#' punif(0.32, min = -1, max = 1)
#'
#' results[[2]](0.32)
#' pnorm(0.32)
ecdf_r <- function(samples) {
  # Step 1: Check if the input type is as expected.
  if (!is.list(samples) && !is.matrix(samples)) {
    stop("The input must be list or matrix.")
  }

  # Step 2: If the input is a matrix, convert it to a list.
  if (is.matrix(samples)) {
    samples <- lapply(seq_len(nrow(samples)), function(i) samples[i, ])
  }

  # Step 3: Construct the eCDF for each sample.
  cdfs <- list()
  for (i in seq_len(length(samples))) {
    cdf <- ecdf(samples[[i]])
    cdfs[[i]] <- cdf
  }

  # Return the list of eCDFs
  return(cdfs)
}



#' Title: CDFs Construction Function
#'
#' Description:
#' This function constructs the Cumulative Distribution Function(s) (CDF(s))
#' from given density estimates at points within a common grid.
#' Each is an interpolating function that can be evaluated at arbitrary points.
#'
#' @param grid A numeric vector of equally spaced points at which the density
#'             was estimated.
#' @param mat_pdf A matrix where each row contains the density estimates
#'                at given grid points for each sample.
#' @return A list of Cumulative Distribution Functions (CDFs), each of which is
#'         a function to be evaluated at any given point, with a length equal to
#'         the number of rows in `mat_pdf`.
#' @example
#' # Generate a random grid
#' grid <- seq(-1, 1, length.out = 1024)
#'
#' # Generate a random matrix of density estimates
#' mat_pdf <- matrix(c(sapply(grid, dunif, min = -1, max = 1),
#'                     sapply(grid, dnorm)),
#'                   nrow = 2, byrow = TRUE)
#'
#' # Run the function
#' results <- get_cdfs(grid, mat_pdf)
#'
#' # Test if the results are as expected
#' results[[1]](0.32)
#' punif(0.32, min = -1, max = 1)
#'
#' results[[2]](0.32)
#' pnorm(0.32)
get_cdfs <- function(grid, mat_pdf) {
  cdfs <- list()

  for (i in seq_len(nrow(mat_pdf))) {
    # Compute the CDF by cumulatively summing the density estimates.
    cdf_est <- cumsum(mat_pdf[i, ] * c(0, diff(grid)))
    cdf_est <- cdf_est / cdf_est[length(cdf_est)] # normalization
    cdf <- approxfun(grid, cdf_est)

    # Append to the list
    cdfs[[i]] <- cdf
  }

  # Return the list
  return(cdfs)
}



#' Title: KDE in R
#'
#' Description:
#' This function computes the Cumulative Distribution Function(s) (CDF(s)) using
#' the Kernel Density Estimation (KDE) method for given numerical dataset(s).
#' Each is an interpolating function that can be evaluated at arbitrary points.
#'
#' @param samples A matrix or list of samples of observations, where each row of
#'                the matrix or element of the list is a sample for which the
#'                CDF will be computed.
#' @param lower A numeric value specifying the lower bound of the support for
#'              all densities.
#' @param upper A numeric value specifying the upper bound of the support for
#'              all densities.
#' @param grid_size The number of equally spaced points at which the densties
#'                  are to be estimated.
#' @return A list of Cumulative Distribution Functions (CDFs), each of which is
#'         a function to be evaluated at any given point, , with a length equal
#'         to the number of rows or elements in `samples`.
#' @example
#' # Generate a random matrix of samples
#' samples <- matrix(c(runif(100, -1, 1), rnorm(100)), nrow = 2, byrow = TRUE)
#'
#' # Set other parameters
#' lower <- -1
#' upper <- 1
#' grid_size <- 1024
#'
#' # Run the function
#' results <- kde_r(samples, lower, upper, grid_size)
#'
#' # Test if the results are as expected
#' results[[1]](0.32)
#' punif(0.32, min = -1, max = 1)
#'
#' results[[2]](0.32)
#' pnorm(0.32)
kde_r <- function(samples, lower, upper, grid_size = 1024) {
  # Step 1: Check if the input type is as expected.
  if (!is.list(samples) && !is.matrix(samples)) {
    stop("The input must be list or matrix.")
  }

  # Step 2: If the input is a matrix, convert it to a list.
  if (is.matrix(samples)) {
    samples <- lapply(seq_len(nrow(samples)), function(i) samples[i, ])
  }

  # Step 3: Set up a grid to be used for density estimation and interpolation
  grid <- seq(lower, upper, length.out = grid_size)
  n <- length(samples)
  mat_pdf <- matrix(0, ncol = length(grid), nrow = n)

  # Step 4: Compute the kernel density estimated values at the grid
  for (i in seq_len(n)) {
    kde <- do.call(
      stats::density,
      c(list(x = samples[[i]]),
        kde.opt = list(bw = "sj", kernel = "g", from = lower, to = upper))
    )
    mat_pdf[i, ] <- approx(
      x = kde$x, y = kde$y, xout = grid,
      rule = 2
    )$y
  }

  # Step 5: Compute the CDFs using the helper function get_cdfs
  results <- get_cdfs(grid, mat_pdf)

  # Return the list returned by get_cdfs
  return(results)
}



#' Title: RDE in R
#'
#' Description:
#' This function computes the Cumulative Distribution Function(s) (CDF(s)) using
#' the Repeated Density Estimation (KDE) method from Qiu et al. (2022)
#' for given numerical dataset(s).
#' Each is an interpolating function that can be evaluated at arbitrary points.
#'
#' @param train_samples A matrix or list of training samples of observations
#'                      from which the approximating exponential family will be
#'                      learned. Each row of the matrix or element of the list
#'                      is a training sample.
#' @param test_samples A matrix or list of test samples of observations, where
#'                     each row of the matrix or element of the list is a
#'                     test sample for which the CDF will be computed.
#' @param lower A numeric value specifying the lower bound of the support for
#'              all densities.
#' @param upper A numeric value specifying the upper bound of the support for
#'              all densities.
#' @param grid_size The number of equally spaced points at which the densties
#'                  are to be estimated. Defaults to 1024.
#' @param max_k The maximum number of dimensions the appoximating exponential
#'              family is expected to have. Defaults to 10.
#' @param method The method to be used for RDE. Options include "FPCA_BLUP",
#'               "FPCA_MAP", and "FPCA_MLE". Defaults to "FPCA_MLE".
#' @return A list of Cumulative Distribution Functions (CDFs), each of which is
#'         a function to be evaluated at any given point, , with a length equal
#'         to the number of rows or elements in `test_samples`.
#' @example
#' # Generate training samples
#' train_samples <- t(replicate(50, extraDistr::rtnorm(200, a = -1, b = 1,
#'                                                     mean = runif(1, -1, 1),
#'                                                     sd = runif(1, 0, 2))))
#'
#' # Generate test samples
#' test_samples <- t(replicate(200, extraDistr::rtnorm(10, a = -1, b = 1,
#'                                                     mean = runif(1, -1, 1),
#'                                                     sd = runif(1, 0, 2))))
#'
#' # Set other parameters
#' lower <- -1
#' upper <- 1
#'
#' # Run the function
#' results <- rde_r(train_samples, test_samples, lower, upper)
#'
#' # Test if the results are as expected
#' results[[sample.int(200, 1)]](0.32)
rde_r <- function(train_samples,
                  test_samples,
                  lower,
                  upper,
                  grid_size = 1024,
                  max_k = 10,
                  method = "FPCA_MLE") {
  # Step 1: Set up a grid to be used for density estimation and interpolation
  grid <- seq(lower, upper, length.out = grid_size)

  # Step 2: Use pre-smoothing to turn discrete observations into densities
  # 2.1: Get bandwidth
  bw <- quantile(apply(train_samples, 1, bw.SJ), 0.5)

  # 2.2: Pre-smooth using KDE
  presmoothed_train_sample <- preSmooth.kde(
    obsv = train_samples,
    grid = grid,
    kde.opt = list(
      bw = bw,
      kernel = "g",
      from = lower,
      to = upper
    )
  )

  # Step 3: Transform the density functions into Hilbert space
  #         via centered log transformation (centered log-ratio)
  # 3.1: Calculate the constant to be used for centering
  tm_c <- normL2(rep(1, grid_size), grid = grid)^2

  # 3.2: Transform into Hilbert space
  train_curve <- toHilbert(
    presmoothed_train_sample,
    grid = grid,
    transFun = function(f, grid) orthLog(f, grid, against = 1 / tm_c),
    eps = .Machine$double.eps^(1 / 2)
  )$mat.curve

  # Step 4: Perform the functional principal component analysis (FPCA)
  fpca_res <- do.call(
    fdapace::FPCA,
    list(
      Ly = asplit(train_curve, 1),
      Lt = replicate(n = nrow(train_curve), expr = grid, simplify = FALSE),
      optns = list(
        error = TRUE,
        lean = TRUE,
        FVEthreshold = 1,
        methodSelectK = "FVE",
        plot = FALSE,
        useBinnedData = "OFF"
      )
    )
  )

  # Step 5: Construct the induced approximating exponential family
  fpca_den_fam <- fpca2DenFam(fpca_res, control = list(num.k = max_k))
  # Numeric checks:
  # checkDenFamNumeric(fpca.den.fam)
  # checkDenFamGrident(fpca.den.fam)

  # Step 6: Estimate parameters using the induced family `fpca.den.fam`
  ls_fpca_esti <- fpcaEsti(
    mat.obsv = test_samples,
    fpca.res = fpca_res,
    esti.method = method,
    control = list(
      num.k = "AIC",
      max.k = max_k,
      method = "LBFGS",
      return.scale = "parameter"
    )
  )

  # Step 7: Construct the density estimates at grid points
  rde <- par2pdf(
    fpca_den_fam,
    fpca_den_fam$fill.par(ls_fpca_esti$res %>% `[<-`(is.na(.), 0)),
    grid = grid
  )

  # Step 8: Compute the CDFs using the helper function get_cdfs
  results <- get_cdfs(grid, rde)

  # Return the list returned by get_cdfs
  return(results)
}