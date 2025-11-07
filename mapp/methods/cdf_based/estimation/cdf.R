# CDF Estimation Functions for Auction Pricing
#
# This file provides R functions for estimating
# cumulative distribution functions (CDFs) from auction bid data
# using different statistical methods:
#
# - get_cdf_r(): Converts density estimates to callable CDF functions
# - kde_cdf_r(): Kernel Density Estimation for smooth CDF estimation
# - rde_cdf_r(): Repeated Density Estimation using FPCA with training data
#
# Note: For empirical CDFs, use R's built-in ecdf() function directly.
#
# These functions are called from Python via rpy2 for auction pricing.


#' Convert Density Estimate to CDF
#'
#' @param grid Vector of equally-spaced evaluation points
#' @param pdf_vals Vector of density estimates at grid points
#' @return Single callable CDF function
get_cdf_r <- function(grid, pdf_vals) {
  cdf_vals <- cumsum(pdf_vals * c(0, diff(grid)))
  approxfun(grid, cdf_vals / cdf_vals[length(cdf_vals)], rule = 2)
}


#' Kernel Density Estimation CDF
#'
#' @param sample Vector of bid observations for single auction
#' @param lower Lower bound of bid support
#' @param upper Upper bound of bid support
#' @param grid_size Number of evaluation points (default: 1024)
#' @return Single smooth callable CDF function
kde_cdf_r <- function(sample, lower, upper, grid_size = 1024) {
  grid <- seq(lower, upper, length.out = grid_size)

  # Kernel density estimation with robust bandwidth selection
  # Try Sheather-Jones first, fall back to more robust methods if it fails
  bw_method <- tryCatch(
    {
      bw.SJ(sample)  # Test if SJ works
      "SJ"
    },
    error = function(e) {
      # SJ failed (sample too sparse), use normal reference distribution
      "nrd0"
    }
  )

  kde <- density(sample, bw = bw_method, from = lower, to = upper)
  pdf_vals <- approx(kde$x, kde$y, xout = grid, rule = 2)$y

  get_cdf_r(grid, pdf_vals)
}


#' Train Repeated Density Estimation Model (Qiu et al., 2022)
#'
#' @param train_samples Matrix (n_train x n_bids) of historical auction data
#' @param lower Lower bound of bid support
#' @param upper Upper bound of bid support
#' @param grid_size Number of evaluation points (default: 1024)
#' @param max_k Maximum FPCA dimensions (default: 10)
#' @return Trained RDE model object
train_rde_r <- function(
  train_samples,
  lower,
  upper,
  grid_size = 1024,
  max_k = 10
) {
  # Step 1: Set up a grid to be used for density estimation and interpolation
  grid <- seq(lower, upper, length.out = grid_size)

  # Step 2: Use pre-smoothing to turn discrete observations into densities
  # 2.1: Get bandwidth with robust fallback
  bw_values <- apply(train_samples, 1, function(sample) {
    tryCatch(
      bw.SJ(sample),
      error = function(e) bw.nrd0(sample)  # Fallback to nrd0 if SJ fails
    )
  })
  bw <- quantile(bw_values, 0.5)
  # 2.2: Pre-smooth using KDE
  presmoothed <- densityFPCA::preSmooth.kde(train_samples, grid, kde.opt = list(
    bw = bw, kernel = "g", from = lower, to = upper
  ))

  # Step 3: Transform the density functions into Hilbert space
  #         via centered log transformation (centered log-ratio)
  # 3.1: Calculate the constant to be used for centering
  tm_c <- densityFPCA::normL2(rep(1, grid_size), grid = grid)^2
  # 3.2: Transform into Hilbert space
  trans_fun <- function(f, grid) {
    densityFPCA::orthLog(f, grid, against = 1 / tm_c)
  }
  train_curve <- densityFPCA::toHilbert(
    presmoothed,
    grid = grid,
    transFun = trans_fun,
    eps = .Machine$double.eps^(1 / 2)
  )$mat.curve

  # Step 4: Perform the functional principal component analysis (FPCA)
  fpca_res <- fdapace::FPCA(
    Ly = asplit(train_curve, 1),
    Lt = replicate(nrow(train_curve), grid, simplify = FALSE),
    optns = list(error = TRUE, lean = TRUE, FVEthreshold = 1,
                 methodSelectK = "FVE", plot = FALSE, useBinnedData = "OFF")
  )

  # Step 5: Construct the induced approximating exponential family
  fpca_den_fam <- densityFPCA::fpca2DenFam(
    fpca_res,
    control = list(num.k = max_k)
  )

  # Return trained model
  list(
    fpca_res = fpca_res,
    fpca_den_fam = fpca_den_fam,
    grid = grid,
    max_k = max_k
  )
}


#' RDE CDF for single sample using trained model
#'
#' @param sample Vector of bid observations for single auction
#' @param rde_model Trained RDE model from train_rde_r()
#' @param method Estimation method: "FPCA_BLUP", "FPCA_MAP", "FPCA_MLE"
#' @return Single callable CDF function
rde_cdf_r <- function(sample, rde_model, method = "FPCA_MLE") {
  # Step 6: Estimate parameters using the induced family
  ls_fpca_esti <- densityFPCA::fpcaEsti(
    matrix(sample, nrow = 1),  # Convert vector to single-row matrix
    rde_model$fpca_res,
    esti.method = method,
    control = list(
      num.k = "AIC",
      max.k = rde_model$max_k,
      method = "LBFGS",
      return.scale = "parameter"
    )
  )

  # Step 7: Construct the density estimates at grid points
  # Check if we have multiple rows and
  # validate only first row should have valid data
  if (is.matrix(ls_fpca_esti$res) && nrow(ls_fpca_esti$res) > 1) {
    # Check if any rows beyond the first have non-NA values
    other_rows_have_data <- apply(
      ls_fpca_esti$res[-1, , drop = FALSE], 1, function(row) any(!is.na(row))
    )
    if (any(other_rows_have_data)) {
      stop("Unexpected: Multiple rows with valid parameter estimates found.
      Expected only first row to have data.")
    }
  }
  filled_params <- ls_fpca_esti$res[1, ]  # Get first (and only valid) row
  filled_params[is.na(filled_params)] <- 0
  rde <- densityFPCA::par2pdf(
    rde_model$fpca_den_fam,
    rde_model$fpca_den_fam$fill.par(matrix(filled_params, nrow = 1)),
    grid = rde_model$grid
  )

  # Step 8: Compute the CDF using the helper function
  pdf_vals <- rde[1, ]  # Get first (and only) row
  get_cdf_r(rde_model$grid, pdf_vals)
}