# Get cdf of kde
kde_r <- function(test_obs_at_t, lower, upper) {
  ###
  #' Kernel density estimation at round t.
  #'
  #' Parameters:
  #' - test_obs_at_t (num vec): Test observations received at round t, i.e.,
  #'                            observations for estimating current density.
  #' - lower (num): Lower support of all densities.
  #' - upper (num): Upper support of all densities.
  #'
  #' Return:
  #' The estimated cdf function.
  ###
  library(spatstat)

  return(CDF(density(test_obs_at_t,
                     bw = "SJ",
                     from = lower,
                     to = upper)))
}



# Get cdf of rde (repeated density estimation)
# 1. Training
rde_training_r <- function(train_hist,
                           train_bws,
                           lower,
                           upper,
                           grid_size) {
  ###
  #' Learns the density family from training data.
  #'
  #' Parameters:
  #' - train_hist (list of num vectors): Training history, i.e.,
  #'                                     stored training observations.
  #' - train_bws (num vec): Bandwidths selected for each training vector.
  #' - lower (num): Lower bound of the common support of all densities.
  #' - upper (num): Upper support of the common support of all densities.
  #' - grid_size (int): Number of grid points to use
  #'                    for evaluating estimated density.
  #'
  #' Return: A list of
  #' - fpca_res (list): Results of principal principal components analysis.
  #' - max_k (int): Maximum number of functional principal components to use.
  #' - fpca_den_fam_pdf (function): Estimated pdf function of the family.
  ###

  library(densityFPCA)


  # Step 1: Generate grid points used for evaluating estimated density
  grid <- seq(lower, upper, length.out = grid_size)


  # Step 2: Use pre-smoothing to turn discrete observations
  #                              into density functions
  # 1) Get bandwidth
  bw <- quantile(train_bws, 0.5)

  # 2) Pre-smooth
  train_sample <- preSmooth.kde(
    obsv = train_hist,
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
  # 1) Calculate the constant to use for centering
  tm_c <- normL2(rep(1, grid_size), grid = grid) ^ 2

  # 2) Transform into Hilbert space
  ls_tm <- toHilbert(
    train_sample,
    grid = grid,
    transFun = function(f, grid) orthLog(f, grid, against = 1 / tm_c),
    eps = .Machine$double.eps^(1 / 2)
  )
  train_curve <- ls_tm$mat.curve


  # Step 4: Functional principal component analysis
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
  max_k <- 10
  fpca_den_fam <- fpca2DenFam(fpca_res, control = list(num.k = max_k))
  # checkDenFamNumeric(fpca.den.fam)
  # checkDenFamGrident(fpca.den.fam)


  # Return
  return(list(fpca_res = fpca_res,
              max_k = max_k,
              fpca_den_fam_pdf = fpca_den_fam$pdf))
}


# 2. Testing
rde_testing_r <- function(test_obs_at_t,
                          method,
                          lower,
                          training_results) {
  ###
  #' Estimates cdf from induced density family using new observations.
  #'
  #' Parameters:
  #' - test_obs_at_t (num vec): Test observations received at round t, i.e.,
  #'                            observations to estimate density of.
  #' - method (cha): Method to use for calculating the estimated parameters
  #'                 ("FPCA_MLE", "FPCA_MAP", "FPCA_BLUP").
  #' - lower (num): Lower bound of the common support of all densities.
  #' - training_results (list): Results from rde_training_r.
  #'
  #' Return:
  #' - est_cdf (function): The estimated cdf function.
  ###

  # Handle training results.
  fpca_res <- training_results["fpca_res"]
  max_k <- training_results["max_k"]
  fpca_den_fam_pdf <- training_results["fpca_den_fam_pdf"]

  # Estimate using the induced family `fpca.den.fam`
  # 1) Estimation
  ls_fpca_esti <- fpcaEsti(
    mat.obsv = list(test_obs_at_t),
    fpca.res = fpca_res,
    esti.method = c(method),
    control = list(
      num.k = "AIC",
      max.k = max_k,
      method = "LBFGS",
      return.scale = "parameter"
    )
  )

  # 2) Extract estimated parameters
  est_params <- ls_fpca_esti$res[1, ]
  est_params <- replace(est_params, is.na(est_params), 0)

  # 3) Construct estimated pdf over the grid
  est_pdf <- purrr::partial(fpca_den_fam_pdf, par = est_params)

  # 4) Consruct estimated cdf from pdf
  est_cdf <- function(x) {
    value <- integrate(est_pdf,
                       lower = lower,
                       upper = x,
                       stop.on.error = FALSE)$value
    return(value)
  }


  # Return
  return(est_cdf)
}