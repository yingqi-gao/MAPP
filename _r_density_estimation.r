# Load the library and test run
# .libPaths('usr/local/lib/R/site-library/')
library(densityFPCA)
library(spatstat)
# library(fdapace)
# library(purrr)


# Get cdf of kde
kde_r <- function(test_obs_at_t, lower, upper){
    ###
    #' Kernel density estimation at round t. 
    #'
    #' Parameters:
    #' - test_obs_at_t (num vec): Test observations received at round t, i.e., observations for estimating current density.
    #' - lower (num): Lower support of all densities.
    #' - upper (num): Upper support of all densities.
    #'
    #' Return:
    #' The estimated cdf function.
    ###

    return(CDF(density(test_obs_at_t, bw = "SJ", from = lower, to = upper)))
}


# Get cdf of rde (repeated density estimation)
rde_r <- function(train_hist, train_bws, test_obs_at_t, lower, upper, grid_size, method){
    ###
    #' Estimates density at round t using the method in “Nonparametric Estimation of Repeated Densities with Heterogeneous Sample Sizes.”
    #'
    #' Parameters:
    #' - train_hist (list of num vectors): Training history, i.e., stored training observations. Each element is a numeric vector storing training observations at round t.
    #' - train_bws (num vec): Bandwidths selected at each round for kernel density estimation.
    #' - test_obs_at_t (num vec): Test observations received at round t, i.e., observations for estimating current density.
    #' - lower (num): Lower support of all densities.
    #' - upper (num): Upper support of all densities.
    #' - grid_size (int): The number of grid points to generate for evaluating estimated density.
    #' - method (cha): A string specifying the method to use for calculating the estimated parameters ("FPCA_MLE", "FPCA_MAP", "FPCA_BLUP").
    #'
    #' Return: A list of
    #' - est_params (num vec): A vector of estimated parameters.
    #' - est_cdf (function): The estimated cdf function.
    ###

    # Step 1: Generate grid points used for evaluating estimated density
    grid <- seq(lower, upper, length.out = grid_size)

    # Step 2: Use pre-smoothing to turn discrete observations into density functions
    # 1) Get bandwidth
    bw <- quantile(train_bws, 0.5)
    # 2) Pre-smooth
    train_sample <- preSmooth.kde(
        obsv = train_hist,
        grid = grid,
        kde.opt = list(
            bw = bw,
            kernel = 'g',
            from = lower,
            to = upper
        )
    )

    # Step 3: Transform the density functions into Hilbert space via centered log transformation (centered log-ratio)
    # 1) Calculate the constant to use for centering
    tm.c <- normL2(rep(1, grid_size), grid = grid) ^ 2
    # 2) Transform into Hilbert space
    ls.tm <- toHilbert(
        train_sample,
        grid = grid,
        transFun = function(f, grid) orthLog(f, grid, against = 1 / tm.c),
        eps = .Machine$double.eps^(1/2)
    )
    train_curve <- ls.tm$mat.curve

    # Step 4: Functional principal component analysis
    fpca.res <- do.call(
        fdapace::FPCA,
        list(
            Ly = asplit(train_curve, 1),
            Lt = replicate(n = nrow(train_curve), expr = grid, simplify = FALSE),
            optns = list(
                error = TRUE,
                lean = TRUE,
                FVEthreshold = 1,
                methodSelectK = 'FVE',
                plot = FALSE,
                useBinnedData = 'OFF'
            )
        )
    )

    # Step 5: Construct the induced approximating exponential family
    max.k <- 10
    fpca.den.fam <- fpca2DenFam(fpca.res, control = list(num.k = max.k))
    # checkDenFamNumeric(fpca.den.fam)
    # checkDenFamGrident(fpca.den.fam)

    # Step 6: Estimate using the induced family `fpca.den.fam`
    # 1) Estimation
    ls.fpca.esti <- fpcaEsti(
        mat.obsv = list(test_obs_at_t),
        fpca.res = fpca.res,
        esti.method = c(method),
        control = list(
            num.k = 'AIC', 
            max.k = max.k,
            method = 'LBFGS', 
            return.scale = 'parameter'
        )
    )
    # 2) Extract estimated parameters
    est_params <- ls.fpca.esti$res[1, ]
    est_params <- replace(est_params, is.na(est_params), 0)
    # 3) Construct estimated pdf over the grid
    est_pdf <- purrr::partial(fpca.den.fam$pdf, par = est_params)
    # 4) Consruct estimated cdf from pdf
    est_cdf <- function(x){
        value <- integrate(est_pdf, lower = lower, upper = x, stop.on.error = FALSE)$value
        return(value)
    }

    # Return
    return(est_cdf)
}