import os
from rpy2.robjects import numpy2ri
import rpy2.robjects as ro

# Set the R_HOME address
os.environ["R_HOME"] = "/Library/Frameworks/R.framework/Resources"

# Activate automatic conversion between R and Python (NumPy)
numpy2ri.activate()

# Source the R script (load all functions from the R file)
ro.r["source"]("_r_density_estimation.r")

# Access the R function from the R environment
kde = ro.globalenv["kde_r"]
rde = ro.globalenv["rde_r"]
ecdf = ro.r["ecdf"]


if __name__ == "__main__":
    print("running tests...")
    from scipy.stats import norm

    # test ecdf
    samples = norm.rvs(loc=0, scale=1, size=100)
    print(ecdf(samples)(0.32))

    # test kde
    samples = norm.rvs(loc=0, scale=1, size=(2,100))
    print(kde(samples, -1, 1, 1024)[1].rx2("pdf")(0.32))
