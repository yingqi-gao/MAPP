from _classes_true_distribution import UniformDistribution, NormalDistribution, ExponentialDistribution, ParetoDistribution
import pickle
import sys
import time



true_dist_refs = {"uniform": UniformDistribution,
                  "normal": NormalDistribution,
                  "exponential": ExponentialDistribution,
                  "pareto": ParetoDistribution}



def get_training_bids(dist_type: str, 
                      num_bidders = 500,
                      num_rounds = 100, 
                      lower = 1, 
                      upper = 10,
                      repetition_no = 1):
    training_bids = []
    for t in range(num_rounds):
        true_dist = true_dist_refs[dist_type](dist_type = dist_type, lower = lower, upper = upper)
        training_bids_at_t = true_dist.generate_bids(num_bidders = num_bidders)
        training_bids.append(training_bids_at_t)

    with open("data/sim/" + dist_type + "/train_bids_" + dist_type + "_rep" + str(repetition_no) + ".pkl", "wb") as file:
        pickle.dump(training_bids, file)
    print(f"All done with training initialization of {dist_type} - Repetition No.{repetition_no}!")



def get_testing_info(dist_type: str,
                     num_bidders = 500,
                     lower = 1, 
                     upper = 10):
    true_dist = true_dist_refs[dist_type](dist_type = dist_type, lower = lower, upper = upper)
    testing_bids = true_dist.generate_bids(num_bidders = num_bidders)
    ideal_price, ideal_revenue = true_dist.get_ideals()
    testing_info = {"true_dist": true_dist,
                    "testing_bids": testing_bids,
                    "ideal_price": ideal_price,
                    "ideal_revenue": ideal_revenue}
    
    with open("data/sim/" + dist_type + "/test_info_" + dist_type + ".pkl", "wb") as file:
        pickle.dump(testing_info, file)
    print(f"All done with testing initiaization of {dist_type}!")



def main():
    t_start = time.time()
    training_or_testing = sys.argv[1]
    dist_type = sys.argv[2]
    if training_or_testing == "training":
        get_training_bids(dist_type = dist_type)
    elif training_or_testing == "testing":
        get_testing_info(dist_type = dist_type)
    else:
        raise ValueError("Argument not supported!")
    print(f"It took {time.time() - t_start} seconds.")

if __name__ == "__main__":
    main()