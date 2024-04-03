from _classes_initialization import OnlineAuctionRandomInitialization
import pickle



def main():
    num_rounds = 200
    lower = 1
    upper = 10
    min_num_bidders = 10

    initializations_to_dump = {}

    initialization_reference = {
        "dist_type": {"uniform": "uniform", "normal": "normal", "exponential": "exponential"},
        "upper_status": {"fixed": False, "floated": True},
        "scale": {"small": 100, "large": 10000}
    }

    for i in range(3):
        name1, dist_type = [*initialization_reference["dist_type"].items()][i]
        for j in range(2):
            name2, upper_status = [*initialization_reference["upper_status"].items()][j]
            for k in range(2):
                name3, scale = [*initialization_reference["scale"].items()][k]

                initialization_params = {
                    "num_rounds": num_rounds,
                    "distribution_type": dist_type,
                    "lower": lower,
                    "upper": upper,
                    "is_upper_floated": upper_status,
                    "min_num_bidders": min_num_bidders,
                    "max_num_bidders": scale
                }
                
                initializations_to_dump["_".join([name1, name2, name3])] = OnlineAuctionRandomInitialization(**initialization_params)
                
    with open("data/initializations.pkl", "wb") as file:
        pickle.dump(initializations_to_dump, file)



if __name__ == "__main__":
    main()