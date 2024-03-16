from argparse import ArgumentParser
import main
import os

if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument('-data', type=str, default="IHDP",
                        choices=["Mdata", "IHDP", "News"])
    parser.add_argument('-w', '--weight_method', type=str,
                        default="ed", choices=["nrw", "mlp", "ed"])

    parser.add_argument('-cache_dir', type=str, default="./cache/")

    parser.add_argument('-z_dim', type=int, default=128)
    parser.add_argument('-h_dim', type=int, default=200)
    parser.add_argument('-batch', type=int, default=100)
    parser.add_argument('-decay', type=float, default=1e-4)

    parser.add_argument('-reps', type=int, default=10)
    parser.add_argument('-lr', type=float, default=1e-3)
    parser.add_argument('-epochs', type=int, default=100)

    args = parser.parse_args()

    if not os.path.exists(args.cache_dir):
        os.mkdir(args.cache_dir)

    try:
        pre_log_num = len([listx for listx in os.listdir("./cache/")])
    except:
        pre_log_num = 0
    args.cache_dir = args.cache_dir + str(pre_log_num) + "/"
    os.mkdir(args.cache_dir)

    print(args)

    main.main(args)
