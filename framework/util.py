from argparse import ArgumentParser


def get_demo_args() -> dict:
    """Get the command line arguments for demo experiments.

    Returns:
        dict -- the arguments as key-value pairs
    """

    parser = ArgumentParser(
        description="A demonstrational tool for the neural framework")
    parser.add_argument("--batch-size",
                        dest="epoch_batch_size",
                        help="Number of epochs to train for per agent per batch",
                        type=int,
                        default=10)
    parser.add_argument("--batches",
                        dest="epoch_batches",
                        help="Number of batches of training to perform for each agent",
                        type=int,
                        default=100)
    parser.add_argument("--columns",
                        dest="cols",
                        type=int,
                        help="The number of columns that the visualisations should be arranged",
                        default=2)
    parser.add_argument("--port",
                        type=int,
                        help="The port to run the server on",
                        default=5000)
    args = parser.parse_args()
    return vars(args)
