from utils.networks import *
import argparse



if __name__ == "__main__":
    arg = argparse.ArgumentParser(description="This is a test")
    
    arg.add_argument("-g", type=str, default="0", help="gpu index")
    args = arg.parse_args()