import argparse
import numpy as np 
import torch
import random

def get_config_ViT():
    config = {
        "F": 5,
        "Pt": 1,
        "Pf": 3,
        "Qf": 10,
        "dim": 144,
        "depth": 1,
        "heads": 8,
        "mlp_dim": 720,
        "dropout": 0.4,
        "emb_dropout": 0.0
	}    
    return config

def print_config(config):

    print("\n"+"="*25+" Print all config value "+"="*25)
    for key in config:
        print(f"config[\'{key}\'] = {config[key]}")

    print("="*70)
    print("", flush=True)

def get_config():
    config = {
        # Scenario parameter
		"subject_list": [i+1 for i in range(40)],
		"exercise_list": [1,2,3],
        # DSP parameter
		"fs": 2000,
        "num_channel": 12,
        "window_size_sec": 0.2,
        "window_step_sec": 0.1,
        "type_filter": "BPF_20_200",
        "type_norm": "mvc",
        # Training parameter
        "num_epoch": 1000,
        "batch_size": 512,
        "lr": 0.001,
        "dropout": 0.4,
        # Other
        "database": "DB2",
        "model_type": "DNN_feature",
        "en_train": True,
        "load_model": True,
        "class_rest": False,
        "feat_extract": True,
        "load_dataset": False,
        "save_dataset": False
	}
    config["window_size"] = int(0.2 * config["fs"]) # 200 ms => 0.2 * fs = 400 sample points
    config["window_step"] = int(0.1 * config["fs"]) # 200 ms => 0.2 * fs = 400 sample points
    config["model_PATH"] = f'./Models/{config["model_type"]}_model.pth'
    config["device"] = "cuda:0" if torch.cuda.is_available() else "cpu"

    exercise_list_np = np.array(config["exercise_list"])
    config["number_gesture"] = int(np.any(exercise_list_np==1))*17 + int(np.any(exercise_list_np==2))*23 + int(np.any(exercise_list_np==3))*9
    
    return config

# config = get_config()
# print("\n"+"="*70)
# for key in config:
#     print(key, "=", config[key])
#     exec(f"{key} = config[\"{key}\"]")
# print("="*70)
# print("", flush=True)

# Define a custom argument type for a list of integers
def list_of_ints(arg):
    return list(map(int, arg.split(',')))

def get_args(raw_args=None):
    parser = argparse.ArgumentParser()
    # Scenario parameter
    parser.add_argument("--subject_list", type=list_of_ints, default=[i+1 for i in range(40)])
    parser.add_argument("--exercise_list", type=list_of_ints, default=[1,2,3])
    # DSP parameter
    parser.add_argument("--fs", type=int, default=2000)
    parser.add_argument("--num_channel", type=int, default=12)
    parser.add_argument("--window_size_sec", type=float, default=0.2)
    parser.add_argument("--window_step_sec", type=float, default=0.1)
    parser.add_argument("--type_filter", default='none')
    parser.add_argument("--type_norm", default='none')
    # Training parameter
    parser.add_argument("--num_epoch", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--isEarlyExit", default=False)
    # Other
    parser.add_argument("--database", type = str, default='DB2')
    parser.add_argument("--model_type", default='DNN_feature')
    parser.add_argument("--en_train", action='store_true', default=False)
    parser.add_argument("--notBetterCount", type=int, default=50)
    parser.add_argument("--schedulerOn", type=bool, default=False)
    parser.add_argument("--pretrain_model_PATH", default='None')
    parser.add_argument("--load_model", default=True)
    parser.add_argument("--class_rest", action='store_true', default=False)
    parser.add_argument("--feat_extract", action='store_true', default=False)
    parser.add_argument("--load_dataset", action='store_true', default=False)
    parser.add_argument("--save_dataset", action='store_true', default=False)
    parser.add_argument("--log_name", default='Lastest_results')

    if raw_args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(raw_args)

    return args

def get_args_info(args):
    window_size = int(args.window_size_sec * args.fs) # 200 ms => 0.2 * fs = 400 sample points
    window_step = int(args.window_step_sec * args.fs) # 100 ms => 0.1 * fs = 200 sample points
    model_PATH = f'./Results/{args.log_name}/{args.model_type}.pth' 
    exercise_list_np = np.array(args.exercise_list)
    number_gesture = int(np.any(exercise_list_np==1))*17 + int(np.any(exercise_list_np==2))*23 + int(np.any(exercise_list_np==3))*9
    device = "cuda" if torch.cuda.is_available() else "cpu"
    notBetterCount = args.notBetterCount
    schedulerOn = args.schedulerOn

    print("\n"+"="*70)

    print("-"*70)
    print("Database: NinaPro ", args.database)
    print("Number of subjects: ", len(args.subject_list), " -> {} subject".format(args.subject_list))
    print("Number of  exercises: ", len(args.exercise_list), " -> {} exercise".format(args.exercise_list))
    print("Number of gestures: ", number_gesture, end='')
    print(", and we {}cludes \'rest\' class".format("in" if args.class_rest else "ex"))

    print("-"*70)
    print("Sampling rate: ", args.fs, " (sps)")
    print("Number of channels: ", args.num_channel)
    print("Window size: ", window_size, " (samples) = ", 1e3*args.window_size_sec, " (ms)")
    print("Window step: ", window_step, " (samples) = ", 1e3*args.window_step_sec, " (ms)")

    print("-"*70)
    print("Device: ", device)
    print("Feature extraction: ", args.feat_extract)
    print("Model type: ", args.model_type)
    print("Model PATH: ", model_PATH)
    print("Load model: ", args.load_model)
    print("Number of epochs: ", args.num_epoch)
    print("Batch size: ", args.batch_size)
    print("Learning rate: ", args.lr)
    print("Not Better Count", args.notBetterCount)
    print("Scheduler", args.schedulerOn)

    print("-"*70)
    print("="*70)
    print("", flush=True)

    return window_size, window_step, number_gesture, model_PATH, device, notBetterCount, schedulerOn

def print_args(args):

    print("\n"+"="*25+" Print all args value "+"="*25)
    for key, value in vars(args).items():
        print(f"{key} = {value}")

    print("="*70)
    print("", flush=True)

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
