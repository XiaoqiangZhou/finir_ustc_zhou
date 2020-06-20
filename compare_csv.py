import csv
import argparse
from toy_experiments import read_csv

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Script to reimplement our result')
    parser.add_argument('-f', '--first', type=str)
    parser.add_argument('-s', '--second', type=str)
    args = parser.parse_args()
    return args

def compare_csv(first_file_path, second_file_path):
    first_content = read_csv(first_file_path)
    second_content = read_csv(second_file_path)
    first_prediction = [float(_) for _ in first_content['label']]
    second_prediction = [float(_) for _ in second_content['label']]
    same_count = 0
    print('Length is ', len(first_prediction))
    for i in range(len(first_prediction)):
        if first_prediction[i] == second_prediction[i]:
            same_count += 1
    print("Same rate is %.2f"%(same_count/len(first_prediction)))

if __name__ == "__main__":
    args = parse_args()
    compare_csv(args.first, args.second)