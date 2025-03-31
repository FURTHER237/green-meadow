"""
COMP20008 Elements of Data Processing
2025 Semester 1
Assignment 1

Solution: main file

DO NOT CHANGE THIS FILE!
"""

import os
import sys


def verify_task3_1():
    try:
        from task3_1 import task3_1
    except ImportError:
        print("Task 3.1's function not found.")
        return

    print("=" * 80)
    print("Executing Task 3.1...\n")
    task3_1()

    print("Checking Task 3.1's output...\n")
    for expected_file in ["task3_1_scatter.png"]:
        if os.path.isfile(expected_file):
            print(f"\tTask 3.1's {expected_file} output found.\n")
            if os.path.getsize(expected_file) == 0:
                print(f"\t❗ Task 3.1's {expected_file} output has size zero - please verify it uploaded correctly.\n")
        else:
            print(f"\t❗ Task 3.1's {expected_file} output NOT found. Please check your code.\n")

    print("Finished Task 3.1")
    print("=" * 80)


def verify_task3_2():

    try:
        from task3_2 import task3_2
    except ImportError:
        print("Task 3.2's function not found.")
        return

    print("=" * 80)
    print("Executing Task 3.2...\n")
    task3_2()

    print("Checking Task 3.2's output...\n")
    for expected_file in ["task3_2_elbow.png"]:
        if os.path.isfile(expected_file):
            print(f"\tTask 3.2's {expected_file} output found.\n")
            if os.path.getsize(expected_file) == 0:
                print(f"\t❗ Task 3.2's {expected_file} output has size zero - please verify it uploaded correctly.\n")
        else:
            print(f"\t❗ Task 3.2's {expected_file} output NOT found. Please check your code.\n")

    print("Finished Task 3.2")
    print("=" * 80)


def verify_task3_3():

    try:
        from task3_3 import task3_3
    except ImportError:
        print("Task 3.3's function not found.")
        return

    print("=" * 80)
    print("Executing Task 3.3...\n")
    task3_3()

    print("Checking Task 3.3's output...\n")
    for expected_file in ["task3_3_scattercolour.png", "task3_3_cluster0.csv"]:
        if os.path.isfile(expected_file):
            print(f"\tTask 3.3's {expected_file} output found.\n")
            if os.path.getsize(expected_file) == 0:
                print(f"\t❗ Task 3.3's {expected_file} output has size zero - please verify it uploaded correctly.\n")
        else:
            print(f"\t❗ Task 3.3's {expected_file} output NOT found. Please check your code.\n")
    
    files_in_directory = [f for f in os.listdir("/home/") if "task3_3_cluster" in f and f[-4:] == ".csv"]
    print(f"Found {len(files_in_directory)} files that contained task3_3_clusterX.csv - {files_in_directory}")
    

    print("Finished executing Task 3.3")
    print("=" * 80)


def main():
    args = sys.argv
    assert len(args) >= 2, "Please provide a task."
    task = args[1]
    valid_tasks = ["all"] + ["task3_" + str(i) for i in range(1, 4)]
    assert task in valid_tasks, \
        f"Invalid task \"{task}\", options are: {valid_tasks}."
    if task == "task3_1":
        verify_task3_1()
    elif task == "task3_2":
        verify_task3_2()
    elif task == "task3_3":
        verify_task3_3()
    elif task == "all":
        verify_task3_1()
        verify_task3_2()
        verify_task3_3()

if __name__ == "__main__":
    main()

