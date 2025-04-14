"""
COMP20008 Elements of Data Processing
2025 Semester 1
Assignment 1

Solution: main file

DO NOT CHANGE THIS FILE!
"""

import os
import sys


def verify_task4_1():
    try:
        from task4_1 import task4_1
    except ImportError:
        print("Task 4.1's function not found.")
        return

    print("=" * 80)
    print("Executing Task 4.1...\n")
    task4_1()

    print("Checking Task 4.1's output...\n")
    for expected_file in ["task4_1_carstat.json", "task4_1_stackbar.png"]:
        if os.path.isfile(expected_file):
            print(f"\tTask 4.1's {expected_file} output found.\n")
            if os.path.getsize(expected_file) == 0:
                print(f"\t❗ Task 4.1's {expected_file} output has size zero - please verify it uploaded correctly.\n")
        else:
            print(f"\t❗ Task 4.1's {expected_file} output NOT found. Please check your code.\n")

    print("Finished Task 4.1")
    print("=" * 80)


def verify_task4_2():

    try:
        from task4_2 import task4_2
    except ImportError:
        print("Task 4.2's function not found.")
        return

    print("=" * 80)
    print("Executing Task 4.2...\n")
    task4_2()

    print("Checking Task 4.2's output...\n")
    for expected_file in ["task4_2_heatmap.png"]:
        if os.path.isfile(expected_file):
            print(f"\tTask 4.2's {expected_file} output found.\n")
            if os.path.getsize(expected_file) == 0:
                print(f"\t❗ Task 4.2's {expected_file} output has size zero - please verify it uploaded correctly.\n")
        else:
            print(f"\t❗ Task 4.2's {expected_file} output NOT found. Please check your code.\n")

    print("Finished Task 4.2")
    print("=" * 80)


def verify_task4_3():

    try:
        from task4_3 import task4_3
    except ImportError:
        print("Task 4.3's function not found.")
        return

    print("=" * 80)
    print("Executing Task 4.3...\n")
    task4_3()

    print("Checking Task 4.3's output...\n")
    for expected_file in ["task4_3_diversity.json", "task4_3_probabilities.json"]:
        if os.path.isfile(expected_file):
            print(f"\tTask 4.3's {expected_file} output found.\n")
            if os.path.getsize(expected_file) == 0:
                print(f"\t❗ Task 4.3's {expected_file} output has size zero - please verify it uploaded correctly.\n")
        else:
            print(f"\t❗ Task 4.3's {expected_file} output NOT found. Please check your code.\n")
    
    print("Finished executing Task 4.3")
    print("=" * 80)


def main():
    args = sys.argv
    assert len(args) >= 2, "Please provide a task."
    task = args[1]
    valid_tasks = ["all"] + ["task4_" + str(i) for i in range(1, 4)]
    assert task in valid_tasks, \
        f"Invalid task \"{task}\", options are: {valid_tasks}."
    if task == "task4_1":
        verify_task4_1()
    elif task == "task4_2":
        verify_task4_2()
    elif task == "task4_3":
        verify_task4_3()
    elif task == "all":
        verify_task4_1()
        verify_task4_2()
        verify_task4_3()



