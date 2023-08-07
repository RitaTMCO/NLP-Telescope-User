import sys

from .classification import Classification

from telescope.utils import read_yaml_file

AVAILABLE_TASKS = [
    Classification
]

names_availabels_tasks = {task.name:task for task in AVAILABLE_TASKS}

tasks_yaml = read_yaml_file("tasks.yaml")

try:
    AVAILABLE_CLASSIFICATION_TASKS = [names_availabels_tasks[task_name] for task_name in tasks_yaml["Classification tasks"]]

except KeyError as error:
    print("Error (yaml): " + str(error) + " as a classification task is not available.")
    sys.exit(1)