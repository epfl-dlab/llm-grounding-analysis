import json
from abc import ABC, abstractmethod
from collections import defaultdict
from contextlib import AbstractContextManager
from typing import Dict

from src.fact import Fact, fact_from_dict
from src.utils.logger import get_logger


class ResumeAndSaveDataset(AbstractContextManager, ABC):
    """
    A context manager that resumes processing a data from where it left off, saves the output periodically,
    and ensures the output is saved in case of an exception.
    """

    def __init__(self, path, save_interval=1000):
        """
        Initializes the ResumeAndSaveDataset context manager.

        Args:
            path (str): The path of the data file.
            save_interval (int, optional): The number of entries to process before saving the output data.
                                           Defaults to 20.
        """
        self.path = path
        self.output_dataset = self.load_output_dataset()
        self.save_interval = save_interval
        self.entries_since_last_save = 0

    def load_output_dataset(self):
        """
        Loads the output data from the specified file. If the file is not found, an empty list is returned.

        Returns:
            List: The output data loaded from the file or an empty list if the file is not found.
        """
        try:
            with open(self.path, "r") as f:
                data = json.load(f)
            get_logger().info(f"Loaded {len(data)} previously computed entries from the data stored in {self.path}.")
            return data
        except FileNotFoundError:
            return []

    @abstractmethod
    def is_input_processed(self, inp: dict):
        """
        Check if the input dictionary has been processed.

        Args:
            inp (dict): A dictionary containing the input data to be checked.

        Returns:
            bool: True if the input has been processed, False otherwise.
        """
        pass

    def add_entry(self, entry):
        """
        Appends a new data entry to the output_dataset and saves the output data if the save_interval is reached.

        Args:
            entry: The data entry to be added.
        """
        self.output_dataset.append(entry)
        self.entries_since_last_save += 1

        if self.entries_since_last_save >= self.save_interval:
            self.save_output_dataset()
            self.entries_since_last_save = 0

    def save_output_dataset(self):
        """
        Saves the output data to the file.
        """
        with open(self.path, "w+") as f:
            json.dump(self.output_dataset, f, indent=4)

    def __enter__(self):
        """
        The enter method for the context manager.

        Returns:
            ResumeAndSaveDataset: The instance of the context manager.
        """
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """
        The exit method for the context manager. Saves the output data to the file and prints a message.

        Args:
            exc_type: The type of the exception, if any.
            exc_value: The instance of the exception, if any.
            traceback: A traceback object, if any.

        Returns:
            bool: False to propagate the exception, True to suppress it.
        """
        self.save_output_dataset()
        if exc_type is not None:
            get_logger().info(f"Output data saved in the following location due to an exception: {self.path}")
        else:
            get_logger().info(f"Output data saved in the following location: {self.path}")
        return False


class ResumeAndSaveFactDataset(ResumeAndSaveDataset):
    def __init__(self, path, save_interval=20):
        super().__init__(path, save_interval)
        self.entry_processed = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: False))))

        for entry in self.output_dataset:
            entry_fact = fact_from_dict(entry["fact"] if "fact" in entry else entry)
            subject = entry_fact.get_subject()
            rel = entry_fact.get_relation_property_id()
            obj = entry_fact.get_object()
            intermediate_paragraph = entry_fact.get_intermediate_paragraph()
            self.entry_processed[subject][rel][obj][intermediate_paragraph] = True

    def is_input_processed(self, fact: Fact):
        return self.entry_processed[fact.get_subject()][fact.get_relation_property_id()][fact.get_object()][
            fact.get_intermediate_paragraph()
        ]

    def add_entry(self, entry: Dict):
        super().add_entry(entry)

        if "fact" in entry:
            fact = fact_from_dict(entry["fact"])
        else:
            fact = fact_from_dict(entry)

        self.entry_processed[fact.get_subject()][fact.get_relation_property_id()][fact.get_object()][
            fact.get_intermediate_paragraph()
        ] = True
