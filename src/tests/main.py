import os
import sys
from unittest import TestLoader, TextTestRunner

dir_name = os.path.abspath(os.path.dirname(__file__))
src_path = os.path.dirname(dir_name)


def run():
    loader = TestLoader()
    test = loader.discover(dir_name)
    runner = TextTestRunner()
    runner.run(test)


if __name__ == "__main__":
    # For import Statement
    sys.path.append(src_path)

    # Run tests
    run()
