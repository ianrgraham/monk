from monk import methods

import pytest


def test_log_trigger():
    trigger = methods.LogTrigger(10, 0, 0.1, -1)

    for i in range(10001):
        if trigger(i):
            # pass
            print(i)


if __name__ == "__main__":
    test_log_trigger()
