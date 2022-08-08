import argparse
import signac
import flow
import yaml

def get_config(path: str = None):
    """Load the project configuration.

    By default, searches for `config.yaml` in the current working directory.
    """
    if path is None:
        path = "config.yaml"
    with open("config.yaml") as file:
        config = yaml.load(file)
    
    assert "root" in config  # assert the a workspace directory is specified
    return config


class MonkProject(flow.FlowProject):

    def main(self):

        parser = argparse.ArgumentParser()

        super(MonkProject).main(parser=parser)