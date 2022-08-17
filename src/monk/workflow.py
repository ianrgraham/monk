import argparse
import signac
import flow
import yaml
import re
import os

path_matcher = re.compile(r'\$\{([^}^{]+)\}')


def path_constructor(loader, node):
    """Extract the matched value, expand env variable, and replace the match."""
    value = node.value
    match = path_matcher.match(value)
    env_var = match.group()[2:-1]
    return os.environ.get(env_var) + value[match.end():]


yaml.add_implicit_resolver('!path', path_matcher)
yaml.add_constructor('!path', path_constructor)


def get_config(path: str = None):
    """Load the project configuration.

    By default, searches for `config.yaml` in the current working directory.
    """
    if path is None:
        path = "config.yaml"
    with open("config.yaml") as file:
        config = yaml.load(file, yaml.FullLoader)

    assert "root" in config  # assert the a workspace directory is specified
    return config


class MonkProject(flow.FlowProject):

    def main(self):

        parser = argparse.ArgumentParser()

        super(MonkProject).main(parser=parser)
