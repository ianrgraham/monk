import argparse
import signac
import flow

class MonkProject(flow.FlowProject):

    def main(self):

        parser = argparse.ArgumentParser()

        super(MonkProject).main(parser=parser)