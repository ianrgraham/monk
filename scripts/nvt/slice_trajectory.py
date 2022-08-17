"""Slice a hoomd trajectory given by the supplied paramters"""

import argparse
import pathlib
from typing import Optional

import gsd.hoomd

valid_input_formats = [".gsd"]
valid_output_formats = [".gsd"]


def optional_int(value: str):
    """Converts strings to Option[int]"""
    try:
        return int(value)
    except ValueError:
        if value == "None":
            return None
        else:
            raise ValueError


parser = argparse.ArgumentParser(
    description="Slice a hoomd trajectory and store \
    the subslice to a new file")
parser.add_argument("ifile",
                    type=str,
                    help=f"Input file (allowed formats: {valid_input_formats}")
parser.add_argument(
    "ofile",
    type=str,
    help=f"Output file (allowed formats: {valid_output_formats}")
parser.add_argument("--slice",
                    nargs="+",
                    type=optional_int,
                    help="Subslice to apply to the input trajectory.",
                    default=[None])
parser.add_argument("--mode", default="xb", help="Write mode.")

args = parser.parse_args()

ifile = pathlib.Path(args.ifile)
ofile = pathlib.Path(args.ofile)

slc = slice(*args.slice)
mode = args.mode

input_traj = gsd.hoomd.open(ifile, mode="rb")
output_traj = gsd.hoomd.open(ofile, mode=mode)

for snapshot in input_traj[slc]:
    output_traj.append(snapshot)

input_traj.close()
output_traj.close()
