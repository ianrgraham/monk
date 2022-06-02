"""Workflow to test the dynamical and structural properties of various systems """

import flow
import hoomd
import numpy as np
import signac
import os.path

import argparse
import sys
import inspect
import logging
import traceback
import subprocess

from flow.project import UserConditionError, UserOperationError, SubmitError, \
    Jinja2TemplateNotFound, IgnoreConditions, _IgnoreConditionsConversion

from monk import project_path, project_view, safe_clean_signac_project, grid
from monk import pair, prep, methods

class Project(flow.FlowProject):

    def _new_main(self, parser=None, subparsers=None, base_parser=None):
        """Reimplementation of the main function from `flow.FlowProject`"""
        # Find file that main is called in. When running through the command
        # line interface, we know exactly what the entrypoint path should be:
        # it's the file where main is called, which we can pull off the stack.
        self._entrypoint.setdefault(
            "path", os.path.realpath(inspect.stack()[-1].filename)
        )

        if parser is None:
            parser = argparse.ArgumentParser()
            if subparsers is None:
                subparsers = parser.add_subparsers()

        if base_parser is None:
            base_parser = argparse.ArgumentParser(add_help=False)

            # The argparse module does not automatically merge options shared between the main
            # parser and the subparsers. We therefore assign different destinations for each
            # option and then merge them manually below.
            for prefix, _parser in (("main_", parser), ("", base_parser)):
                _parser.add_argument(
                    "-v",
                    "--verbose",
                    dest=prefix + "verbose",
                    action="count",
                    default=0,
                    help="Increase output verbosity.",
                )
                _parser.add_argument(
                    "--show-traceback",
                    dest=prefix + "show_traceback",
                    action="store_true",
                    help="Show the full traceback on error.",
                )
                _parser.add_argument(
                    "--debug",
                    dest=prefix + "debug",
                    action="store_true",
                    help="This option implies `-vv --show-traceback`.",
                )

        parser_status = subparsers.add_parser(
            "status",
            parents=[base_parser],
            description="Parallelization of the status command can be "
            "controlled by setting the flow.status_parallelization config "
            "value to 'thread' (default), 'none', or 'process'. To do this, "
            "execute `signac config set flow.status_parallelization VALUE`.",
        )
        self._add_print_status_args(parser_status)
        parser_status.add_argument(
            "--profile",
            const=inspect.getsourcefile(inspect.getmodule(self)),
            nargs="?",
            help="Collect statistics to determine code paths that are responsible "
            "for the majority of runtime required for status determination. "
            "Optionally provide a filename pattern to select for what files "
            "to show result for. Defaults to the main module. "
            "(requires pprofile)",
        )
        parser_status.set_defaults(func=self._main_status)

        parser_next = subparsers.add_parser(
            "next",
            parents=[base_parser],
            description="Determine jobs that are eligible for a specific operation.",
        )
        parser_next.add_argument("name", type=str, help="The name of the operation.")
        parser_next.set_defaults(func=self._main_next)

        parser_run = subparsers.add_parser(
            "run",
            parents=[base_parser],
        )
        self._add_operation_selection_arg_group(parser_run)

        execution_group = parser_run.add_argument_group("execution")
        execution_group.add_argument(
            "--pretend",
            action="store_true",
            help="Do not actually execute commands, just show them.",
        )
        execution_group.add_argument(
            "--progress",
            action="store_true",
            help="Display a progress bar during execution.",
        )
        execution_group.add_argument(
            "--num-passes",
            type=int,
            default=1,
            help="Specify how many times a particular job-operation may be executed within one "
            "session (default=1). This is to prevent accidental infinite loops, "
            "where operations are executed indefinitely, because postconditions "
            "were not properly set. Use -1 to allow for an infinite number of passes.",
        )
        execution_group.add_argument(
            "-t",
            "--timeout",
            type=float,
            help="A timeout in seconds after which the execution of one operation is canceled.",
        )
        execution_group.add_argument(
            "--switch-to-project-root",
            action="store_true",
            help="Temporarily add the current working directory to the python search path and "
            "switch to the root directory prior to execution.",
        )
        execution_group.add_argument(
            "-p",
            "--parallel",
            type=int,
            nargs="?",
            const="-1",
            help="Specify the number of cores to parallelize to. Defaults to all available "
            "processing units.",
        )
        execution_group.add_argument(
            "--order",
            type=str,
            choices=["none", "by-job", "cyclic", "random"],
            default=None,
            help="Specify the execution order of operations for each execution pass.",
        )
        execution_group.add_argument(
            "--ignore-conditions",
            type=str,
            choices=["none", "pre", "post", "all"],
            default=IgnoreConditions.NONE,
            action=_IgnoreConditionsConversion,
            help="Specify conditions to ignore for eligibility check.",
        )
        parser_run.set_defaults(func=self._main_run)

        parser_submit = subparsers.add_parser(
            "submit",
            parents=[base_parser],
            conflict_handler="resolve",
        )
        self._add_submit_args(parser_submit)
        env_group = parser_submit.add_argument_group(
            f"{self._environment.__name__} options"
        )
        self._environment.add_args(env_group)
        parser_submit.set_defaults(func=self._main_submit)
        print(
            "Using environment configuration:",
            self._environment.__name__,
            file=sys.stderr,
        )

        parser_exec = subparsers.add_parser(
            "exec",
            parents=[base_parser],
        )
        parser_exec.add_argument(
            "operation",
            type=str,
            choices=list(sorted(self._operations)),
            help="The operation to execute.",
        )
        parser_exec.add_argument(
            "job_id",
            type=str,
            nargs="*",
            help="The job ids or aggregate ids in the FlowProject. "
            "Defaults to all jobs and aggregates.",
        )
        parser_exec.set_defaults(func=self._main_exec)

        args = parser.parse_args()
        if not hasattr(args, "func"):
            parser.print_usage()
            sys.exit(2)

        # Manually 'merge' the various global options defined for both the main parser
        # and the parent parser that are shared by all subparsers:
        for dest in ("verbose", "show_traceback", "debug"):
            setattr(args, dest, getattr(args, "main_" + dest) or getattr(args, dest))
            delattr(args, "main_" + dest)

        # Read the config file and set the internal flag.
        # Do not overwrite with False if not present in config file
        if self._flow_config["show_traceback"]:
            args.show_traceback = True

        if args.debug:  # Implies '-vv' and '--show-traceback'
            args.verbose = max(2, args.verbose)
            args.show_traceback = True

        # Support print_status argument alias
        if args.func == self._main_status and args.full:
            args.detailed = args.all_ops = True

        # Empty parameters argument on the command line means: show all varying parameters.
        if hasattr(args, "parameters"):
            if args.parameters is not None and len(args.parameters) == 0:
                args.parameters = self.PRINT_STATUS_ALL_VARYING_PARAMETERS

        # Set verbosity level according to the `-v` argument.
        logging.basicConfig(level=max(0, logging.WARNING - 10 * args.verbose))

        def _show_traceback_and_exit(error):
            is_user_error = isinstance(error, (UserOperationError, UserConditionError))
            if is_user_error:
                # Always show the user traceback cause.
                error = error.__cause__
            if args.show_traceback or is_user_error:
                traceback.print_exception(type(error), error, error.__traceback__)
            if not args.show_traceback:
                print(
                    "Execute with '--show-traceback' or '--debug' to show the "
                    "full traceback.",
                    file=sys.stderr,
                )
            sys.exit(1)

        try:
            args.func(args)
        except SubmitError as error:
            print("Submission error:", error, file=sys.stderr)
            _show_traceback_and_exit(error)
        except (TimeoutError, subprocess.TimeoutExpired) as error:
            print(
                "Error: Failed to complete execution due to "
                f"timeout ({args.timeout} seconds).",
                file=sys.stderr,
            )
            _show_traceback_and_exit(error)
        except Jinja2TemplateNotFound as error:
            print(f"Did not find template script '{error}'.", file=sys.stderr)
            _show_traceback_and_exit(error)
        except AssertionError as error:
            if not args.show_traceback:
                print(
                    "ERROR: Encountered internal error during program execution.",
                    file=sys.stderr,
                )
            _show_traceback_and_exit(error)
        except (UserOperationError, UserConditionError) as error:
            if str(error):
                print(f"ERROR: {error}\n", file=sys.stderr)
            else:
                print(
                    "ERROR: Encountered error during program execution.\n",
                    file=sys.stderr,
                )
            _show_traceback_and_exit(error)
        except Exception as error:
            if str(error):
                print(
                    "ERROR: Encountered error during program execution: "
                    f"'{error}'\n",
                    file=sys.stderr,
                )
            else:
                print(
                    "ERROR: Encountered error during program execution.\n",
                    file=sys.stderr,
                )
            _show_traceback_and_exit(error)


    def _main_init(self, args):

        steps = args.steps
        if steps is None:
            steps = 100_000_000

        equil_steps = args.equil_steps
        if equil_steps is None:
            equil_steps = 1_000_000

        is_test = args.test

        if is_test:
            num_iter = 1
        else:
            num_iter = 3
        
        project = self
        
        if "initialized" not in project.doc:
            print("Initializing project")
            project.doc["initialized"] = True
        else:
            raise RuntimeError("Project has already been initialized")

        if "cur_seed" not in project.doc:
            project.doc["cur_seed"] = 0

        # Initialize the data space

        statepoint_grid_ka_lj = {
            "it": range(num_iter), 
            "phi": [1.0, 1.2, 1.3],
            "A_frac": [80, 70, 60]
        }

        for sp in grid(statepoint_grid_ka_lj):
            universal = dict(N=10_000, init_kT=1.4, final_kT=0.4, dt=2.5e-3, steps=steps, equil_steps=equil_steps, dumps=100)
            sp.update(universal)
            job = project.open_job(sp).init()
            if "init" not in job.doc:
                job.document["seed"] = project.doc["cur_seed"]
                job.document['init'] = True
                project.doc["cur_seed"] += 1

    def _main_clear(self, _args):
        safe_clean_signac_project(self.root_directory(), prepend_monk=False)

    def main(self):

        parser = argparse.ArgumentParser()
        base_parser = argparse.ArgumentParser(add_help=False)

        for prefix, _parser in (("main_", parser), ("", base_parser)):
            _parser.add_argument(
                "-v",
                "--verbose",
                dest=prefix + "verbose",
                action="count",
                default=0,
                help="Increase output verbosity.",
            )
            _parser.add_argument(
                "--show-traceback",
                dest=prefix + "show_traceback",
                action="store_true",
                help="Show the full traceback on error.",
            )
            _parser.add_argument(
                "--debug",
                dest=prefix + "debug",
                action="store_true",
                help="This option implies `-vv --show-traceback`.",
            )

        subparsers = parser.add_subparsers()

        parser_init = subparsers.add_parser(
            "init",
            parents=[base_parser],
            description="Initialize statepoints for processing.",
        )
        parser_init.add_argument(
            "--steps",
            default=None,
            type=int,
            help="Total simulation time",
        )
        parser_init.add_argument(
            "--equil-steps",
            default=None,
            type=int,
            help="Total time spent equilibrating the simulation",
        )
        parser_init.add_argument(
            "--test",
            action=argparse._StoreTrueAction,
            help="Run parameters for testing",
        )
        parser_init.set_defaults(func=self._main_init)
        

        parser_clear = subparsers.add_parser(
            "clear",
            parents=[base_parser],
            description="Clear current state points",
        )
        parser_clear.set_defaults(func=self._main_clear)


        self._new_main(parser=parser, subparsers=subparsers, base_parser=base_parser)


@Project.operation
@Project.pre.true('init')
@Project.post.true('init_state')
def init_state(job: signac.Project.Job):
    N = job.sp["N"]
    phi = job.sp["phi"]
    seed = job.doc["seed"]
    A_frac = job.sp["A_frac"]

    device = hoomd.device.auto_select()
    sim = hoomd.Simulation(device, seed=seed)

    rng = prep.init_rng(seed)
    L = prep.len_from_phi(N, phi)
    snap = prep.approx_euclidean_snapshot(
        N, L, rng, dim=3, particle_types=["A", "B"], ratios=[A_frac, 100-A_frac], diams=[1.0, 0.88]
    )
    sim.create_state_from_snapshot(snap)

    hoomd.write.GSD.write(sim.state, job.fn("init.gsd"))

    job.doc["init_state"] = True


@Project.operation
@Project.pre.after(init_state)
@Project.post.true('simulated')
def run_nvt_sim(job: signac.Project.Job):
    seed = job.doc["seed"]
    init_kT = job.sp["init_kT"]
    final_kT = job.sp["final_kT"]
    dt = job.sp["dt"]
    equil_steps = job.sp["equil_steps"]
    steps = job.sp["steps"]
    dumps = job.sp["dumps"]

    device = hoomd.device.auto_select()
    sim = hoomd.Simulation(device, seed=seed)

    sim.create_state_from_gsd(job.fn("init.gsd"))

    integrator = hoomd.md.Integrator(dt=dt)
    nlist = hoomd.md.nlist.Tree(0.2)
    pot_pair = pair.KA_LJ(nlist)
    integrator.forces.append(pot_pair)

    tstart = sim.timestep + equil_steps
    tramp = tstart + steps

    kT_variant = hoomd.variant.Ramp(init_kT, final_kT, tstart, tramp)

    class VariantLogger:

        def __init__(self, sim, variant):
            self._sim = sim
            self._variant = variant

        @property
        def value(self):
            return self._variant(self._sim.timestep)

    nvt = hoomd.md.methods.NVT(
        kT=kT_variant,
        filter=hoomd.filter.All(),
        tau=1.0
    )

    integrator.methods.append(nvt)
    sim.operations.integrator = integrator

    thermodynamic_properties = hoomd.md.compute.ThermodynamicQuantities(
    filter=hoomd.filter.All())
    sim.operations.computes.append(thermodynamic_properties)

    print("Thermalizing system")
    sim.run(0)
    nvt.thermalize_thermostat_dof()

    print(nvt.loggables)
    print(kT_variant(sim.timestep))

    kT_logger = VariantLogger(sim, kT_variant)

    logger = hoomd.logging.Logger(categories=['scalar'])
    logger.add(sim, quantities=["timestep"])
    logger[('NVT', 'kT')] = (kT_logger, 'value', 'scalar')
    logger.add(thermodynamic_properties, quantities=["kinetic_temperature", "kinetic_energy", "potential_energy"])

    table = hoomd.write.Table(trigger=hoomd.trigger.Periodic(period=10_000),
                          logger=logger)

    gsd_writer = hoomd.write.GSD(
        filename=job.fn("equil.gsd"),
        trigger=hoomd.trigger.Periodic(int(equil_steps/dumps), phase=sim.timestep),
        mode='wb',
        filter=hoomd.filter.All(),
        log=logger
    )
    sim.operations.writers.append(table)
    sim.operations.writers.append(gsd_writer)

    print("Performing initial equilibration")

    sim.run(equil_steps, True)

    sim.operations.writers.pop()
    del gsd_writer

    print("Running quench to the glassy phase")

    gsd_writer = hoomd.write.GSD(
        filename=job.fn("traj.gsd"),
        trigger=hoomd.trigger.Periodic(int(steps/dumps), phase=sim.timestep),
        mode='wb',
        filter=hoomd.filter.All(),
        log=logger
    )

    sim.operations.writers.append(gsd_writer)

    sim.run(steps+1, True)

    job.doc["simulated"] = True

@Project.operation
@Project.pre.after(run_nvt_sim)
@Project.post.true('validated')
def validate(job: signac.Project.Job):
    pass


project: Project = Project.init_project(name="GenGlassStates3D", root=project_path("initial-configs/3d-glass/param_explor"))

if __name__ == '__main__':
    project.main()