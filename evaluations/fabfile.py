# fabfile.py
from fabric import task
from pathlib import Path

import shlex
import sys

python_bin = sys.executable

# Different cost values for the simulation runs
COSTS = ["0.0", "0.001", "0.002", "0.003"]

# Common arguments shared across all runs
BASE_MSP_ARGS = [
    "--slots 10000",
    "--validators 1000",
    "--time_window 10000",
    "--model MSP",
]

BASE_SSP_ARGS = [
    "--slots 10000",
    "--validators 1000",
    "--time_window 10000",
    "--model SSP",
]


def _tmux(c, cmd, **kwargs):
    """Run a tmux command and return the result."""
    return c.run(f"tmux {cmd}", warn=True, pty=False, hide=True, **kwargs)


def _has_session(c, session):
    """Check if a tmux session exists."""
    r = _tmux(c, f"has-session -t {shlex.quote(session)}")
    return r.ok


def _ensure_session(c, session):
    """Create a tmux session if it does not exist yet."""
    if not _has_session(c, session):
        _tmux(c, f"new-session -d -s {shlex.quote(session)}")


def _quoted(s: str) -> str:
    """Helper for shell-quoting strings safely."""
    return shlex.quote(s)


def _build_cmd(model: str, root: Path, config_path: str, outdir: str, appended_args: list) -> str:
    """Build the command string for one simulation run."""
    if model == "SSP":
        args = [f"--config {config_path}"] + BASE_SSP_ARGS + [f"--output-dir {outdir}"] + appended_args
    elif model == "MSP":
        args = [f"--config {config_path}"] + BASE_MSP_ARGS + [f"--output-dir {outdir}"] + appended_args
    else:
        raise ValueError(f"Unknown model type: {model}")
    # Each command first changes into the parent directory before running
    return f"cd {_quoted(str(root))} && {python_bin} simulation.py " + " ".join(args)


@task
def run_baseline(c, session="simulation-baseline"):
    """
    Run four baseline evaluations in parallel inside tmux panes.
    
    Usage:
        fab run_baseline             # default session name is 'simulation-baseline'
        fab run_baseline:session=foo # custom session name
    """
    # Resolve parent directory (equivalent to cd "$SCRIPT_DIR/..")
    script_dir = Path(__file__).resolve().parent
    root = script_dir.parent

    # Make sure the tmux session exists
    _ensure_session(c, session)

    num_jobs = 0

    # Create a new window within the session
    for model in ["SSP", "MSP"]:
        _tmux(c, f"new-window -t {session} -n {model}")
        window = f"{session}:{model}"

        config_path = f"params/{model}-baseline.yaml"
        outdir = f"output/baseline/{model}/validators_1000_slots_10000"

        # Prepare commands for all cost values
        cmds = [
            _build_cmd(model, root, config_path, f"{outdir}_cost_{cost}", [f"--cost {cost}"])
            for cost in COSTS
        ]
        num_jobs += len(cmds)
        
        # Run the first command in the first pane
        _tmux(c, f"select-window -t {window}")
        _tmux(c, f"send-keys -t {window}.0 {_quoted(cmds[0])} Enter")

        # For the rest, split the window and run commands in new panes
        for index, cmd in enumerate(cmds[1:], start=1):
            _tmux(c, f"split-window -t {window} -v")  # vertical split; use -h for horizontal
            _tmux(c, f"select-pane -t {window}.{index}")
            _tmux(c, f"send-keys -t {window}.{index} {_quoted(cmd)} Enter")

        # Arrange panes in a tiled layout
        _tmux(c, f"select-layout -t {window} tiled")

    # Print instructions for the user
    print(f"[ok] Started {num_jobs} jobs in tmux session '{session}'.")
    print(f"Attach with: tmux attach -t {session}")


@task
def run_heterogeneous_information_sources(c, session="hetero-info"):
    """
    Run a simulation with heterogeneous information sources in a tmux session.
    
    Usage:
        fab run-heterogeneous-information-sources             # default session name is 'hetero-info'
        fab run-heterogeneous-information-sources:session=foo # custom session name
    """
    # Resolve parent directory (equivalent to cd "$SCRIPT_DIR/..")
    script_dir = Path(__file__).resolve().parent
    root = script_dir.parent

    # Make sure the tmux session exists
    _ensure_session(c, session)
    
    num_jobs = 0

    cost = 0.002
    for model in ["SSP", "MSP"]:
        _tmux(c, f"new-window -t {session} -n {model}")
        window = f"{session}:{model}"
        pane_index = 0

        for latency_mode in ["latency-aligned", "latency-misaligned"]:
            config_path = f"params/{model}-{latency_mode}.yaml"
            outdir = f"output/hetero_info/{model}/validators_1000_slots_10000_cost_{cost}_latency_{latency_mode}"

            cmd = _build_cmd(model, root, config_path, outdir, [f"--cost {cost}", "--info-distribution heterogeneous"])

            if pane_index > 0:
                _tmux(c, f"split-window -t {window} -v")  # vertical split; use -h for horizontal
            
            _tmux(c, f"select-pane -t {window}.{pane_index}")
            _tmux(c, f"send-keys -t {window}.{pane_index} {_quoted(cmd)} Enter")
            pane_index += 1
            num_jobs += 1

        _tmux(c, f"select-layout -t {window} tiled")

    # Print instructions for the user
    print(f"[ok] Started {num_jobs} jobs in tmux session '{session}'.")
    print(f"Attach with: tmux attach -t {session}")


@task
def run_heterogeneous_validators(c, session="hetero-validators"):
    """
    Run a simulation with heterogeneous validators in a tmux session.
    
    Usage:
        fab run-heterogeneous-validators             # default session name is 'hetero-validators'
        fab run-heterogeneous-validators:session=foo # custom session name
    """
    # Resolve parent directory (equivalent to cd "$SCRIPT_DIR/..")
    script_dir = Path(__file__).resolve().parent
    root = script_dir.parent

    # Make sure the tmux session exists
    _ensure_session(c, session)
    
    num_jobs = 0
    
    for model in ["SSP", "MSP"]:
        _tmux(c, f"new-window -t {session} -n {model}")
        window = f"{session}:{model}"
        config_path = f"params/{model}-baseline.yaml"
        pane_index = 0

        for cost in [0.0, 0.002]:
            outdir = f"output/hetero_validators/{model}/slots_10000_cost_{cost}_validators_heterogeneous"
            cmd = _build_cmd(model, root, config_path, outdir, [f"--cost {cost}", "--distribution heterogeneous"])

            if pane_index > 0:
                _tmux(c, f"split-window -t {window} -v")  # vertical split; use -h for horizontal

            _tmux(c, f"select-window -t {window}.{pane_index}")
            _tmux(c, f"send-keys -t {window}.{pane_index} {_quoted(cmd)} Enter")
            pane_index += 1
            num_jobs += 1
        
        _tmux(c, f"select-layout -t {window} tiled")

    # Print instructions for the user
    print(f"[ok] Started {num_jobs} jobs in tmux session '{session}'.")
    print(f"Attach with: tmux attach -t {session}")


@task
def run_hetero_both(c, session="hetero-both"):
    """
    Run a simulation with both heterogeneous validators and information sources in a tmux session.
    
    Usage:
        fab run-hetero-both             # default session name is 'hetero-both'
        fab run-hetero-both:session=foo # custom session name
    """
    # Resolve parent directory (equivalent to cd "$SCRIPT_DIR/..")
    script_dir = Path(__file__).resolve().parent
    root = script_dir.parent

    # Make sure the tmux session exists
    _ensure_session(c, session)
    
    num_jobs = 0

    cost = 0.002
    
    for model in ["SSP", "MSP"]:
        _tmux(c, f"new-window -t {session} -n {model}")
        window = f"{session}:{model}"
        pane_index = 0

        for latency_mode in ["latency-aligned", "latency-misaligned"]:
            config_path = f"params/{model}-{latency_mode}.yaml"
            outdir = f"output/hetero_both/{model}/validators_heterogeneous_slots_10000_cost_{cost}_latency_{latency_mode}"
            cmd = _build_cmd(model, root, config_path, outdir, [f"--cost {cost}", "--distribution heterogeneous", "--info-distribution heterogeneous"])

            if pane_index > 0:
                _tmux(c, f"split-window -t {window} -v")  # vertical split; use -h for horizontal
            
            _tmux(c, f"select-pane -t {window}.{pane_index}")
            _tmux(c, f"send-keys -t {window}.{pane_index} {_quoted(cmd)} Enter")
            pane_index += 1
            num_jobs += 1

        _tmux(c, f"select-layout -t {window} tiled")

    # Print instructions for the user
    print(f"[ok] Started {num_jobs} jobs in tmux session '{session}'.")
    print(f"Attach with: tmux attach -t {session}")
    

@task
def run_different_gammas(c, session="different-gammas"):
    """
    Run simulations with different gamma values in a tmux session.
    
    Usage:
        fab run-different-gammas             # default session name is 'different-gammas'
        fab run-different-gammas:session=foo # custom session name
    """
    # Resolve parent directory (equivalent to cd "$SCRIPT_DIR/..")
    script_dir = Path(__file__).resolve().parent
    root = script_dir.parent

    # Make sure the tmux session exists
    _ensure_session(c, session)
    
    num_jobs = 0

    cost = 0.002
    
    for model in ["SSP", "MSP"]:
        _tmux(c, f"new-window -t {session} -n {model}")
        window = f"{session}:{model}"
        config_path = f"params/{model}-baseline.yaml"

        pane_index = 0

        for gamma in [0.3333, 0.5, 0.6667, 0.8]:
            outdir = f"output/different_gammas/{model}/validators_1000_slots_10000_cost_{cost}_gamma_{gamma}"

            cmd = _build_cmd(model, root, config_path, outdir, [f"--cost {cost}", f"--gamma {gamma}"])

            if pane_index > 0:
                _tmux(c, f"split-window -t {window} -v")  # vertical split; use -h for horizontal
                
            _tmux(c, f"select-window -t {window}.{pane_index}")
            _tmux(c, f"send-keys -t {window}.{pane_index} {_quoted(cmd)} Enter")
            pane_index += 1
            num_jobs += 1

    # Print instructions for the user
    print(f"[ok] Started {num_jobs} jobs in tmux session '{session}'.")
    print(f"Attach with: tmux attach -t {session}")


@task
def run_eip7782(c, session="eip7782"):
    """
    Run simulations with EIP-7782 enabled in a tmux session.
    
    Usage:
        fab run-eip7782             # default session name is 'eip7782'
        fab run-eip7782:session=foo # custom session name
    """
    # Resolve parent directory (equivalent to cd "$SCRIPT_DIR/..")
    script_dir = Path(__file__).resolve().parent
    root = script_dir.parent

    # Make sure the tmux session exists
    _ensure_session(c, session)
    
    num_jobs = 0

    cost = 0.002
    
    for model in ["SSP", "MSP"]:
        _tmux(c, f"new-window -t {session} -n {model}")
        window = f"{session}:{model}"
        config_path = f"params/{model}-baseline.yaml"
        delta = 6000
        cutoff = 3000
        pane_index = 0

        outdir = f"output/eip7782/{model}/validators_1000_slots_10000_cost_{cost}_delta_{delta}_cutoff_{cutoff}"

        cmd = _build_cmd(model, root, config_path, outdir, [f"--cost {cost}", f"--delta {delta}", f"--cutoff {cutoff}"])

        _tmux(c, f"select-window -t {window}.{pane_index}")
        _tmux(c, f"send-keys -t {window}.{pane_index} {_quoted(cmd)} Enter")
        pane_index += 1
        num_jobs += 1

    # Print instructions for the user
    print(f"[ok] Started {num_jobs} jobs in tmux session '{session}'.")
    print(f"Attach with: tmux attach -t {session}")


@task
def run_multiple_proposers_baseline(c, session="multiple-proposers-baseline", hetero_info=False):
    """
    Run simulations with multiple proposers per slot in a tmux session.

    Usage:
        fab run-multiple-proposers-baseline
        fab run-multiple-proposers-baseline --hetero-info
        fab run-multiple-proposers-baseline:session=foo
        fab run-multiple-proposers-baseline:session=foo --hetero-info
    """
    script_dir = Path(__file__).resolve().parent
    root = script_dir.parent

    _ensure_session(c, session)

    cost = 0

    # model is MSP only for this evaluation
    for num_proposers in [1, 2, 4, 8]:
        _tmux(c, f"new-window -t {session} -n proposers_{num_proposers}")
        window = f"{session}:proposers_{num_proposers}"

        config_path = "params/MSP-baseline.yaml"

        suffix = "_hetero_info" if hetero_info else ""
        outdir = (
            f"output/multiple_proposers_baseline/"
            f"validators_1000_slots_10000_cost_{cost}_proposers_{num_proposers}{suffix}"
        )

        extra_args = [
            f"--cost {cost}",
            f"--num_proposers_per_slot {num_proposers}",
            f"--proposer_mode MCP",
        ]
        if hetero_info:
            extra_args.append("--info-distribution heterogeneous")

        cmd = _build_cmd("MSP", root, config_path, outdir, extra_args)

        _tmux(c, f"select-window -t {window}.0")
        _tmux(c, f"send-keys -t {window}.0 {_quoted(cmd)} Enter")

@task
def run_multiple_proposers_few_signals(c, session="multiple-proposers-few-signals"):
    """
    Run simulations with multiple proposers per slot and 1 or 2 signals per continent in a tmux session.
    
    Usage:
        fab run-multiple-proposers-few-signals             # default session name is 'multiple-proposers-few-signals'
        fab run-multiple-proposers-few-signals:session=foo # custom session name
    """
    # Resolve parent directory (equivalent to cd "$SCRIPT_DIR/..")
    script_dir = Path(__file__).resolve().parent
    root = script_dir.parent

    # Make sure the tmux session exists
    _ensure_session(c, session)
    
    num_jobs = 0

    cost = 0
    
    for num_proposers in [1, 2, 4, 8]:
        _tmux(c, f"new-window -t {session} -n proposers_{num_proposers}")
        window = f"{session}:proposers_{num_proposers}"
        config_path = f"params/MCP-few_signals.yaml"
        outdir = f"output/multiple_proposers_few_signals/validators_1000_slots_10000_cost_{cost}_proposers_{num_proposers}"
        extra_args = [
            f"--cost {cost}",
            f"--num_proposers_per_slot {num_proposers}",
            "--info-distribution heterogeneous",
            "--proposer_mode MCP",
        ]

        cmd = _build_cmd("MSP", root, config_path, outdir, extra_args)

        _tmux(c, f"select-window -t {window}.0")
        _tmux(c, f"send-keys -t {window}.0 {_quoted(cmd)} Enter")
        num_jobs += 1

@task
def attach(c, session="simulation-baseline"):
    """Attach to the tmux session to view logs interactively."""
    c.run(f"tmux attach -t {shlex.quote(session)}", pty=True)


@task
def kill(c, session="simulation-baseline"):
    """Kill the tmux session to stop all jobs."""
    if _has_session(c, session):
        _tmux(c, f"kill-session -t {shlex.quote(session)}")
        print(f"[ok] Killed tmux session '{session}'.")
    else:
        print(f"[info] No tmux session named '{session}' was found.")
