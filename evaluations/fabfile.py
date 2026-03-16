# fabfile.py
from fabric import task
from pathlib import Path

import shlex
import sys
import time

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


def _normalize_task_name(task_name: str) -> str:
    """Accept either Fabric CLI names or Python function-style task names."""
    return task_name.strip().replace("-", "_")


def _parse_csv(value) -> list[str]:
    """Parse a comma-separated CLI argument into a clean list."""
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        items = value
    else:
        items = str(value).split(",")
    return [item.strip() for item in items if item and item.strip()]


def _parse_latency_std_dev_ratios(value) -> list:
    """Parse one or many latency std-dev ratios from CLI input."""
    if value is None:
        return [None]

    ratios = []
    for item in _parse_csv(value):
        ratio = float(item)
        if ratio < 0:
            raise ValueError("latency_std_dev_ratio must be non-negative")
        ratios.append(ratio)

    return ratios or [None]


def _optional_cli_args(seed=None, latency_std_dev_ratio=None) -> list:
    """Build optional CLI arguments shared across evaluation tasks."""
    args = []
    if seed is not None:
        args.append(f"--seed {seed}")
    if latency_std_dev_ratio is not None:
        args.append(f"--latency-std-dev-ratio {latency_std_dev_ratio}")
    return args


def _format_latency_std_dev_ratio(latency_std_dev_ratio) -> str:
    """Create a stable string representation for latency std-dev ratio suffixes."""
    return format(float(latency_std_dev_ratio), "g")


def _with_run_suffix(outdir: str, seed=None, latency_std_dev_ratio=None) -> str:
    """Append optional run parameters to output directory names."""
    if seed is not None:
        outdir = f"{outdir}_seed_{seed}"
    if latency_std_dev_ratio is not None:
        outdir = f"{outdir}_latstd_{_format_latency_std_dev_ratio(latency_std_dev_ratio)}"
    return outdir


def _with_seed_session(session: str, seed=None) -> str:
    """Append seed to tmux session names when provided."""
    if seed is None:
        return session
    return f"{session}-seed-{seed}"


def _with_ratio_session(session: str, latency_std_dev_ratio=None) -> str:
    """Append latency std-dev ratio to tmux session names when provided."""
    if latency_std_dev_ratio is None:
        return session
    return f"{session}-latstd-{_format_latency_std_dev_ratio(latency_std_dev_ratio)}"


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


def _session_pane_commands(c, session: str) -> list[str]:
    """Return the current command running in every pane of a tmux session."""
    if not _has_session(c, session):
        return []
    result = _tmux(c, f"list-panes -s -t {shlex.quote(session)} -F '#{{pane_current_command}}'")
    if not result.ok or not result.stdout.strip():
        return []
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def _is_session_busy(c, session: str) -> bool:
    """Treat a tmux session as busy while any pane is still running a non-shell command."""
    shell_commands = {"bash", "sh", "zsh", "fish", "login"}
    commands = _session_pane_commands(c, session)
    if not commands:
        return False
    return any(command.lstrip("-") not in shell_commands for command in commands)


def _pane_current_command(c, target: str) -> str | None:
    """Return the current command for a single tmux pane target."""
    result = _tmux(
        c,
        f"display-message -p -t {shlex.quote(target)} '#{{pane_current_command}}'",
    )
    if not result.ok:
        return None
    return result.stdout.strip() or None


def _is_pane_busy(c, target: str) -> bool:
    """Check whether a tmux pane is still executing a non-shell command."""
    command = _pane_current_command(c, target)
    if command is None:
        return False
    return command.lstrip("-") not in {"bash", "sh", "zsh", "fish", "login"}


def _run_jobs_with_worker_windows(c, session: str, jobs: list[dict], max_parallel: int, poll_interval: int = 30) -> None:
    """Run queued simulation commands with a bounded number of tmux worker windows."""
    if max_parallel < 1:
        raise ValueError("--max-parallel must be at least 1")
    if poll_interval < 1:
        raise ValueError("--poll-interval must be at least 1 second")
    if not jobs:
        print(f"[info] No jobs to run in tmux session '{session}'.")
        return

    _ensure_session(c, session)
    worker_prefix = f"worker-{int(time.time())}"
    workers = []

    for index in range(min(max_parallel, len(jobs))):
        window_name = f"{worker_prefix}-{index + 1}"
        _tmux(c, f"new-window -t {shlex.quote(session)} -n {shlex.quote(window_name)}")
        workers.append(
            {
                "target": f"{session}:{window_name}.0",
                "job": None,
            }
        )

    queue_index = 0
    active_jobs = 0
    completed_jobs = 0

    while queue_index < len(jobs) or active_jobs > 0:
        for worker in workers:
            if worker["job"] is not None and not _is_pane_busy(c, worker["target"]):
                finished_job = worker["job"]
                print(f"[done] {finished_job['label']}")
                worker["job"] = None
                active_jobs -= 1
                completed_jobs += 1

            if worker["job"] is None and queue_index < len(jobs):
                next_job = jobs[queue_index]
                _tmux(c, f"send-keys -t {shlex.quote(worker['target'])} {_quoted(next_job['cmd'])} Enter")
                worker["job"] = next_job
                queue_index += 1
                active_jobs += 1
                print(f"[start] {next_job['label']}")

        if queue_index < len(jobs) or active_jobs > 0:
            time.sleep(poll_interval)

    print(f"[ok] Finished {completed_jobs} jobs in tmux session '{session}'.")


def _task_registry():
    """Registry for evaluation tasks that can be orchestrated in batches."""
    return {
        "run_baseline": run_baseline,
        "run_heterogeneous_information_sources": run_heterogeneous_information_sources,
        "run_heterogeneous_validators": run_heterogeneous_validators,
        "run_hetero_both": run_hetero_both,
        "run_different_gammas": run_different_gammas,
        "run_eip7782": run_eip7782,
    }


@task
def run_baseline(c, session="simulation-baseline", seed=None, latency_std_dev_ratio=None, max_parallel=None, poll_interval=30):
    """
    Run four baseline evaluations in parallel inside tmux panes.
    
    Usage:
        fab run-baseline                  # default session name is 'simulation-baseline'
        fab run-baseline --session=foo   # custom session name
        fab run-baseline --seed=12345    # run this batch with a fixed seed
        fab run-baseline --latency-std-dev-ratio=0.25
        fab run-baseline --latency-std-dev-ratio=0.25,0.5 --max-parallel=4
    """
    # Resolve parent directory (equivalent to cd "$SCRIPT_DIR/..")
    script_dir = Path(__file__).resolve().parent
    root = script_dir.parent
    session = _with_seed_session(session, seed)
    latency_std_dev_ratios = _parse_latency_std_dev_ratios(latency_std_dev_ratio)

    jobs = []

    # Create a new window within the session
    for model in ["SSP", "MSP"]:
        config_path = f"params/{model}-baseline.yaml"
        outdir = f"output/baseline/{model}/validators_1000_slots_10000"

        for ratio in latency_std_dev_ratios:
            for cost in COSTS:
                jobs.append(
                    {
                        "label": f"{model} cost={cost} seed={seed} latstd={ratio}",
                        "cmd": _build_cmd(
                            model,
                            root,
                            config_path,
                            _with_run_suffix(f"{outdir}_cost_{cost}", seed, ratio),
                            [f"--cost {cost}"] + _optional_cli_args(seed, ratio),
                        ),
                    }
                )

    if max_parallel is not None:
        _run_jobs_with_worker_windows(c, session, jobs, max_parallel, poll_interval)
        print(f"Attach with: tmux attach -t {session}")
        return

    _ensure_session(c, session)
    num_jobs = 0

    for model in ["SSP", "MSP"]:
        _tmux(c, f"new-window -t {session} -n {model}")
        window = f"{session}:{model}"
        config_path = f"params/{model}-baseline.yaml"
        outdir = f"output/baseline/{model}/validators_1000_slots_10000"

        cmds = [
            _build_cmd(
                model,
                root,
                config_path,
                _with_run_suffix(f"{outdir}_cost_{cost}", seed, ratio),
                [f"--cost {cost}"] + _optional_cli_args(seed, ratio),
            )
            for ratio in latency_std_dev_ratios
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
def run_heterogeneous_information_sources(c, session="hetero-info", seed=None, latency_std_dev_ratio=None, max_parallel=None, poll_interval=30):
    """
    Run a simulation with heterogeneous information sources in a tmux session.
    
    Usage:
        fab run-heterogeneous-information-sources             # default session name is 'hetero-info'
        fab run-heterogeneous-information-sources --session=foo # custom session name
        fab run-heterogeneous-information-sources --seed=12345  # run this batch with a fixed seed
        fab run-heterogeneous-information-sources --latency-std-dev-ratio=0.25
        fab run-heterogeneous-information-sources --latency-std-dev-ratio=0.25,0.5 --max-parallel=4
    """
    # Resolve parent directory (equivalent to cd "$SCRIPT_DIR/..")
    script_dir = Path(__file__).resolve().parent
    root = script_dir.parent
    session = _with_seed_session(session, seed)
    latency_std_dev_ratios = _parse_latency_std_dev_ratios(latency_std_dev_ratio)

    cost = 0.002
    jobs = []

    for model in ["SSP", "MSP"]:
        for ratio in latency_std_dev_ratios:
            for latency_mode in ["latency-aligned", "latency-misaligned"]:
                config_path = f"params/{model}-{latency_mode}.yaml"
                jobs.append(
                    {
                        "label": f"{model} latency={latency_mode} seed={seed} latstd={ratio}",
                        "cmd": _build_cmd(
                            model,
                            root,
                            config_path,
                            _with_run_suffix(
                                f"output/hetero_info/{model}/validators_1000_slots_10000_cost_{cost}_latency_{latency_mode}",
                                seed,
                                ratio,
                            ),
                            [f"--cost {cost}", "--info-distribution heterogeneous"] + _optional_cli_args(seed, ratio),
                        ),
                    }
                )

    if max_parallel is not None:
        _run_jobs_with_worker_windows(c, session, jobs, max_parallel, poll_interval)
        print(f"Attach with: tmux attach -t {session}")
        return

    _ensure_session(c, session)
    num_jobs = 0

    for model in ["SSP", "MSP"]:
        _tmux(c, f"new-window -t {session} -n {model}")
        window = f"{session}:{model}"
        pane_index = 0

        for ratio in latency_std_dev_ratios:
            for latency_mode in ["latency-aligned", "latency-misaligned"]:
                config_path = f"params/{model}-{latency_mode}.yaml"
                outdir = _with_run_suffix(
                    f"output/hetero_info/{model}/validators_1000_slots_10000_cost_{cost}_latency_{latency_mode}",
                    seed,
                    ratio,
                )

                cmd = _build_cmd(
                    model,
                    root,
                    config_path,
                    outdir,
                    [f"--cost {cost}", "--info-distribution heterogeneous"] + _optional_cli_args(seed, ratio),
                )

                if pane_index > 0:
                    _tmux(c, f"split-window -t {window} -v")
                
                _tmux(c, f"select-pane -t {window}.{pane_index}")
                _tmux(c, f"send-keys -t {window}.{pane_index} {_quoted(cmd)} Enter")
                pane_index += 1
                num_jobs += 1

        _tmux(c, f"select-layout -t {window} tiled")

    # Print instructions for the user
    print(f"[ok] Started {num_jobs} jobs in tmux session '{session}'.")
    print(f"Attach with: tmux attach -t {session}")


@task
def run_heterogeneous_validators(c, session="hetero-validators", seed=None, latency_std_dev_ratio=None, max_parallel=None, poll_interval=30):
    """
    Run a simulation with heterogeneous validators in a tmux session.
    
    Usage:
        fab run-heterogeneous-validators             # default session name is 'hetero-validators'
        fab run-heterogeneous-validators --session=foo # custom session name
        fab run-heterogeneous-validators --seed=12345  # run this batch with a fixed seed
        fab run-heterogeneous-validators --latency-std-dev-ratio=0.25
        fab run-heterogeneous-validators --latency-std-dev-ratio=0.25,0.5 --max-parallel=4
    """
    # Resolve parent directory (equivalent to cd "$SCRIPT_DIR/..")
    script_dir = Path(__file__).resolve().parent
    root = script_dir.parent
    session = _with_seed_session(session, seed)
    latency_std_dev_ratios = _parse_latency_std_dev_ratios(latency_std_dev_ratio)

    jobs = []

    for model in ["SSP", "MSP"]:
        config_path = f"params/{model}-baseline.yaml"

        for ratio in latency_std_dev_ratios:
            for cost in [0.0, 0.002]:
                jobs.append(
                    {
                        "label": f"{model} cost={cost} validators=heterogeneous seed={seed} latstd={ratio}",
                        "cmd": _build_cmd(
                            model,
                            root,
                            config_path,
                            _with_run_suffix(
                                f"output/hetero_validators/{model}/slots_10000_cost_{cost}_validators_heterogeneous",
                                seed,
                                ratio,
                            ),
                            [f"--cost {cost}", "--distribution heterogeneous"] + _optional_cli_args(seed, ratio),
                        ),
                    }
                )

    if max_parallel is not None:
        _run_jobs_with_worker_windows(c, session, jobs, max_parallel, poll_interval)
        print(f"Attach with: tmux attach -t {session}")
        return

    _ensure_session(c, session)
    num_jobs = 0

    for model in ["SSP", "MSP"]:
        _tmux(c, f"new-window -t {session} -n {model}")
        window = f"{session}:{model}"
        config_path = f"params/{model}-baseline.yaml"
        pane_index = 0

        for ratio in latency_std_dev_ratios:
            for cost in [0.0, 0.002]:
                outdir = _with_run_suffix(
                    f"output/hetero_validators/{model}/slots_10000_cost_{cost}_validators_heterogeneous",
                    seed,
                    ratio,
                )
                cmd = _build_cmd(
                    model,
                    root,
                    config_path,
                    outdir,
                    [f"--cost {cost}", "--distribution heterogeneous"] + _optional_cli_args(seed, ratio),
                )

                if pane_index > 0:
                    _tmux(c, f"split-window -t {window} -v")

                _tmux(c, f"select-window -t {window}.{pane_index}")
                _tmux(c, f"send-keys -t {window}.{pane_index} {_quoted(cmd)} Enter")
                pane_index += 1
                num_jobs += 1
        
        _tmux(c, f"select-layout -t {window} tiled")

    # Print instructions for the user
    print(f"[ok] Started {num_jobs} jobs in tmux session '{session}'.")
    print(f"Attach with: tmux attach -t {session}")


@task
def run_hetero_both(c, session="hetero-both", seed=None, latency_std_dev_ratio=None, max_parallel=None, poll_interval=30):
    """
    Run a simulation with both heterogeneous validators and information sources in a tmux session.
    
    Usage:
        fab run-hetero-both             # default session name is 'hetero-both'
        fab run-hetero-both --session=foo # custom session name
        fab run-hetero-both --seed=12345  # run this batch with a fixed seed
        fab run-hetero-both --latency-std-dev-ratio=0.25
        fab run-hetero-both --latency-std-dev-ratio=0.25,0.5 --max-parallel=4
    """
    # Resolve parent directory (equivalent to cd "$SCRIPT_DIR/..")
    script_dir = Path(__file__).resolve().parent
    root = script_dir.parent
    session = _with_seed_session(session, seed)
    latency_std_dev_ratios = _parse_latency_std_dev_ratios(latency_std_dev_ratio)

    cost = 0.002
    jobs = []
    
    for model in ["SSP", "MSP"]:
        for ratio in latency_std_dev_ratios:
            for latency_mode in ["latency-aligned", "latency-misaligned"]:
                config_path = f"params/{model}-{latency_mode}.yaml"
                jobs.append(
                    {
                        "label": f"{model} both latency={latency_mode} seed={seed} latstd={ratio}",
                        "cmd": _build_cmd(
                            model,
                            root,
                            config_path,
                            _with_run_suffix(
                                f"output/hetero_both/{model}/validators_heterogeneous_slots_10000_cost_{cost}_latency_{latency_mode}",
                                seed,
                                ratio,
                            ),
                            [f"--cost {cost}", "--distribution heterogeneous", "--info-distribution heterogeneous"] + _optional_cli_args(seed, ratio),
                        ),
                    }
                )

    if max_parallel is not None:
        _run_jobs_with_worker_windows(c, session, jobs, max_parallel, poll_interval)
        print(f"Attach with: tmux attach -t {session}")
        return

    _ensure_session(c, session)
    num_jobs = 0

    for model in ["SSP", "MSP"]:
        _tmux(c, f"new-window -t {session} -n {model}")
        window = f"{session}:{model}"
        pane_index = 0

        for ratio in latency_std_dev_ratios:
            for latency_mode in ["latency-aligned", "latency-misaligned"]:
                config_path = f"params/{model}-{latency_mode}.yaml"
                outdir = _with_run_suffix(
                    f"output/hetero_both/{model}/validators_heterogeneous_slots_10000_cost_{cost}_latency_{latency_mode}",
                    seed,
                    ratio,
                )
                cmd = _build_cmd(
                    model,
                    root,
                    config_path,
                    outdir,
                    [f"--cost {cost}", "--distribution heterogeneous", "--info-distribution heterogeneous"] + _optional_cli_args(seed, ratio),
                )

                if pane_index > 0:
                    _tmux(c, f"split-window -t {window} -v")
                
                _tmux(c, f"select-pane -t {window}.{pane_index}")
                _tmux(c, f"send-keys -t {window}.{pane_index} {_quoted(cmd)} Enter")
                pane_index += 1
                num_jobs += 1

        _tmux(c, f"select-layout -t {window} tiled")

    # Print instructions for the user
    print(f"[ok] Started {num_jobs} jobs in tmux session '{session}'.")
    print(f"Attach with: tmux attach -t {session}")
    

@task
def run_different_gammas(c, session="different-gammas", seed=None, latency_std_dev_ratio=None, max_parallel=None, poll_interval=30):
    """
    Run simulations with different gamma values in a tmux session.
    
    Usage:
        fab run-different-gammas             # default session name is 'different-gammas'
        fab run-different-gammas --session=foo # custom session name
        fab run-different-gammas --seed=12345  # run this batch with a fixed seed
        fab run-different-gammas --latency-std-dev-ratio=0.25
        fab run-different-gammas --latency-std-dev-ratio=0.25,0.5 --max-parallel=4
    """
    # Resolve parent directory (equivalent to cd "$SCRIPT_DIR/..")
    script_dir = Path(__file__).resolve().parent
    root = script_dir.parent
    session = _with_seed_session(session, seed)
    latency_std_dev_ratios = _parse_latency_std_dev_ratios(latency_std_dev_ratio)

    cost = 0.002
    jobs = []
    
    for model in ["SSP", "MSP"]:
        config_path = f"params/{model}-baseline.yaml"

        for ratio in latency_std_dev_ratios:
            for gamma in [0.3333, 0.5, 0.6667, 0.8]:
                jobs.append(
                    {
                        "label": f"{model} gamma={gamma} seed={seed} latstd={ratio}",
                        "cmd": _build_cmd(
                            model,
                            root,
                            config_path,
                            _with_run_suffix(
                                f"output/different_gammas/{model}/validators_1000_slots_10000_cost_{cost}_gamma_{gamma}",
                                seed,
                                ratio,
                            ),
                            [f"--cost {cost}", f"--gamma {gamma}"] + _optional_cli_args(seed, ratio),
                        ),
                    }
                )

    if max_parallel is not None:
        _run_jobs_with_worker_windows(c, session, jobs, max_parallel, poll_interval)
        print(f"Attach with: tmux attach -t {session}")
        return

    _ensure_session(c, session)
    num_jobs = 0

    for model in ["SSP", "MSP"]:
        _tmux(c, f"new-window -t {session} -n {model}")
        window = f"{session}:{model}"
        config_path = f"params/{model}-baseline.yaml"
        pane_index = 0

        for ratio in latency_std_dev_ratios:
            for gamma in [0.3333, 0.5, 0.6667, 0.8]:
                outdir = _with_run_suffix(
                    f"output/different_gammas/{model}/validators_1000_slots_10000_cost_{cost}_gamma_{gamma}",
                    seed,
                    ratio,
                )

                cmd = _build_cmd(
                    model,
                    root,
                    config_path,
                    outdir,
                    [f"--cost {cost}", f"--gamma {gamma}"] + _optional_cli_args(seed, ratio),
                )

                if pane_index > 0:
                    _tmux(c, f"split-window -t {window} -v")
                    
                _tmux(c, f"select-window -t {window}.{pane_index}")
                _tmux(c, f"send-keys -t {window}.{pane_index} {_quoted(cmd)} Enter")
                pane_index += 1
                num_jobs += 1

    # Print instructions for the user
    print(f"[ok] Started {num_jobs} jobs in tmux session '{session}'.")
    print(f"Attach with: tmux attach -t {session}")


@task
def run_eip7782(c, session="eip7782", seed=None, latency_std_dev_ratio=None, max_parallel=None, poll_interval=30):
    """
    Run simulations with EIP-7782 enabled in a tmux session.
    
    Usage:
        fab run-eip7782             # default session name is 'eip7782'
        fab run-eip7782 --session=foo # custom session name
        fab run-eip7782 --seed=12345  # run this batch with a fixed seed
        fab run-eip7782 --latency-std-dev-ratio=0.25
        fab run-eip7782 --latency-std-dev-ratio=0.25,0.5 --max-parallel=4
    """
    # Resolve parent directory (equivalent to cd "$SCRIPT_DIR/..")
    script_dir = Path(__file__).resolve().parent
    root = script_dir.parent
    session = _with_seed_session(session, seed)
    latency_std_dev_ratios = _parse_latency_std_dev_ratios(latency_std_dev_ratio)

    cost = 0.002
    jobs = []
    
    for model in ["SSP", "MSP"]:
        config_path = f"params/{model}-baseline.yaml"
        delta = 6000
        cutoff = 3000

        for ratio in latency_std_dev_ratios:
            jobs.append(
                {
                    "label": f"{model} delta={delta} cutoff={cutoff} seed={seed} latstd={ratio}",
                    "cmd": _build_cmd(
                        model,
                        root,
                        config_path,
                        _with_run_suffix(
                            f"output/eip7782/{model}/validators_1000_slots_10000_cost_{cost}_delta_{delta}_cutoff_{cutoff}",
                            seed,
                            ratio,
                        ),
                        [f"--cost {cost}", f"--delta {delta}", f"--cutoff {cutoff}"] + _optional_cli_args(seed, ratio),
                    ),
                }
            )

    if max_parallel is not None:
        _run_jobs_with_worker_windows(c, session, jobs, max_parallel, poll_interval)
        print(f"Attach with: tmux attach -t {session}")
        return

    _ensure_session(c, session)
    num_jobs = 0

    for model in ["SSP", "MSP"]:
        _tmux(c, f"new-window -t {session} -n {model}")
        window = f"{session}:{model}"
        config_path = f"params/{model}-baseline.yaml"
        delta = 6000
        cutoff = 3000
        pane_index = 0

        for ratio in latency_std_dev_ratios:
            outdir = _with_run_suffix(
                f"output/eip7782/{model}/validators_1000_slots_10000_cost_{cost}_delta_{delta}_cutoff_{cutoff}",
                seed,
                ratio,
            )

            cmd = _build_cmd(
                model,
                root,
                config_path,
                outdir,
                [f"--cost {cost}", f"--delta {delta}", f"--cutoff {cutoff}"] + _optional_cli_args(seed, ratio),
            )

            if pane_index > 0:
                _tmux(c, f"split-window -t {window} -v")

            _tmux(c, f"select-window -t {window}.{pane_index}")
            _tmux(c, f"send-keys -t {window}.{pane_index} {_quoted(cmd)} Enter")
            pane_index += 1
            num_jobs += 1

    # Print instructions for the user
    print(f"[ok] Started {num_jobs} jobs in tmux session '{session}'.")
    print(f"Attach with: tmux attach -t {session}")


@task
def run_seed_queue(
    c,
    seeds,
    tasks=None,
    max_parallel=1,
    poll_interval=30,
    session_prefix="batch",
    kill_when_done=False,
    latency_std_dev_ratio=None,
    latency_std_dev_ratios=None,
):
    """
    Run multiple evaluation batches with bounded concurrency.

    Each queued job is one existing `run-*` Fabric task with a specific seed.
    The scheduler waits until all tmux panes inside that task have finished
    before starting another queued job.

    Usage:
        fab run-seed-queue --seeds=1,2,3 --max-parallel=2
        fab run-seed-queue --seeds=11,22 --tasks=run-baseline,run-hetero-both
        fab run-seed-queue --seeds=11,22 --latency-std-dev-ratios=0.25,0.5
    """
    parsed_seeds = _parse_csv(seeds)
    registry = _task_registry()
    ratio_input = latency_std_dev_ratios if latency_std_dev_ratios is not None else latency_std_dev_ratio
    parsed_latency_std_dev_ratios = _parse_latency_std_dev_ratios(ratio_input)
    parsed_tasks = (
        [_normalize_task_name(task_name) for task_name in _parse_csv(tasks)]
        if tasks is not None
        else list(registry.keys())
    )

    if not parsed_seeds:
        raise ValueError("Please provide at least one seed via --seeds=1,2,3")
    if tasks is not None and not parsed_tasks:
        raise ValueError("Please provide at least one task via --tasks=run-baseline")
    if max_parallel < 1:
        raise ValueError("--max-parallel must be at least 1")
    if poll_interval < 1:
        raise ValueError("--poll-interval must be at least 1 second")

    unknown_tasks = [task_name for task_name in parsed_tasks if task_name not in registry]
    if unknown_tasks:
        available = ", ".join(sorted(name.replace("_", "-") for name in registry))
        raise ValueError(
            "Unknown task(s): "
            + ", ".join(task_name.replace("_", "-") for task_name in unknown_tasks)
            + f". Available tasks: {available}"
        )

    queue = []
    for task_name in parsed_tasks:
        for ratio in parsed_latency_std_dev_ratios:
            for seed in parsed_seeds:
                base_session = _with_ratio_session(
                    f"{session_prefix}-{task_name.replace('_', '-')}",
                    ratio,
                )
                queue.append(
                    {
                        "task_name": task_name,
                        "seed": seed,
                        "latency_std_dev_ratio": ratio,
                        "base_session": base_session,
                        "session": _with_seed_session(base_session, seed),
                        "runner": registry[task_name],
                    }
                )

    active_jobs = []
    queue_index = 0

    while queue_index < len(queue) or active_jobs:
        while queue_index < len(queue) and len(active_jobs) < max_parallel:
            job = queue[queue_index]
            print(
                f"[start] {job['task_name'].replace('_', '-')} seed={job['seed']} "
                f"latstd={job['latency_std_dev_ratio']} "
                f"session={job['session']}"
            )
            job["runner"](
                c,
                session=job["base_session"],
                seed=job["seed"],
                latency_std_dev_ratio=job["latency_std_dev_ratio"],
            )
            active_jobs.append(job)
            queue_index += 1

        time.sleep(poll_interval)

        remaining_jobs = []
        for job in active_jobs:
            if _is_session_busy(c, job["session"]):
                remaining_jobs.append(job)
                continue

            print(
                f"[done] {job['task_name'].replace('_', '-')} seed={job['seed']} "
                f"latstd={job['latency_std_dev_ratio']} "
                f"session={job['session']}"
            )
            if kill_when_done and _has_session(c, job["session"]):
                _tmux(c, f"kill-session -t {shlex.quote(job['session'])}")
                print(f"[cleanup] killed tmux session '{job['session']}'")

        active_jobs = remaining_jobs

    print("[ok] All queued jobs have finished.")


@task
def attach(c, session="simulation-baseline", seed=None):
    """Attach to the tmux session to view logs interactively."""
    session = _with_seed_session(session, seed)
    c.run(f"tmux attach -t {shlex.quote(session)}", pty=True)


@task
def kill(c, session="simulation-baseline", seed=None):
    """Kill the tmux session to stop all jobs."""
    session = _with_seed_session(session, seed)
    if _has_session(c, session):
        _tmux(c, f"kill-session -t {shlex.quote(session)}")
        print(f"[ok] Killed tmux session '{session}'.")
    else:
        print(f"[info] No tmux session named '{session}' was found.")
