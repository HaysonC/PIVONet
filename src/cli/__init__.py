"""Conversational CLI interface using Rich and Questionary.

===================================================================================
OVERVIEW
===================================================================================
Implements an interactive command-line interface for PIVONet with:
  - Rich text formatting and progress bars
  - Questionary prompts for user input
  - Stateful workflow (remembers previous choices)
  - Command history and context awareness
  - Error recovery with helpful hints

Entry point: pivo

===================================================================================
SUBMODULE STRUCTURE
===================================================================================

main.py:
    _cli(argv) → Entry point
    Dispatches to subcommands based on user selection
    Handles --list-experiments, --run-experiment <slug>, etc.

commands.py:
    InteractionChannel - Protocol for CLI communication
    simulate() → Trajectory simulation workflow
    train() → Model training workflow
    visualize() → Result visualization workflow
    import_trajectories() → Import velocity snapshots
    Each returns InteractionChannel for error/warning messaging

chat.py:
    FlowChat - Stateful chat interface
    Attributes:
      - console: Rich Console for formatting
      - _last_paths: dict remembering user choices
    Methods:
      - greet() → Welcome message
      - say() → Print formatted message
      - success() → Print success (green)
      - hint() → Show LaunchOptions hint
      - wrap_error() → Format and display errors

===================================================================================
WORKFLOW DIAGRAM
===================================================================================

    pivo (console script entry point)
            │
            ▼
    main._cli(argv)
            │
    ┌───────┼───────────────────────┐
    │       │                       │
    ▼       ▼                       ▼
  --list  --run-exp  Interactive Menu
  exps    <slug>     (questionary)
    │       │             │
    │       │         ┌───┴───────┬────────┬─────┐
    │       │         │           │        │     │
    │       │         ▼           ▼        ▼     ▼
    │       │       import      simulate  train visualize
    │       │       velocities  particles models results
    │       │         │           │        │     │
    │       └─────────┴───────────┴────────┴─────┘
    │                           │
    ▼                           ▼
  List or Print             Orchestration
  YAML specs                (ExperimentOrchestrator.run())
                                  │
                                  ▼
                            Execute steps,
                            track progress,
                            save artifacts

===================================================================================
KEY TYPES
===================================================================================

FlowChat:
    Attributes:
      - console: Rich.Console
      - _last_paths: dict[str, Path]
    Methods:
      - greet()
      - say(message, style)
      - success(message)
      - hint(option: LaunchOptions)
      - wrap_error(error, option)

InteractionChannel (Protocol):
    Methods:
      - info(message: str)
      - warn(message: str)
      - error(message: str)
      - progress(item, total, description)

LaunchOptions:
    velocity_dir: Path
    particles: int
    max_steps: int
    dt: float
    model_type: str ('cnf', 'vsde')
    device: str ('auto', 'cuda', 'mps', 'cpu')

===================================================================================
USAGE EXAMPLES
===================================================================================

# Interactive mode
$ pivo

# List available experiments
$ pivo --list-experiments

# Run specific experiment
$ pivo --run-experiment demo-baseline

# Run with parameter overrides
$ pivo --run-experiment demo-baseline --overrides particles=200 dt=0.05

# Check version
$ pivo --version

===================================================================================
CONSTRAINTS
===================================================================================

1. Questionary prompts require TTY (no piping stdin)
2. Rich formatting requires color-capable terminal (auto-detection)
3. File uploads in CLI limited by terminal window size
4. Experiment YAML must be valid; will error with helpful message
5. User paths are cached per session; cleared on exit

===================================================================================
"""
