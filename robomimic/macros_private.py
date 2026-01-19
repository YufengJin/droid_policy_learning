"""
Set of global variables shared across robomimic
"""
# Sets debugging mode. Should be set at top-level script so that internal
# debugging functionalities are made active
DEBUG = False

# Whether to visualize the before & after of an observation randomizer
VISUALIZE_RANDOMIZER = False

# wandb entity (eg. username or team name)
WANDB_ENTITY = 'yufeng-jin'

# wandb api key (obtain from https://wandb.ai/authorize)
# alternatively, set up wandb from terminal with `wandb login`
WANDB_API_KEY = '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'

try:
    from robomimic.macros_private import *
except ImportError:
    from robomimic.utils.log_utils import log_warning
    import robomimic
    log_warning(
        "No private macro file found!"\
        "\nIt is recommended to use a private macro file"\
        "\nTo setup, run: python {}/scripts/setup_macros.py".format(robomimic.__path__[0])
    )
