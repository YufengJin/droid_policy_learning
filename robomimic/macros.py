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

# wandb api key：勿提交真实 key。优先 `wandb login`；或在 macros_private.py 中设置。
WANDB_API_KEY = None

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
