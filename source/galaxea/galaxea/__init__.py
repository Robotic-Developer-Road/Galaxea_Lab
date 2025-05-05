"""
Python module serving as a project/extension template.
"""

# Register Gym environments.
from .tasks import *
from .assets import *

# Register UI extensions.
from .ui_extension_example import *

from .tasks.direct import *
from .tasks.manager_based import *
