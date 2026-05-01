"""
Import all model modules here so their @register_model decorators fire
when `src.models` is imported.  Add new model modules to this list.
"""

from src.models import factory  # noqa: F401 – must come first
from src.models import unet_resnet50  # noqa: F401
from src.models import vit_segmentation  # noqa: F401
from src.models import prithvi_adapter  # noqa: F401
from src.models import clay_adapter     # noqa: F401

from src.models.factory import build_model, list_models  # noqa: F401
