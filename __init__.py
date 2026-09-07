import os

import folder_paths

# models/text_cond/ holds the frozen text-conditioning safetensors (fixed_embed_fwd_anyframe)
folder_paths.add_model_folder_path("text_cond", os.path.join(folder_paths.models_dir, "text_cond"))

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
from .loop_nodes import NODE_CLASS_MAPPINGS as LOOP_NODES, NODE_DISPLAY_NAME_MAPPINGS as LOOP_NAMES

NODE_CLASS_MAPPINGS = {**NODE_CLASS_MAPPINGS, **LOOP_NODES}
NODE_DISPLAY_NAME_MAPPINGS = {**NODE_DISPLAY_NAME_MAPPINGS, **LOOP_NAMES}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
