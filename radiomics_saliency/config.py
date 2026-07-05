"""Shared mutable runtime state for the experiment pipeline.

``device`` and ``image_max_width`` are set once by the CLI entrypoint and
read from other modules at call time, so import this module (``from
radiomics_saliency import config``) rather than importing the attributes
directly.
"""

device = None
image_max_width = 64
