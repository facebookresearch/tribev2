# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Safer YAML loading for TRIBE config files.

The pretrained-model loader used to call ``yaml.load`` with
``yaml.UnsafeLoader``, which constructs arbitrary Python objects from
YAML tags. That allows ``!!python/object/new:os.system [...]``-style
payloads to execute code at load time.

This module exposes ``load_config`` which uses a ``yaml.SafeLoader``
subclass that explicitly opts in to ``!!python/tuple``, the only
Python-specific tag known to appear in published TRIBE config files.
All other Python-specific tags (``!!python/object``,
``!!python/object/new``, ``!!python/name``, ``!!python/module``, …) are
rejected by the safe base class.
"""

from pathlib import Path

import yaml
from exca import ConfDict


class TribeConfigLoader(yaml.SafeLoader):
    """Safe YAML loader for TRIBE configs with explicit tuple support."""


def _construct_python_tuple(loader: yaml.Loader, node: yaml.Node) -> tuple:
    return tuple(loader.construct_sequence(node))


TribeConfigLoader.add_constructor(
    "tag:yaml.org,2002:python/tuple",
    _construct_python_tuple,
)


def load_config(path: str | Path) -> ConfDict:
    """Load a TRIBE config YAML safely and return a ConfDict."""
    with open(path, "r", encoding="utf-8") as f:
        return ConfDict(yaml.load(f, Loader=TribeConfigLoader))
