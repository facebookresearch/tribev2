# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

__all__ = ["TribeModel"]


def __getattr__(name):
    if name == "TribeModel":
        from tribev2.demo_utils import TribeModel

        return TribeModel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
