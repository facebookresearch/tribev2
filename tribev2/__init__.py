# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

try:
    from tribev2.demo_utils import TribeModel
except ImportError:
    # Optional demo dependencies may be unavailable
    TribeModel = None

__all__ = ["TribeModel"]
