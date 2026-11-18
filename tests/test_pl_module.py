# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import pytest

pl_module = pytest.importorskip("tribev2.pl_module")


class GroupedMetric:
    """Stand-in for a grouped metric: compute() returns {group_id: value}."""

    def __init__(self, values):
        self.values = values
        self.was_reset = False

    def compute(self):
        return self.values

    def reset(self):
        self.was_reset = True


def _run_epoch_end(mapping, values):
    module = pl_module.BrainModule.__new__(pl_module.BrainModule)
    logged = {}
    object.__setattr__(
        module, "config", {"data.subject_id.predefined_mapping": mapping}
    )
    object.__setattr__(
        module, "metrics", {"test/pearson_grouped": GroupedMetric(values)}
    )
    object.__setattr__(module, "log_dict", logged.update)
    module.on_val_or_test_epoch_end("test")
    return logged, module.metrics["test/pearson_grouped"]


def test_grouped_metrics_use_subject_names():
    logged, metric = _run_epoch_end({"sub-01": 0, "sub-02": 1}, {0: 0.1, 1: 0.2})
    assert logged == {
        "test/pearson_grouped/sub-01": 0.1,
        "test/pearson_grouped/sub-02": 0.2,
    }
    assert metric.was_reset


def test_grouped_metrics_without_mapping_in_average_subjects_mode():
    # average_subjects=True sets predefined_mapping to None (main.py)
    logged, metric = _run_epoch_end(None, {0: 0.21})
    assert logged == {"test/pearson_grouped/0": 0.21}
    assert metric.was_reset
