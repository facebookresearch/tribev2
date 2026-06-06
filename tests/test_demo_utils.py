# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for the ``demo_utils`` input-suffix contract."""

from tribev2.demo_utils import VALID_SUFFIXES


def test_expected_input_modalities():
    assert set(VALID_SUFFIXES) == {"text_path", "audio_path", "video_path"}


def test_every_modality_has_at_least_one_suffix():
    for name, suffixes in VALID_SUFFIXES.items():
        assert suffixes, f"{name} has no accepted suffixes"


def test_suffixes_are_lowercase_and_dot_prefixed():
    for suffixes in VALID_SUFFIXES.values():
        for suffix in suffixes:
            assert suffix.startswith("."), f"{suffix!r} is missing a leading dot"
            assert suffix == suffix.lower(), f"{suffix!r} is not lowercase"


def test_known_suffixes_are_accepted():
    assert ".txt" in VALID_SUFFIXES["text_path"]
    assert ".wav" in VALID_SUFFIXES["audio_path"]
    assert ".mp4" in VALID_SUFFIXES["video_path"]


def test_suffixes_do_not_overlap_across_modalities():
    all_suffixes = [s for suffixes in VALID_SUFFIXES.values() for s in suffixes]
    assert len(all_suffixes) == len(set(all_suffixes)), "a suffix maps to >1 modality"
