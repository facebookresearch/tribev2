# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for ``ExtractWordsFromAudio`` language validation.

The language guard runs before any transcription subprocess, so these tests
need no audio file, GPU, or the ``uvx``/``whisperx`` toolchain.
"""

from pathlib import Path

import pytest

from tribev2.eventstransforms import ExtractWordsFromAudio


def test_unsupported_language_raises_value_error():
    with pytest.raises(ValueError, match="not supported"):
        ExtractWordsFromAudio._get_transcript_from_audio(
            Path("nonexistent.wav"), "klingon"
        )


def test_default_language_is_english():
    assert ExtractWordsFromAudio().language == "english"
