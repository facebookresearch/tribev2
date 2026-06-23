# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import textwrap

import pytest
import yaml

from tribev2.config_utils import TribeConfigLoader, load_config


def test_load_config_accepts_python_tuple_tag(tmp_path):
    """The published facebook/tribev2 config uses !!python/tuple; we must
    keep loading it as an actual tuple to stay backward compatible."""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        textwrap.dedent("""\
            workdir:
              excludes: !!python/tuple
                - __pycache__
                - .git
            data:
              study:
                path: .
            """),
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert isinstance(config["workdir"]["excludes"], tuple)
    assert config["workdir"]["excludes"] == ("__pycache__", ".git")
    assert config["data"]["study"]["path"] == "."


def test_load_config_handles_plain_yaml(tmp_path):
    """Configs without any Python-specific tags must still load cleanly."""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        textwrap.dedent("""\
            data:
              study:
                path: .
            infra:
              folder: /tmp/out
            average_subjects: true
            """),
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config["data"]["study"]["path"] == "."
    assert config["infra"]["folder"] == "/tmp/out"
    assert config["average_subjects"] is True


def test_loader_rejects_python_object_new_tags():
    """The whole point of the change: arbitrary Python object construction
    via !!python/object/new:* must be rejected."""
    payload = 'cmd: !!python/object/new:os.system ["echo hacked"]\n'

    with pytest.raises(yaml.constructor.ConstructorError):
        yaml.load(payload, Loader=TribeConfigLoader)


def test_loader_rejects_python_name_tags():
    """!!python/name lookups (another arbitrary-import vector) must also
    be rejected."""
    payload = "cmd: !!python/name:os.system\n"

    with pytest.raises(yaml.constructor.ConstructorError):
        yaml.load(payload, Loader=TribeConfigLoader)
