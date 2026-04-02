import numpy as np
from neurolens.roi import ROI_GROUPS, get_roi_group_names, summarize_by_roi_group


def test_roi_groups_are_nonempty():
    assert len(ROI_GROUPS) > 0
    for name, regions in ROI_GROUPS.items():
        assert isinstance(regions, list)
        assert len(regions) > 0


def test_get_roi_group_names():
    names = get_roi_group_names()
    assert isinstance(names, list)
    assert "Visual Cortex" in names
    assert "Auditory Cortex" in names
    assert "Language Areas" in names


def test_summarize_by_roi_group():
    # 20484 vertices = fsaverage5 (10242 per hemisphere * 2)
    fake_data = np.random.randn(20484)
    result = summarize_by_roi_group(fake_data)
    assert isinstance(result, dict)
    assert "Visual Cortex" in result
    for name, value in result.items():
        assert isinstance(value, float)
