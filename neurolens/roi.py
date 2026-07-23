"""Human-friendly ROI groups mapped to HCP atlas regions."""

import numpy as np

# Maps friendly names to lists of HCP MMP1.0 region name prefixes.
ROI_GROUPS: dict[str, list[str]] = {
    "Visual Cortex": ["V1", "V2", "V3", "V4"],
    "Auditory Cortex": ["A1", "A4", "A5", "RI", "MBelt", "LBelt", "PBelt"],
    "Language Areas": ["44", "45", "47l", "IFSa", "IFSp", "IFJa", "IFJp",
                       "STSda", "STSdp", "STSva", "STSvp", "STV",
                       "TPOJ1", "TPOJ2", "TPOJ3"],
    "Motor Cortex": ["4", "3a", "3b", "1", "2"],
    "Prefrontal Cortex": ["8Av", "8Ad", "8BL", "8C", "9a", "9p", "9m",
                          "10d", "10r", "10v", "46", "p9-46v", "a9-46v"],
    "Temporal Cortex": ["TE1a", "TE1m", "TE1p", "TE2a", "TE2p",
                        "TGd", "TGv", "TF"],
    "Parietal Cortex": ["7AL", "7Am", "7PC", "7PL", "7Pm",
                        "AIP", "IP0", "IP1", "IP2", "LIPd", "LIPv",
                        "MIP", "VIP"],
    "Somatosensory Cortex": ["3a", "3b", "1", "2"],
    "Face-Selective Areas": ["FFC", "OFA", "PeEc"],
}


def get_roi_group_names() -> list[str]:
    """Return sorted list of all ROI group names."""
    return sorted(ROI_GROUPS.keys())


def summarize_by_roi_group(
    data: np.ndarray, mesh: str = "fsaverage5"
) -> dict[str, float]:
    """Compute mean activation per ROI group.

    Parameters
    ----------
    data : np.ndarray
        1D array of shape (n_vertices,) on fsaverage5 (20484 vertices).
    mesh : str
        Mesh resolution name.

    Returns
    -------
    dict mapping ROI group name to mean activation (float).
    """
    from tribev2.utils import get_hcp_roi_indices

    result = {}
    for group_name, regions in ROI_GROUPS.items():
        all_indices = []
        for region in regions:
            try:
                indices = get_hcp_roi_indices(region, hemi="both", mesh=mesh)
                all_indices.append(indices)
            except ValueError:
                # Region not found in atlas — skip it
                continue
        if all_indices:
            combined = np.concatenate(all_indices)
            # Deduplicate indices
            combined = np.unique(combined)
            result[group_name] = float(data[combined].mean())
        else:
            result[group_name] = 0.0
    return result
