"""HCP-MMP1 region-to-network grouping tables.

Region names match MNE-Python's ``fetch_hcp_mmp_parcellation()`` output
after stripping the ``L_``/``R_`` prefix and ``_ROI`` suffix (e.g.
``"L_V1_ROI"`` -> ``"V1"``).

Sources:
    Fine groups: MNE-Python's HCPMMP1_combined annotation, from
    Glasser et al. (2016) "A multi-modal parcellation of human cerebral
    cortex" — Nature 536:171-178.

    Coarse groups: Yeo et al. (2011) 7-network assignment via spatial
    overlap between Glasser parcels and Yeo resting-state networks on
    the fsaverage surface.

Each region appears exactly once per grouping level. 180 regions total
(per hemisphere).
"""

# ======================================================================
# Fine-grained 22-network grouping (Glasser 2016 / MNE HCPMMP1_combined)
# ======================================================================

FINE_GROUPS: dict[str, list[str]] = {
    "Primary Visual": [
        "V1",
    ],
    "Early Visual": [
        "V2", "V3", "V4",
    ],
    "Dorsal Stream Visual": [
        "V3A", "V3B", "V6", "V6A", "V7", "IPS1",
    ],
    "Ventral Stream Visual": [
        "V8", "VVC", "PIT", "FFC", "VMV1", "VMV2", "VMV3",
    ],
    "MT+ Complex and Neighboring Areas": [
        "V3CD", "LO1", "LO2", "LO3", "V4t", "FST", "MT", "MST", "PH",
    ],
    "Somatosensory and Motor": [
        "4", "3a", "3b", "1", "2",
    ],
    "Paracentral Lobular and Mid-Cingulate": [
        "24dd", "24dv", "6mp", "6ma", "SCEF", "5m", "5L", "5mv",
    ],
    "Premotor": [
        "55b", "6d", "6a", "FEF", "6v", "6r", "PEF",
    ],
    "Posterior Opercular": [
        "43", "FOP1", "OP4", "OP1", "OP2-3", "PFcm",
    ],
    "Early Auditory": [
        "A1", "LBelt", "MBelt", "PBelt", "RI",
    ],
    "Auditory Association": [
        "A4", "A5", "STSdp", "STSda", "STSvp", "STSva", "STGa", "TA2",
    ],
    "Insular and Frontal Opercular": [
        "52", "PI", "Ig", "PoI1", "PoI2", "FOP2", "FOP3", "MI", "AVI",
        "AAIC", "Pir", "FOP4", "FOP5",
    ],
    "Medial Temporal": [
        "H", "PreS", "EC", "PeEc", "PHA1", "PHA2", "PHA3",
    ],
    "Lateral Temporal": [
        "PHT", "TE1p", "TE1m", "TE1a", "TE2p", "TE2a", "TGv", "TGd", "TF",
    ],
    "Temporo-Parieto-Occipital Junction": [
        "TPOJ1", "TPOJ2", "TPOJ3", "STV", "PSL",
    ],
    "Superior Parietal": [
        "LIPv", "LIPd", "VIP", "AIP", "MIP", "7PC", "7AL", "7Am", "7PL",
        "7Pm",
    ],
    "Inferior Parietal": [
        "PGp", "PGs", "PGi", "PFm", "PF", "PFt", "PFop", "IP0", "IP1",
        "IP2",
    ],
    "Posterior Cingulate": [
        "DVT", "ProS", "POS1", "POS2", "RSC", "v23ab", "d23ab", "31pv",
        "31pd", "31a", "23d", "23c", "PCV", "7m",
    ],
    "Anterior Cingulate and Medial Prefrontal": [
        "33pr", "p24pr", "a24pr", "p24", "a24", "p32pr", "a32pr", "d32",
        "p32", "s32", "8BM", "9m", "10v", "10r", "25",
    ],
    "Orbital and Polar Frontal": [
        "47s", "47m", "a47r", "11l", "13l", "a10p", "p10p", "10pp", "10d",
        "OFC", "pOFC",
    ],
    "Inferior Frontal": [
        "44", "45", "IFJp", "IFJa", "IFSp", "IFSa", "47l", "p47r",
    ],
    "Dorsolateral Prefrontal": [
        "8C", "8Av", "i6-8", "s6-8", "SFL", "8BL", "9p", "9a", "8Ad",
        "p9-46v", "a9-46v", "46", "9-46d",
    ],
}

# ======================================================================
# Coarse 7-network grouping (Yeo 2011 spatial overlap)
#
# NOTE: These are derived from spatial overlap between Glasser parcels
# and Yeo resting-state networks, NOT by collapsing the 22 fine groups.
# Some assignments may seem surprising (e.g. A1 -> Somatomotor,
# H -> Visual) because the Yeo parcellation is surface-based and some
# regions straddle network boundaries.
# ======================================================================

COARSE_GROUPS: dict[str, list[str]] = {
    "Visual": [
        "DVT", "FFC", "FST", "H", "LO1", "LO2", "LO3", "MST", "MT",
        "PHA1", "PHA2", "PHA3", "PIT", "PreS", "ProS", "TPOJ3", "V1", "V2",
        "V3", "V3A", "V3B", "V3CD", "V4", "V4t", "V6", "V6A", "V7", "V8",
        "VMV1", "VMV2", "VMV3", "VVC",
    ],
    "Somatomotor": [
        "1", "24dd", "24dv", "3a", "3b", "4", "43", "52", "5L", "5m", "6d",
        "6mp", "6v", "A1", "A4", "A5", "FOP2", "Ig", "LBelt", "MBelt",
        "OP1", "OP2-3", "OP4", "PBelt", "PFcm", "RI", "TA2", "TPOJ1",
    ],
    "Dorsal Attention": [
        "2", "6a", "6ma", "7AL", "7Am", "7PC", "7PL", "AIP", "FEF", "IFJp",
        "IP0", "IPS1", "LIPd", "LIPv", "MIP", "PEF", "PFt", "PGp", "PH",
        "PHT", "TE2p", "TPOJ2", "VIP",
    ],
    "Ventral Attention": [
        "23c", "33pr", "5mv", "6r", "9-46d", "AAIC", "AVI", "FOP1", "FOP3",
        "FOP4", "FOP5", "MI", "PF", "PFop", "PI", "PSL", "Pir", "PoI1",
        "PoI2", "SCEF", "SFL", "a24pr", "p24pr", "p32pr",
    ],
    "Limbic": [
        "10pp", "13l", "25", "EC", "OFC", "PeEc", "TE2a", "TF", "TGd",
        "TGv", "pOFC",
    ],
    "Frontoparietal": [
        "11l", "31a", "44", "46", "55b", "7Pm", "8Av", "8BM", "8C", "IFJa",
        "IFSa", "IFSp", "IP1", "IP2", "POS2", "TE1m", "TE1p", "a10p",
        "a32pr", "a47r", "a9-46v", "i6-8", "p10p", "p47r", "p9-46v", "s6-8",
    ],
    "Default": [
        "10d", "10r", "10v", "23d", "31pd", "31pv", "45", "47l", "47m",
        "47s", "7m", "8Ad", "8BL", "9a", "9m", "9p", "PCV", "PFm", "PGi",
        "PGs", "POS1", "RSC", "STGa", "STSda", "STSdp", "STSva", "STSvp",
        "STV", "TE1a", "a24", "d23ab", "d32", "p24", "p32", "s32", "v23ab",
    ],
}


# ======================================================================
# Validation — confirm all 180 regions are accounted for, no duplicates
# ======================================================================

def _validate_groups(groups: dict[str, list[str]], name: str) -> None:
    all_regions = []
    for regions in groups.values():
        all_regions.extend(regions)
    unique = set(all_regions)
    dupes = [r for r in all_regions if all_regions.count(r) > 1]
    assert not dupes, f"{name} has duplicates: {set(dupes)}"
    assert len(unique) == 180, (
        f"{name} has {len(unique)} unique regions, expected 180"
    )


_validate_groups(FINE_GROUPS, "FINE_GROUPS")
_validate_groups(COARSE_GROUPS, "COARSE_GROUPS")
