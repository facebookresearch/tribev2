# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import typing as tp
from functools import lru_cache

import matplotlib
import nibabel as nib
import numpy as np
import pydantic
from neuralset.extractors.neuro import FSAVERAGE_SIZES
from nilearn import datasets, image, maskers, surface
from scipy.spatial import cKDTree

cached_fetch_surf_fsaverage = lru_cache(datasets.fetch_surf_fsaverage)


class BasePlotBrain(pydantic.BaseModel):
    mesh: (
        tp.Literal["fsaverage3", "fsaverage4", "fsaverage5", "fsaverage6", "fsaverage7"]
        | None
    ) = "fsaverage5"
    inflate: bool | tp.Literal["half"] = "half"
    bg_map: tp.Literal["sulcal", "curvature", "thresholded"] = "sulcal"
    hemisphere_gap: float = 0
    atlas_name: str | None = None
    atlas_dim: int | None = None
    vol_to_surf_kwargs: dict | None = None
    model_config = pydantic.ConfigDict(extra="forbid")

    VIEW_DICT: tp.ClassVar[dict] = {}

    def model_post_init(self, __context: tp.Any) -> None:
        self._mesh = self.get_mesh()

    # ------------------------------------------------------------------
    # Axes helpers
    # ------------------------------------------------------------------

    def get_axarr_and_views(self, axes, views):
        if isinstance(axes, dict):
            axes = {k: self._convert_ax(ax) for k, ax in axes.items()}
            if all(k in self.VIEW_DICT for k in axes):
                views, axarr = zip(*axes.items())
            else:
                axarr = list(axes.values())
        elif isinstance(axes, (list, np.ndarray)):
            axarr = axes
        elif isinstance(axes, matplotlib.axes.Axes):
            axarr = [axes]
        assert len(views) == len(
            axarr
        ), f"Number of views and axes must match, got {len(views)} and {len(axarr)}"
        return views, axarr

    def _convert_ax(self, ax):
        """Hook for subclasses that need to convert axes (e.g. 3D -> 2D)."""
        return ax

    # ------------------------------------------------------------------
    # Atlas / volume-to-surface helpers
    # ------------------------------------------------------------------

    def get_atlas(self):
        if not hasattr(self, "_atlas"):
            if self.atlas_name == "schaefer_2018":
                atlas = datasets.fetch_atlas_schaefer_2018(n_rois=self.atlas_dim)
            elif self.atlas_name == "difumo":
                atlas = datasets.fetch_atlas_difumo(dimension=self.atlas_dim)
            self._atlas = atlas
        return self._atlas

    @property
    def atlas_masker(self):
        if not hasattr(self, "_atlas_masker"):
            atlas = self.get_atlas()
            if self.atlas_name == "schaefer_2018":
                atlas_masker = maskers.NiftiLabelsMasker(labels_img=atlas["maps"])
            elif self.atlas_name == "difumo":
                atlas_masker = maskers.NiftiMapsMasker(maps_img=atlas["maps"])
            atlas_masker.fit()
            self._atlas_masker = atlas_masker
        return self._atlas_masker

    def atlas_to_surf(self, signals, img_threshold: float | None = None):
        signals_nii = self.signals_to_nii(signals)
        return self.vol_to_surf(signals_nii, img_threshold=img_threshold)

    def vol_to_surf(self, signals_nii, img_threshold: float | None = None):
        vol_to_surf_kwargs = self.vol_to_surf_kwargs or {}
        if img_threshold is not None:
            signals_nii = image.threshold_img(
                signals_nii,
                threshold=img_threshold,
                copy=False,
                copy_header=True,
            )
        fsaverage = cached_fetch_surf_fsaverage(mesh=self.mesh)
        hemis = [
            surface.vol_to_surf(
                signals_nii,
                surf_mesh=fsaverage[f"pial_{hemi}"],
                kind="ball",
                **vol_to_surf_kwargs,
            )
            for hemi in ("left", "right")
        ]
        return np.concatenate(hemis)

    def signals_to_nii(self, signals):
        out = self.atlas_masker.inverse_transform(signals)
        if isinstance(self.atlas_masker, maskers.NiftiMapsMasker):
            data = out.get_fdata()
            lo, hi = signals.min(), signals.max()
            data = (data - data.min()) / (data.max() - data.min())
            data = data * (hi - lo) + lo
            out = nib.Nifti1Image(data, out.affine, out.header)
        return out

    # ------------------------------------------------------------------
    # Mesh loading (eager – called once in model_post_init)
    # ------------------------------------------------------------------

    def get_mesh(self) -> dict:
        """Load mesh geometry and background maps for both hemispheres.

        Returns a dict with keys ``'left'``, ``'right'``, ``'both'``,
        each mapping to ``{'coords': array, 'faces': array, 'bg_map': array}``.
        The ``'both'`` entry has hemisphere_gap applied.
        """
        fs_out = cached_fetch_surf_fsaverage(self.mesh)

        out = {}
        for hemi in ("left", "right"):
            infl_out_xyz, _ = nib.load(getattr(fs_out, f"infl_{hemi}")).darrays
            pial_xyz, faces = nib.load(getattr(fs_out, f"pial_{hemi}")).darrays

            alpha = 0.5
            jr_xyz = infl_out_xyz.data * alpha + (1 - alpha) * pial_xyz.data
            if self.inflate == "half":
                coords = jr_xyz
            elif self.inflate is True:
                coords = infl_out_xyz.data
            elif self.inflate is False:
                coords = pial_xyz.data

            bg_key = "curv" if self.bg_map == "curvature" else "sulc"
            bg_map = nib.load(getattr(fs_out, f"{bg_key}_{hemi}")).darrays[0].data
            if self.bg_map == "thresholded":
                bg_map = 1.0 * (bg_map > -0.10)
                bg_map[-1] = -5
                bg_map[-2] = 2.0
            if hemi == "left":
                coords[:, 0] = coords[:, 0] - coords[:, 0].max() - self.hemisphere_gap
            else:
                coords[:, 0] = coords[:, 0] - coords[:, 0].min() + self.hemisphere_gap

            out[hemi] = dict(coords=coords, faces=faces.data, bg_map=bg_map)

        out["both"] = dict(
            coords=np.r_[out["left"]["coords"], out["right"]["coords"]],
            faces=np.r_[
                out["left"]["faces"],
                out["right"]["faces"] + out["left"]["faces"].max() + 1,
            ],
            bg_map=np.r_[out["left"]["bg_map"], out["right"]["bg_map"]],
        )

        return out

    # ------------------------------------------------------------------
    # Stat-map upsampling (lazy – called per data array)
    # ------------------------------------------------------------------

    def get_stat_map(self, data: np.ndarray) -> dict:
        """Split vertex data into hemispheres, upsampling if needed.

        Returns ``{'left': array, 'right': array, 'both': array}``.
        """
        in_mesh = None
        for name, size in FSAVERAGE_SIZES.items():
            if data.shape[0] // 2 == size:
                in_mesh = name
                break
        if in_mesh is None:
            raise ValueError(f"Incoherent number of vertices: {data.shape[0]}")

        left = data[: len(data) // 2]
        right = data[len(data) // 2 :]

        if in_mesh != self.mesh:
            fs_in = cached_fetch_surf_fsaverage(in_mesh)
            fs_out = cached_fetch_surf_fsaverage(self.mesh)
            resampled = {}
            for hemi, values in (("left", left), ("right", right)):
                infl_in_xyz, _ = nib.load(getattr(fs_in, f"infl_{hemi}")).darrays
                infl_out_xyz, _ = nib.load(getattr(fs_out, f"infl_{hemi}")).darrays
                tree = cKDTree(infl_in_xyz.data)
                distances, indices = tree.query(infl_out_xyz.data, k=5)
                if "int" in data.dtype.name:
                    # get most frequent
                    resampled[hemi] = np.apply_along_axis(
                        lambda x: np.bincount(x).argmax(), axis=1, arr=values[indices]
                    )
                else:
                    distances = np.where(distances == 0, 1e-12, distances)
                    weights = 1 / distances
                    weights = weights / weights.sum(axis=1, keepdims=True)
                    resampled[hemi] = np.sum(values[indices] * weights, axis=1)
            left, right = resampled["left"], resampled["right"]

        return dict(left=left, right=right, both=np.r_[left, right])

    def get_hemis(self, data: np.ndarray) -> dict:
        """Convenience: combine ``self._mesh`` geometry with stat-map data."""
        stat_maps = self.get_stat_map(data)
        out = {}
        for hemi in ("left", "right", "both"):
            m = self._mesh[hemi]
            out[hemi] = dict(
                stat_map=stat_maps[hemi],
                surf_mesh=(m["coords"], m["faces"]),
                bg_map=m["bg_map"],
                hemi=hemi,
            )
        return out

    # ------------------------------------------------------------------
    # Multi-timestep plotting
    # ------------------------------------------------------------------

    def plot_timesteps(
        self,
        neuro: np.ndarray | dict[str, np.ndarray],
        segments=None,
        *,
        plot_every_k_timesteps: int = 1,
        trues=None,
        norm_percentile=None,
        show_stimuli=False,
        views: str | dict[str, str] = "left",
        timestamps: list[float] | None = None,
        **kwargs,
    ):
        import matplotlib.pyplot as plt
        from tqdm import tqdm

        from tribev2.plotting.utils import robust_normalize

        TEXT_KEY, SOUND_KEY, VIDEO_KEY = "Text", "Audio", "Video"

        if isinstance(neuro, np.ndarray):
            neuro = {"Brain reponse": neuro}
        assert all(
            v.ndim == 2 for v in neuro.values()
        ), "Neuro must be a dictionary of 2D arrays"
        if isinstance(views, dict):
            assert all(
                key in views.keys() for key in neuro.keys()
            ), f"Views keys {views.keys()} do not match neuro keys {neuro.keys()}"
        total_n_timesteps = len(list(neuro.values())[0])
        assert (
            total_n_timesteps % plot_every_k_timesteps == 0
        ), f"Total number of timesteps {total_n_timesteps} must be divisible by plot_every_k_timesteps {plot_every_k_timesteps}"
        neuro = {k: v[::plot_every_k_timesteps] for k, v in neuro.items()}
        n_timesteps = len(list(neuro.values())[0])
        if timestamps is None:
            timestamps = range(
                0, n_timesteps * plot_every_k_timesteps, plot_every_k_timesteps
            )
        else:
            assert (
                len(timestamps) == n_timesteps
            ), f"Number of timestamps {len(timestamps)} must match number of timesteps {n_timesteps}"
        if norm_percentile is not None:
            neuro = {
                k: robust_normalize(v, percentile=norm_percentile)
                for k, v in neuro.items()
            }

        mosaic = [[f"{k}_{i}" for i in range(n_timesteps)] for k in neuro]
        height_ratios = [1 for _ in neuro]
        if show_stimuli:
            from tribev2.plotting.utils import get_clip

            has_image = any(get_clip(segment) is not None for segment in segments)
            stimuli_mosaic = [
                [SOUND_KEY] * n_timesteps,
                [TEXT_KEY] * n_timesteps,
            ]
            stimuli_height_ratios = [0.3, 0.3]
            if has_image:
                stimuli_mosaic = [
                    [f"{VIDEO_KEY}_{i}" for i in range(n_timesteps)]
                ] + stimuli_mosaic
                stimuli_height_ratios = [0.7] + stimuli_height_ratios
            mosaic = stimuli_mosaic + mosaic
            height_ratios = stimuli_height_ratios + height_ratios

        fig, axes = plt.subplot_mosaic(
            mosaic,
            height_ratios=height_ratios,
            figsize=(2.5 * n_timesteps, 2 * sum(height_ratios)),
            gridspec_kw={"wspace": 0.0, "hspace": 0},
        )
        for k, ax in axes.items():
            if (
                k.startswith(TEXT_KEY)
                or k.startswith(SOUND_KEY)
                or k.startswith(VIDEO_KEY)
            ):
                fig.delaxes(ax)
                axes[k] = fig.add_subplot(ax.get_subplotspec())

        for i in tqdm(range(n_timesteps), desc="Plotting..."):
            for j, (key, value) in enumerate(neuro.items()):
                self.plot_surf(
                    value[i],
                    axes=axes[f"{key}_{i}"],
                    views=views[key] if isinstance(views, dict) else views,
                    **kwargs,
                )
                if j == len(neuro) - 1:
                    title = (
                        f"t={timestamps[i]}s" if timestamps is not None else f"t={i}s"
                    )
                    fig.text(
                        0.5,
                        -0.1,
                        title,
                        transform=axes[f"{key}_{i}"].transAxes,
                        ha="center",
                        va="center",
                    )

        if show_stimuli:
            self.plot_stimuli(
                segments, axes, plot_every_k_timesteps=plot_every_k_timesteps
            )

        first_neuro_keys = [key + "_0" for key in list(neuro.keys())]
        left, full_width = (
            axes[first_neuro_keys[0]].get_position().x0,
            fig.get_figwidth(),
        )
        for key, label in zip(
            first_neuro_keys + [TEXT_KEY, SOUND_KEY, f"{VIDEO_KEY}_0"],
            list(neuro.keys()) + [TEXT_KEY, SOUND_KEY, VIDEO_KEY],
        ):
            if key not in axes:
                continue
            pos = axes[key].get_position()
            fig.text(
                left,
                (pos.y0 + pos.y1) / 2,
                label + "\n\n\n",
                rotation="vertical",
                va="center",
                ha="center",
                transform=fig.transFigure,
            )
        return fig

    @staticmethod
    def plot_stimuli(
        segments,
        axes,
        plot_every_k_timesteps=1,
    ):
        import matplotlib.pyplot as plt

        from tribev2.plotting.utils import get_audio, get_clip

        TEXT_KEY, SOUND_KEY, VIDEO_KEY = "Text", "Audio", "Video"

        # Audio is optional (e.g., silent screen recordings produce no audio).
        audio = get_audio(
            segments[0], stop_offset=(len(segments) - 1) * segments[0].duration
        )
        if audio is not None:
            soundarray = audio.to_soundarray().mean(axis=1)
            axes[SOUND_KEY].plot(soundarray, color="k")
            axes[SOUND_KEY].set_xlim(0, len(soundarray))
        if SOUND_KEY in axes:
            axes[SOUND_KEY].axis("off")
        axes[TEXT_KEY].axis("off")
        full_start, full_duration = (
            segments[0].start,
            len(segments) * segments[0].duration,
        )

        for i, segment in enumerate(segments):
            if f"{VIDEO_KEY}_0" in axes and i % plot_every_k_timesteps == 0:
                ax_idx = i // plot_every_k_timesteps
                img = get_clip(segment).get_frame(0)
                margin = img.shape[1] * 0.0
                ax = axes[f"{VIDEO_KEY}_{ax_idx}"]
                im = ax.imshow(img)
                patch = plt.matplotlib.patches.FancyBboxPatch(
                    (0, 0),
                    img.shape[1],
                    img.shape[0],
                    boxstyle="round,pad=0,rounding_size=200",
                    transform=ax.transData,
                    clip_on=False,
                    facecolor="none",
                    edgecolor="none",
                )
                ax.add_patch(patch)
                im.set_clip_path(patch)
                ax.set_xlim(-margin, img.shape[1] + margin)
                ax.set_ylim(img.shape[0] + margin, -margin)
                ax.axis("off")
            # Text/words are optional (silent or non-speech clips have no word events).
            events = segment.events
            if events is not None and "type" in events.columns:
                words = events[events.type == "Word"]
                for word in words.itertuples():
                    if word.start < full_start:
                        continue
                    axes[TEXT_KEY].text(
                        (word.start - full_start) / full_duration,
                        0.5,
                        word.text,
                        color="k",
                        transform=axes[TEXT_KEY].transAxes,
                        ha="center",
                        va="center",
                        rotation=45,
                        fontsize=10,
                    )

    def plot_timesteps_mp4(
        self,
        neuro,
        filepath,
        *,
        segments=None,
        suptitle=None,
        interpolated_fps=None,
        norm_percentile=100,
        **plot_kwargs,
    ):
        import subprocess
        from pathlib import Path

        import matplotlib.pyplot as plt
        from tqdm import tqdm

        filepath = Path(filepath)
        tmp_dir = filepath.parent / "tmp"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        for i in tqdm(range(len(neuro)), desc="Plotting..."):
            out_fig, ax = plt.subplots(1, 1, figsize=(3, 3))
            self.plot_surf(
                neuro[i],
                axes=[ax],
                **plot_kwargs,
            )
            title = suptitle or f"t = {i}s"
            out_fig.suptitle(title, fontsize=14, fontweight="bold")
            if segments:
                from tribev2.plotting.utils import get_text

                words = " ".join(get_text(segments[i]).split(" ")[-8:])
                out_fig.text(0.1, 0.92, words, fontsize=9, ha="left", va="top")
            tmp_fig = tmp_dir / f"tmp_{i:05d}.png"
            out_fig.savefig(tmp_fig, dpi=300)
            plt.close(out_fig)
        cmd = [
            "ffmpeg",
            "-y",
            "-framerate",
            str(1),
            "-i",
            f"{str(tmp_dir)}/tmp_%05d.png",
        ]
        if interpolated_fps is not None:
            cmd.append("-vf")
            cmd.append(f"minterpolate=fps={interpolated_fps}")
        cmd.extend(
            [
                "-c:v",
                "libx264",
                "-crf",
                "18",
                "-pix_fmt",
                "yuv420p",
                str(filepath),
            ]
        )
        subprocess.run(cmd)

    # ------------------------------------------------------------------
    # Rendering (subclasses must implement)
    # ------------------------------------------------------------------

    def plot_surf(self, *args, **kwargs):
        raise NotImplementedError
