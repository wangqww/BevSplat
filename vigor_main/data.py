"""Thin wrapper around ``dataLoader.Vigor_dataset_gs.load_vigor_data``.

A train sample is a tuple of 8:

    0  grd          [3, GrdH, GrdW]     panoramic ground (sky masked)
    1  sat          [3, 512, 512]       satellite tile
    2  depth_img    [1, GrdH, GrdW]     pre-computed metric depth
    3  grd_ori      [3, GrdH, GrdW]     original panoramic image (unused by the Stage-1 model)
    4  gt_shift_u   []                  GT lon shift, normalized
    5  gt_shift_v   []                  GT lat shift, normalized
    6  gt_rot       []                  GT heading offset (only meaningful when --rotation_range > 0)
    7  meter_per_pixel []               per-sample meter-per-pixel (city-dependent)

The legacy ``load_vigor_data`` returns ``(train_loader, val_loader)`` for
training and ``(None, test_loader)`` for ``train=False``.
"""

from dataLoader.Vigor_dataset_gs import load_vigor_data as _load

from .config import DATA_AMOUNT, ROTATION_RANGE, SUPERVISION, VIGOR_ROOT


def load_train_and_val(
    batch_size: int,
    *,
    area: str = "same",
    rotation_range: float = ROTATION_RANGE,
    weak_supervise: bool = (SUPERVISION == "Weakly"),
    amount: float = DATA_AMOUNT,
):
    """Returns ``(train_loader, val_loader)`` for VIGOR same/cross area."""
    train_loader, val_loader = _load(
        batch_size,
        area=area,
        rotation_range=rotation_range,
        train=True,
        weak_supervise=weak_supervise,
        amount=amount,
        data_root=VIGOR_ROOT,
    )
    return train_loader, val_loader


def load_test(
    batch_size: int,
    *,
    area: str = "same",
    rotation_range: float = ROTATION_RANGE,
    weak_supervise: bool = (SUPERVISION == "Weakly"),
    amount: float = DATA_AMOUNT,
):
    """Returns the held-out test loader for VIGOR same/cross area."""
    _, test_loader = _load(
        batch_size,
        area=area,
        rotation_range=rotation_range,
        train=False,
        weak_supervise=weak_supervise,
        amount=amount,
        data_root=VIGOR_ROOT,
    )
    return test_loader
