import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
import logging
import os
from typing import Dict, List, Tuple
from src.data.data_io import load_hdf5_to_dict
from src.data.layout import _file_start_idx as _get_number_from_filename
from src.data.data_naming_constants import (
    Q_CART,
    Q_POLAR,
    D_MH,
    D_RS,
    Q_CART_LPF,
    X_VALS,
    RHO_VALS,
    THETA_VALS,
)
from src.utils.scale_separation_utils import (
    fourier_transform_2d,
    inverse_fourier_transform_2d,
)
from src.training_utils.loss_functions import relative_l2_error


# Sets fonts to be the same as what is used in paper
# plt.rcParams.update({"text.usetex": True, "font.family": "Computer Modern Roman"})
# (2024-12-11, OOT): this doesn't work for me -- maybe I don't have computer modern roman on my system
def set_font(font_family="Computer Modern Roman"):
    plt.rcParams.update({"text.usetex": True, "font.family": font_family})


def _set_pixel_ax(ax, label_size=None, title=None, title_size=None) -> None:
    ax.set_xlabel("x", size=label_size)
    ax.set_ylabel("y", size=label_size)
    ax.set_title(title, size=title_size)


def _set_fourier_ax(ax, label_size=None, title=None, title_size=None) -> None:
    ax.set_xlabel("$\\xi_1$", size=label_size)
    ax.set_ylabel("$\\xi_2$", size=label_size)
    ax.set_title(title, size=title_size)


def load_q_cart_from_dir(d: str) -> np.ndarray:
    """Loads the Q_CART samples from the specified directory or file."""

    if d.endswith(".h5"):
        q_cart_dd = load_hdf5_to_dict(d)
        q_cart_arr = q_cart_dd["q_cart_pred"]
    elif d.endswith(".npy"):
        q_cart_arr = np.load(d)
    else:
        file_lst = sorted(os.listdir(d), key=_get_number_from_filename)
        q_cart_lst = [load_hdf5_to_dict(os.path.join(d, x)) for x in file_lst]
        q_cart_arr = np.concatenate([x[Q_CART] for x in q_cart_lst])
    return q_cart_arr


def make_plot(
    pred: np.ndarray,
    target: np.ndarray,
    method_name: str,
    pts: np.ndarray = None,
    q_min: float = None,
    q_max: float = None,
    err_min: float = None,
    err_max: float = None,
    err_ft_min: float = None,
    err_ft_max: float = None,
    include_ft: bool = False,
) -> None:
    """
    1. Loads the predictions from the specified directory.
    2. Computes errors and FT of errors
    3. Plots
    """
    ######################################
    # Load from specified directory
    rel_l2_err = relative_l2_error(
        torch.from_numpy(pred).unsqueeze(0), torch.from_numpy(target).unsqueeze(0)
    ).item()
    logging.debug(
        "Plotting preds from method=%s, rel l2 error is %.3e",
        method_name,
        rel_l2_err,
    )
    TITLE_SIZE = 18
    LABEL_SIZE = 16
    SUPTITLE_SIZE = 20

    if q_min is None or q_max is None:
        q_min = np.min(np.concatenate((pred, target)))
        q_max = np.max(np.concatenate((pred, target)))

    ######################################
    # Compute errors
    diffs = np.abs(pred - target)
    if err_min is None or err_max is None:
        err_max = np.max(diffs[np.isfinite(diffs)])
        err_min = np.min(diffs[np.isfinite(diffs)])

    #######################################
    # FT of errors

    if include_ft:
        diffs_ft, freq_grid = fourier_transform_2d(pred - target, pts)
        diffs_ft = np.log10(np.abs(diffs_ft))
        if err_ft_max is None and err_ft_min is None:
            err_ft_max = np.nanmax(diffs_ft)
            err_ft_min = np.nanmin(diffs_ft)
        EXTENT_FOURIER = np.array(
            [freq_grid.min(), freq_grid.max(), freq_grid.min(), freq_grid.max()]
        )

    #######################################
    # Plots

    fig, ax = plt.subplots(1, 3 + int(include_ft))
    fig.set_size_inches(15 + 5 * int(include_ft), 5)
    fig.patch.set_facecolor("white")
    EXTENT = np.array([-0.5, 0.5, -0.5, 0.5])
    ASPECT = 1.0

    # title = "{}\n\n Rel. L2 Err= {:.3f}".format(method_name, rel_l2_err)
    title = "{}".format(method_name)

    fig.text(0.02, 0.5, title, va="center", rotation="vertical", fontsize=SUPTITLE_SIZE)

    # Image 0: the model's predictions
    im_0 = ax[0].imshow(pred, extent=EXTENT, aspect=ASPECT)
    im_0.set_clim(q_min, q_max)
    plt.colorbar(im_0, ax=ax[0])
    _set_pixel_ax(ax[0], LABEL_SIZE, "$\\hat{q}$", TITLE_SIZE)

    # Image 1: the target
    im_1 = ax[1].imshow(target, extent=EXTENT, aspect=ASPECT)
    im_1.set_clim(q_min, q_max)
    plt.colorbar(im_1, ax=ax[1])
    _set_pixel_ax(ax[1], LABEL_SIZE, "$q$", TITLE_SIZE)

    # Image 2: errors in pixel space
    im_4 = ax[2].imshow(diffs, extent=EXTENT, aspect=ASPECT, cmap="hot")
    im_4.set_clim(err_min, err_max)
    plt.colorbar(im_4, ax=ax[2])
    _set_pixel_ax(ax[2], LABEL_SIZE, "$|\\hat{q} - q|$", TITLE_SIZE)

    if include_ft:
        # Image 3: FT of the prediction errors
        im_10 = ax[3].imshow(diffs_ft, extent=EXTENT_FOURIER, aspect=ASPECT, cmap="hot")
        im_10.set_clim(err_ft_min, err_ft_max)
        plt.colorbar(im_10, ax=ax[3])
        _set_fourier_ax(ax[3], LABEL_SIZE, "$\\log_{10}$(F.T.(Errors))", TITLE_SIZE)
        plt.subplots_adjust(left=0.1, right=1.0, wspace=0.3)

    fig.tight_layout()
    plt.show()
    plt.clf()


def make_sequence_of_plots(
    targets_dir: str,
    preds_dd: Dict[str, str],
    sample_idxes: List[int],
    pts: np.ndarray,
    include_ft: bool = False,
) -> None:
    """
    1. Loads targets and all of the predictions.
    2. For each sample index:
        3. Computes the relavant error mins/maxes.
        4. For each method:
            5: Makes the plot

    Args:
        targets_dir (str): Directory of where to load targets from
        preds_dd (Dict[str, str]): Keys are method names to appear in the plots, values
                    are the directories in which the preds are stored.
        sample_idxes (List[int]): Which indices in the test set we want predictions
                    from.
    """

    targets_arr = load_q_cart_from_dir(targets_dir)
    preds_arr_dd = {k: load_q_cart_from_dir(v) for k, v in preds_dd.items()}

    for i in sample_idxes:
        logging.info("Working on sample=%i", i)
        target_i = targets_arr[i]
        preds_i_dd = {k: v[i] for k, v in preds_arr_dd.items()}
        q_max = np.max([x for x in preds_i_dd.values()])
        q_max = max(np.max(target_i), q_max)
        q_min = 0.0

        errors_i_dd = {k: target_i - v for k, v in preds_i_dd.items()}

        error_max = np.max([np.abs(x) for x in errors_i_dd.values()])
        error_min = 0.0

        ft_errors_i = {}
        for k, v in errors_i_dd.items():
            e, _ = fourier_transform_2d(v, pts)
            ft_errors_i[k] = np.log10(np.abs(e))

        ft_error_max = np.max([x for x in ft_errors_i.values()])
        ft_error_min = np.min([x for x in ft_errors_i.values()])

        for k, v in preds_i_dd.items():
            make_plot(
                pred=v,
                target=target_i,
                method_name=k,
                q_min=q_min,
                q_max=q_max,
                err_min=error_min,
                err_max=error_max,
                err_ft_min=ft_error_min,
                err_ft_max=ft_error_max,
                pts=pts,
                include_ft=include_ft,
            )


def make_plot_for_paper(
    targets_dir: str,
    preds_dd: Dict[str, str],
    sample_idx: int,
    save_fp: str,
) -> None:
    """
    1. Loads targets and all of the predictions.
    3. Computes the relavant error mins/maxes.
    4. For each method:
        5: Makes the plot in a row
    6. Saves the figure

    Args:
        targets_dir (str): Directory of where to load targets from
        preds_dd (Dict[str, str]): Keys are method names to appear in the plots, values
                    are the directories in which the preds are stored.
        sample_idx int: Which index in the test set we want predictions
                    from.
        save_fp (str): Where to save the figure.
    """
    EXTENT = np.array([-0.5, 0.5, -0.5, 0.5])
    ASPECT = 1.0
    LABEL_SIZE = 16
    TITLE_SIZE = 18
    SUPTITLE_SIZE = 20

    targets_arr = load_q_cart_from_dir(targets_dir)
    preds_arr_dd = {k: load_q_cart_from_dir(v) for k, v in preds_dd.items()}

    target_i = targets_arr[sample_idx]
    preds_i_dd = {k: v[sample_idx] for k, v in preds_arr_dd.items()}
    q_max = np.max([x for x in preds_i_dd.values()])
    q_max = max(np.max(target_i), q_max)
    q_min = 0.0

    logging.info("q_max is : %f", q_max)

    errors_i_dd = {k: target_i - v for k, v in preds_i_dd.items()}

    error_max = np.max([np.abs(x) for x in errors_i_dd.values()])
    error_min = 0.0

    fig, ax = plt.subplots(len(preds_i_dd), 2)
    fig.set_size_inches(10, 5 * len(preds_i_dd))

    for i, (k, v) in enumerate(preds_i_dd.items()):

        title = "{}".format(k)

        # fig.text(
        #     0.02,
        #     1 * (i + 1) / len(preds_i_dd) - 0.3,
        #     title,
        #     va="center",
        #     rotation="vertical",
        #     fontsize=SUPTITLE_SIZE,
        # )
        _fill_ax_for_paper(
            ax[i],
            pred=v,
            target=target_i,
            q_max=q_max,
            q_min=q_min,
            err_max=error_max,
            err_min=error_min,
            method_name=k,
            EXTENT=EXTENT,
            ASPECT=ASPECT,
            LABEL_SIZE=LABEL_SIZE,
            TITLE_SIZE=TITLE_SIZE,
        )

    fig.tight_layout()
    plt.savefig(save_fp)
    plt.show()
    plt.clf()


def _fill_ax_for_paper(
    ax,
    pred: np.ndarray,
    target: np.ndarray,
    q_max: float,
    q_min: float,
    err_max: float,
    err_min: float,
    method_name: str,
    EXTENT: np.ndarray,
    ASPECT: float,
    LABEL_SIZE: int,
    TITLE_SIZE: int,
    TICK_SIZE: int = None,
) -> None:
    """Expects ax to be a list of 2 axes. In the first axis, it will plot the
    predictions. Second axis will plot the absolute value of the errors."""

    # Image 0: the model's predictions
    im_0 = ax[0].imshow(pred, extent=EXTENT, aspect=ASPECT)
    im_0.set_clim(q_min, q_max)
    cm_0 = plt.colorbar(im_0, ax=ax[0])
    ax[0].set_ylabel(method_name, size=TITLE_SIZE)
    # _set_pixel_ax(ax[0], LABEL_SIZE, "$\\hat{q}$", TITLE_SIZE)

    # Image 1: errors in pixel space
    im_4 = ax[1].imshow(np.abs(pred - target), extent=EXTENT, aspect=ASPECT, cmap="hot")
    im_4.set_clim(err_min, err_max)
    cm_1 = plt.colorbar(im_4, ax=ax[1])
    # _set_pixel_ax(ax[1], LABEL_SIZE, "$|\\hat{q} - q|$", TITLE_SIZE)

    ax[0].set_xticks(np.array([-0.4, 0.0, 0.4]))
    ax[0].set_yticks(np.array([-0.4, 0.0, 0.4]))

    ax[1].set_xticks(np.array([-0.4, 0.0, 0.4]))
    ax[1].set_yticks(np.array([-0.4, 0.0, 0.4]))

    if TICK_SIZE is not None:
        ax[0].tick_params(axis="both", which="major", labelsize=TICK_SIZE)
        ax[1].tick_params(axis="both", which="major", labelsize=TICK_SIZE)
        cm_0.ax.tick_params(labelsize=TICK_SIZE)
        cm_1.ax.tick_params(labelsize=TICK_SIZE)
    #     plt.colorbar(im_0, ax=ax[0], labelsize=TICK_SIZE)
    #     plt.colorbar(im_4, ax=ax[1], labelsize=TICK_SIZE)
    # else:
    #     plt.colorbar(im_0, ax=ax[0])
    #     plt.colorbar(im_4, ax=ax[1])


def make_plot_along_n_freqs(
    targets_dir: str,
    preds_dd: Dict[str, List[str]],
    sample_idx: int,
    save_fp: str,
    col_labels: List[str],
    names_ordered_lst: List[str] = None,
) -> None:
    """
    1. Loads targets and all predictions.
    2. Computes the relavant error mins/maxes.
    3. For each method:
        4. For each number of input frequencies:
            5. Plots the prediction and errors in a row.
    6. Saves the figure

    Args:
        targets_dir (str): Directory of where to load targets from
        preds_dd (Dict[str, List[str]]): Keys are names of the methods, values are lists
                of directories where the predictions are stored.
        sample_idx (int): Which index in the test set we want to show
        save_fp (str): Where to save the figure.
        names_ordered_lst (List[str]): The order in which to plot the methods. If None, alphabetical.
    """
    EXTENT = np.array([-0.5, 0.5, -0.5, 0.5])
    ASPECT = 1.0
    LABEL_SIZE = 16
    TITLE_SIZE = 18
    SUPTITLE_SIZE = 20
    OUTER_SPACE_FRAC = 0.17
    INNER_SPACE_FRAC = 0.15
    VERTICAL_SPACE_FRAC = 0.1

    n_methods = len(preds_dd)
    n_freqs = len(list(preds_dd.values())[0])

    targets_arr = load_q_cart_from_dir(targets_dir)
    target_i = targets_arr[sample_idx]

    preds_i_dd = {}
    for k, v in preds_dd.items():
        preds_i_dd[k] = [load_q_cart_from_dir(x)[sample_idx] for x in v]

    # Compute the max and min values for the predictions and errors
    q_max = np.max([x for x in preds_i_dd.values()])
    q_max = max(np.max(target_i), q_max)
    q_min = 0.0

    # Loop through preds_i_dd and subtract from target_i to get errors_i_dd
    errors_i_dd = {}
    for k, v in preds_i_dd.items():
        errors_i_dd[k] = [target_i - x for x in v]

    # Compute max errors
    error_max = np.max([np.abs(x) for x in errors_i_dd.values()])
    error_min = 0.0

    # fig, ax = plt.subplots(n_methods, n_freqs * 2)
    fig = plt.figure()
    fig.set_size_inches(10 * n_freqs, 5 * n_methods)

    if names_ordered_lst is None:
        names_ordered_lst = sorted(preds_i_dd.keys())

    # Outer gridspec to make n_freqs columns
    gs_outer = GridSpec(
        1, n_freqs, wspace=OUTER_SPACE_FRAC, bottom=0.05, top=0.9, left=0.05, right=0.95
    )

    for j in range(n_freqs):
        gs_inner_j = GridSpecFromSubplotSpec(
            n_methods,
            2,
            gs_outer[j],
            wspace=INNER_SPACE_FRAC,
            hspace=VERTICAL_SPACE_FRAC,
        )

        # Write column_label[j] halfway between the 0th and 1st column
        col_label = col_labels[j]

        for i in range(n_methods):
            if j == 0:
                name_plt = names_ordered_lst[i]
            else:
                name_plt = ""

            ax_ij_preds = fig.add_subplot(gs_inner_j[i, 0])
            ax_ij_errors = fig.add_subplot(gs_inner_j[i, 1])

            if i == 0:
                ax_ij_preds.set_title("Prediction", size=TITLE_SIZE)
                ax_ij_errors.set_title("Error", size=TITLE_SIZE)

                # Get the bounding box of the top-row prediction axis object.
                # Use this to find the x position of where to write the column label.
                bbox_points = ax_ij_preds.get_position().get_points()
                xpos = bbox_points[1, 0]
                fig.text(
                    xpos,
                    0.95,
                    col_label,
                    va="center",
                    ha="center",
                    fontsize=SUPTITLE_SIZE,
                )

            _fill_ax_for_paper(
                [ax_ij_preds, ax_ij_errors],
                pred=preds_i_dd[names_ordered_lst[i]][j],
                target=target_i,
                q_max=q_max,
                q_min=q_min,
                err_max=error_max,
                err_min=error_min,
                method_name=name_plt,
                EXTENT=EXTENT,
                ASPECT=ASPECT,
                LABEL_SIZE=LABEL_SIZE,
                TITLE_SIZE=TITLE_SIZE,
            )

    plt.savefig(save_fp)
    plt.show()
    plt.clf()


def make_plot_along_n_freqs_2(
    targets_dir: str,
    preds_dd: Dict[str, List[str]],
    sample_idx: int,
    save_fp_format: str,
    names_ordered_lst: List[str] = None,
) -> None:
    """
    1. Loads targets and all predictions.
    2. Computes the relavant error mins/maxes.
    3. For each method:
        4. For each number of input frequencies:
            5. Plots the prediction and errors in a row.
    6. Saves the figure

    Args:
        targets_dir (str): Directory of where to load targets from
        preds_dd (Dict[str, List[str]]): Keys are names of the methods, values are lists
                of directories where the predictions are stored.
        sample_idx (int): Which index in the test set we want to show
        save_fp_format (str): Where to save the figure.
        names_ordered_lst (List[str]): The order in which to plot the methods. If None, alphabetical.
    """
    EXTENT = np.array([-0.5, 0.5, -0.5, 0.5])
    ASPECT = 1.0
    LABEL_SIZE = 16
    TITLE_SIZE = 18
    SUPTITLE_SIZE = 20
    OUTER_SPACE_FRAC = 0.17
    INNER_SPACE_FRAC = 0.15
    VERTICAL_SPACE_FRAC = 0.1
    TOP = 0.9
    BOTTOM = 0.05
    LEFT = 0.05
    RIGHT = 0.95
    TICK_SIZE = 17
    BIGTICK_SIZE = 22

    n_methods = len(preds_dd) - 1
    n_freqs = len(list(preds_dd.values())[0])

    targets_arr = load_q_cart_from_dir(targets_dir)
    target_i = targets_arr[sample_idx]

    preds_i_dd = {}
    for k, v in preds_dd.items():
        preds_i_dd[k] = [load_q_cart_from_dir(x)[sample_idx] for x in v]

    fynet_preds = preds_i_dd.pop("FYNet")[0]

    # Compute the max and min values for the predictions and errors
    q_max = np.max([x for x in preds_i_dd.values()])
    q_max = max(np.max(target_i), q_max, fynet_preds.max())
    q_min = 0.0

    # Loop through preds_i_dd and subtract from target_i to get errors_i_dd
    errors_i_dd = {}
    for k, v in preds_i_dd.items():
        errors_i_dd[k] = [target_i - x for x in v]

    # Compute max errors
    error_max = np.max([np.abs(x) for x in errors_i_dd.values()])
    error_min = 0.0

    if names_ordered_lst is None:
        names_ordered_lst = sorted(preds_i_dd.keys())

    ################################################################################
    # FIRST FIG: FYNet preds and errors, and the ground-truth scattering object.
    fig = plt.figure()
    fig.set_size_inches(10, 15)

    gs = GridSpec(3, 2, top=TOP, bottom=BOTTOM, left=LEFT, right=RIGHT)

    ax_preds = fig.add_subplot(gs[0, 0])
    ax_errors = fig.add_subplot(gs[0, 1])
    ax_preds.set_title("Prediction", size=TITLE_SIZE)
    ax_errors.set_title("Error", size=TITLE_SIZE)

    _fill_ax_for_paper(
        [ax_preds, ax_errors],
        pred=fynet_preds,
        target=target_i,
        q_max=q_max,
        q_min=q_min,
        err_max=error_max,
        err_min=error_min,
        method_name="FYNet",
        EXTENT=EXTENT,
        ASPECT=ASPECT,
        LABEL_SIZE=LABEL_SIZE,
        TITLE_SIZE=TITLE_SIZE,
        TICK_SIZE=TICK_SIZE,
    )

    ax_ground_truth = fig.add_subplot(gs[1:, :])

    # Plot the ground-truth scattering object
    im_1 = ax_ground_truth.imshow(target_i, extent=EXTENT, aspect=ASPECT)
    im_1.set_clim(q_min, q_max)
    cb = plt.colorbar(im_1, ax=ax_ground_truth)
    # increase the font size of the ticks and colorbar
    cb.ax.tick_params(labelsize=BIGTICK_SIZE)
    ax_ground_truth.tick_params(axis="both", which="major", labelsize=BIGTICK_SIZE)

    plt.savefig(save_fp_format.format(0))
    plt.show()
    plt.clf()

    ################################################################################
    # SECOND FIG: Predictions and errors for the first element in each list of preds_i_dd

    fig = plt.figure()
    fig.set_size_inches(10, 5 * n_methods)

    gs = GridSpec(n_methods, 2, top=TOP, bottom=BOTTOM, left=LEFT, right=RIGHT)

    for i in range(n_methods):

        ax_preds = fig.add_subplot(gs[i, 0])
        ax_errors = fig.add_subplot(gs[i, 1])

        if i == 0:
            ax_preds.set_title("Prediction", size=TITLE_SIZE)
            ax_errors.set_title("Error", size=TITLE_SIZE)

        name_for_plt = names_ordered_lst[i]

        _fill_ax_for_paper(
            [ax_preds, ax_errors],
            pred=preds_i_dd[names_ordered_lst[i]][0],
            target=target_i,
            q_max=q_max,
            q_min=q_min,
            err_max=error_max,
            err_min=error_min,
            method_name=name_for_plt,
            EXTENT=EXTENT,
            ASPECT=ASPECT,
            LABEL_SIZE=LABEL_SIZE,
            TITLE_SIZE=TITLE_SIZE,
            TICK_SIZE=TICK_SIZE,
        )

    plt.savefig(save_fp_format.format(1))
    plt.show()
    plt.clf()

    ################################################################################
    # THIRD FIG: Predictions and errors for the second element in each list of preds_i_dd

    fig = plt.figure()
    fig.set_size_inches(10, 5 * n_methods)

    gs = GridSpec(n_methods, 2, top=TOP, bottom=BOTTOM, left=LEFT, right=RIGHT)

    for i in range(n_methods):

        ax_preds = fig.add_subplot(gs[i, 0])
        ax_errors = fig.add_subplot(gs[i, 1])

        if i == 0:
            ax_preds.set_title("Prediction", size=TITLE_SIZE)
            ax_errors.set_title("Error", size=TITLE_SIZE)

        name_for_plt = names_ordered_lst[i]

        _fill_ax_for_paper(
            [ax_preds, ax_errors],
            pred=preds_i_dd[names_ordered_lst[i]][1],
            target=target_i,
            q_max=q_max,
            q_min=q_min,
            err_max=error_max,
            err_min=error_min,
            method_name=name_for_plt,
            EXTENT=EXTENT,
            ASPECT=ASPECT,
            LABEL_SIZE=LABEL_SIZE,
            TITLE_SIZE=TITLE_SIZE,
            TICK_SIZE=TICK_SIZE,
        )

    plt.savefig(save_fp_format.format(2))
    plt.show()
    plt.clf()


def make_plot_along_n_freqs_3(
    targets_dir: str,
    preds_dd: Dict[str, List[str]],
    sample_idx: int,
    save_fp_format: str,
    names_ordered_lst: List[str] = None,
) -> None:
    """
    This is for the main text of the JCP submission.


    1. Loads targets and all predictions.
    2. Computes the relavant error mins/maxes.
    3. For each method:
        4. For each number of input frequencies:
            5. Plots the prediction and errors in a row.
    6. Saves the figure

    Args:
        targets_dir (str): Directory of where to load targets from
        preds_dd (Dict[str, List[str]]): Keys are names of the methods, values are lists
                of directories where the predictions are stored.
        sample_idx (int): Which index in the test set we want to show
        save_fp_format (str): Where to save the figure.
        names_ordered_lst (List[str]): The order in which to plot the methods. If None, alphabetical.
    """
    EXTENT = np.array([-0.5, 0.5, -0.5, 0.5])
    ASPECT = 1.0
    LABEL_SIZE = 16
    TITLE_SIZE = 22
    SUPTITLE_SIZE = 20
    OUTER_SPACE_FRAC = 0.17
    INNER_SPACE_FRAC = 0.15
    VERTICAL_SPACE_FRAC = 0.1
    TOP = 0.9
    BOTTOM = 0.05
    LEFT = 0.05
    RIGHT = 0.95
    TICK_SIZE = 17
    BIGTICK_SIZE = 22

    n_methods = len(preds_dd) - 1
    n_freqs = len(list(preds_dd.values())[0])

    targets_arr = load_q_cart_from_dir(targets_dir)
    target_i = targets_arr[sample_idx]

    preds_i_dd = {}
    for k, v in preds_dd.items():
        preds_i_dd[k] = [load_q_cart_from_dir(x)[sample_idx] for x in v]

    fynet_preds = preds_i_dd.pop("FYNet")[0]

    # Compute the max and min values for the predictions and errors
    q_max = np.max([x for x in preds_i_dd.values()])
    q_max = max(np.max(target_i), q_max, fynet_preds.max())
    q_min = 0.0

    # Loop through preds_i_dd and subtract from target_i to get errors_i_dd
    errors_i_dd = {}
    for k, v in preds_i_dd.items():
        errors_i_dd[k] = [target_i - x for x in v]

    # Compute max errors
    error_max = np.max([np.abs(x) for x in errors_i_dd.values()])
    error_min = 0.0

    if names_ordered_lst is None:
        names_ordered_lst = sorted(preds_i_dd.keys())

    fig = plt.figure()
    fig.set_size_inches(15, 25)

    gs = GridSpec(5, 3, top=TOP, bottom=BOTTOM, left=LEFT, right=RIGHT)

    ################################################################################
    # FIRST ROW: Ground-truth, FYNet preds and errors.
    ax_gt = fig.add_subplot(gs[0, 0])
    ax_preds = fig.add_subplot(gs[0, 1])
    ax_errors = fig.add_subplot(gs[0, 2])

    _fill_ax_for_paper(
        [ax_preds, ax_errors],
        pred=fynet_preds,
        target=target_i,
        q_max=q_max,
        q_min=q_min,
        err_max=error_max,
        err_min=error_min,
        method_name=None,
        EXTENT=EXTENT,
        ASPECT=ASPECT,
        LABEL_SIZE=LABEL_SIZE,
        TITLE_SIZE=TITLE_SIZE,
        TICK_SIZE=BIGTICK_SIZE,
    )

    im_1 = ax_gt.imshow(target_i, extent=EXTENT, aspect=ASPECT)
    im_1.set_clim(q_min, q_max)
    cb = plt.colorbar(im_1, ax=ax_gt)
    # increase the font size of the ticks and colorbar
    cb.ax.tick_params(labelsize=BIGTICK_SIZE)
    ax_gt.tick_params(axis="both", which="major", labelsize=BIGTICK_SIZE)
    ax_gt.set_ylabel("Ground-Truth and $N_k=1$", size=TITLE_SIZE)

    ax_gt.set_title("Ground-Truth", size=TITLE_SIZE)
    ax_preds.set_title("FYNet Predictions", size=TITLE_SIZE)
    ax_errors.set_title("FYNet Errors", size=TITLE_SIZE)

    ################################################################################
    # SECOND AND THIRD ROWS: N_k = 3 predictions and errors.

    for i in range(3):
        ax_preds = fig.add_subplot(gs[1, i])
        ax_errors = fig.add_subplot(gs[2, i])

        _fill_ax_for_paper(
            [ax_preds, ax_errors],
            pred=preds_i_dd[names_ordered_lst[i]][0],
            target=target_i,
            q_max=q_max,
            q_min=q_min,
            err_max=error_max,
            err_min=error_min,
            method_name=None,
            EXTENT=EXTENT,
            ASPECT=ASPECT,
            LABEL_SIZE=LABEL_SIZE,
            TITLE_SIZE=TITLE_SIZE,
            TICK_SIZE=BIGTICK_SIZE,
        )

        ax_preds.set_title(names_ordered_lst[i], size=TITLE_SIZE)

        if i == 0:
            ax_preds.set_ylabel("Predictions, $N_k=3$", size=TITLE_SIZE)
            ax_errors.set_ylabel("Errors, $N_k=3$", size=TITLE_SIZE)

    ################################################################################
    # FOURTH AND FIFTH ROWS: N_k = 5 predictions and errors.

    for i in range(3):
        ax_preds = fig.add_subplot(gs[3, i])
        ax_errors = fig.add_subplot(gs[4, i])

        _fill_ax_for_paper(
            [ax_preds, ax_errors],
            pred=preds_i_dd[names_ordered_lst[i]][1],
            target=target_i,
            q_max=q_max,
            q_min=q_min,
            err_max=error_max,
            err_min=error_min,
            method_name=None,
            EXTENT=EXTENT,
            ASPECT=ASPECT,
            LABEL_SIZE=LABEL_SIZE,
            TITLE_SIZE=TITLE_SIZE,
            TICK_SIZE=BIGTICK_SIZE,
        )

        if i == 0:
            ax_preds.set_ylabel("Predictions, $N_k=5$", size=TITLE_SIZE)
            ax_errors.set_ylabel("Errors, $N_k=5$", size=TITLE_SIZE)

        ax_preds.set_title(names_ordered_lst[i], size=TITLE_SIZE)
    plt.savefig(save_fp_format.format(0))
    plt.show()
    plt.clf()


def make_plot_along_n_freqs_4(
    targets_dir: str,
    preds_dd: Dict[str, List[str]],
    sample_idx: int,
    save_fp_format: str,
    names_ordered_lst: List[str] = None,
) -> None:
    """
    This is for the main text of the JCP submission.


    1. Loads targets and all predictions.
    2. Computes the relavant error mins/maxes.
    3. For each method:
        4. For each number of input frequencies:
            5. Plots the prediction and errors in a row.
    6. Saves the figure

    Args:
        targets_dir (str): Directory of where to load targets from
        preds_dd (Dict[str, List[str]]): Keys are names of the methods, values are lists
                of directories where the predictions are stored.
        sample_idx (int): Which index in the test set we want to show
        save_fp_format (str): Where to save the figure.
        names_ordered_lst (List[str]): The order in which to plot the methods. If None, alphabetical.
    """
    EXTENT = np.array([-0.5, 0.5, -0.5, 0.5])
    ASPECT = 1.0
    LABEL_SIZE = 8
    TITLE_SIZE = 8
    TICK_SIZE = 8

    # TOP = 0.99
    # BOTTOM = 0.01
    # LEFT = 0.1
    # RIGHT = 0.9
    TOP = None
    BOTTOM = None
    LEFT = None
    RIGHT = None
    WSPACE = 0.3
    HSPACE = 0.3

    BIGTICK_SIZE = 11
    BIGLABEL_SIZE = 11
    BIGTITLE_SIZE = 11

    n_methods = len(preds_dd) - 1
    n_freqs = len(list(preds_dd.values())[0])

    targets_arr = load_q_cart_from_dir(targets_dir)
    target_i = targets_arr[sample_idx]

    preds_i_dd = {}
    for k, v in preds_dd.items():
        preds_i_dd[k] = [load_q_cart_from_dir(x)[sample_idx] for x in v]

    fynet_preds = preds_i_dd.pop("FYNet")[0]

    # Compute the max and min values for the predictions and errors
    q_max = np.max([x for x in preds_i_dd.values()])
    q_max = max(np.max(target_i), q_max, fynet_preds.max())
    q_min = 0.0

    # Loop through preds_i_dd and subtract from target_i to get errors_i_dd
    errors_i_dd = {}
    for k, v in preds_i_dd.items():
        errors_i_dd[k] = [target_i - x for x in v]

    # Compute max errors
    error_max = np.max([np.abs(x) for x in errors_i_dd.values()])
    error_min = 0.0

    if names_ordered_lst is None:
        names_ordered_lst = sorted(preds_i_dd.keys())

    fig = plt.figure()
    fig.set_size_inches(3.50, 3.45 / 2 * 7)

    gs = GridSpec(
        7,
        2,
        top=TOP,
        bottom=BOTTOM,
        left=LEFT,
        right=RIGHT,
        wspace=WSPACE,
        hspace=HSPACE,
    )

    ################################################################################
    # FIRST ROW: Ground-truth, FYNet preds and errors.
    ax_gt = fig.add_subplot(gs[:2, :2])
    ax_preds = fig.add_subplot(gs[2, 0])
    ax_errors = fig.add_subplot(gs[2, 1])

    _fill_ax_for_paper(
        [ax_preds, ax_errors],
        pred=fynet_preds,
        target=target_i,
        q_max=q_max,
        q_min=q_min,
        err_max=error_max,
        err_min=error_min,
        method_name=None,
        EXTENT=EXTENT,
        ASPECT=ASPECT,
        LABEL_SIZE=LABEL_SIZE,
        TITLE_SIZE=TITLE_SIZE,
        TICK_SIZE=TICK_SIZE,
    )

    im_1 = ax_gt.imshow(target_i, extent=EXTENT, aspect=ASPECT)
    im_1.set_clim(q_min, q_max)
    cb = plt.colorbar(im_1, ax=ax_gt)
    # increase the font size of the ticks and colorbar
    cb.ax.tick_params(labelsize=BIGTICK_SIZE)
    ax_gt.tick_params(axis="both", which="major", labelsize=BIGTICK_SIZE)
    ax_preds.set_ylabel("FYNet \n $N_k=1$", size=TITLE_SIZE, labelpad=2)

    ax_gt.set_title("Ground-Truth", size=BIGTITLE_SIZE)
    ax_preds.set_title("Predictions", size=TITLE_SIZE)
    ax_errors.set_title("Errors", size=TITLE_SIZE)

    ax_gt.set_xticks(np.array([-0.4, 0.0, 0.4]))
    ax_gt.set_yticks(np.array([-0.4, 0.0, 0.4]))

    ################################################################################
    # SECOND AND THIRD ROWS: N_k = 3 predictions and errors.

    for i, name in enumerate(names_ordered_lst):

        ax_method = fig.add_subplot(gs[3 + 2 * i : 5 + 2 * i, :])
        ax_method.spines[["right", "top", "left", "bottom"]].set_visible(False)
        ax_method.set_xticks([])
        ax_method.set_yticks([])
        ax_method.set_ylabel(name, size=TITLE_SIZE, labelpad=36)

        # # ax_lst = fig.add_subplot(2, 2, 0)
        # ax_preds_2 = fig.add_subplot(2, 2, 1)
        # ax_errors_2 = fig.add_subplot(2, 2, 2)

        # ax_preds_3 = fig.add_subplot(2, 2, 3)
        # ax_errors_3 = fig.add_subplot(2, 2, 4)

        ax_preds_2 = fig.add_subplot(gs[3 + 2 * i, 0])
        ax_errors_2 = fig.add_subplot(gs[3 + 2 * i, 1])

        _fill_ax_for_paper(
            [ax_preds_2, ax_errors_2],
            pred=preds_i_dd[name][0],
            target=target_i,
            q_max=q_max,
            q_min=q_min,
            err_max=error_max,
            err_min=error_min,
            method_name=None,
            EXTENT=EXTENT,
            ASPECT=ASPECT,
            LABEL_SIZE=LABEL_SIZE,
            TITLE_SIZE=TITLE_SIZE,
            TICK_SIZE=TICK_SIZE,
        )

        ax_preds_2.set_ylabel("$N_k=2$", size=TITLE_SIZE)
        # ax_preds_2.set_ylabel(name + "\n $N_k=2$", size=TITLE_SIZE)
        ax_preds_2.set_title("Predictions", size=TITLE_SIZE)
        ax_errors_2.set_title("Errors", size=TITLE_SIZE)

        ax_preds_3 = fig.add_subplot(gs[4 + 2 * i, 0])
        ax_errors_3 = fig.add_subplot(gs[4 + 2 * i, 1])

        _fill_ax_for_paper(
            [ax_preds_3, ax_errors_3],
            pred=preds_i_dd[name][1],
            target=target_i,
            q_max=q_max,
            q_min=q_min,
            err_max=error_max,
            err_min=error_min,
            method_name=None,
            EXTENT=EXTENT,
            ASPECT=ASPECT,
            LABEL_SIZE=LABEL_SIZE,
            TITLE_SIZE=TITLE_SIZE,
            TICK_SIZE=TICK_SIZE,
        )
        ax_preds_3.set_ylabel("$N_k=3$", size=TITLE_SIZE, labelpad=2)
        # ax_preds_3.set_ylabel(name + "\n $N_k=3$", size=TITLE_SIZE)
        ax_preds_3.set_title("Predictions", size=TITLE_SIZE)
        ax_errors_3.set_title("Errors", size=TITLE_SIZE)
    plt.savefig(save_fp_format.format(0), bbox_inches="tight")
    plt.show()
    plt.clf()


def make_plot_along_n_freqs_5(
    targets_dir: str,
    preds_dd: Dict[str, List[str]],
    n_samples: int,
    save_fp_format: str,
    names_ordered_lst: List[str] = None,
) -> None:
    """
    This is for the appendix of the JCP submission.


    1. Loads targets and all predictions.
    2. Computes the relavant error mins/maxes.
    3. For each method:
        4. For each number of input frequencies:
            5. Plots the prediction and errors in a row.
    6. Saves the figure

    Args:
        targets_dir (str): Directory of where to load targets from
        preds_dd (Dict[str, List[str]]): Keys are names of the methods, values are lists
                of directories where the predictions are stored.
        sample_idx (int): Which index in the test set we want to show
        save_fp_format (str): Where to save the figure.
        names_ordered_lst (List[str]): The order in which to plot the methods. If None, alphabetical.
    """
    EXTENT = np.array([-0.5, 0.5, -0.5, 0.5])
    ASPECT = 1.0
    LABEL_SIZE = 16
    TITLE_SIZE = 22
    SUPTITLE_SIZE = 20
    OUTER_SPACE_FRAC = 0.17
    INNER_SPACE_FRAC = 0.15
    VERTICAL_SPACE_FRAC = 0.1
    TOP = 0.9
    BOTTOM = 0.05
    LEFT = 0.05
    RIGHT = 0.95
    TICK_SIZE = 17
    BIGTICK_SIZE = 22

    np.random.seed(42)
    sample_idxes = np.random.choice(np.arange(500), n_samples, replace=False)

    n_methods = len(preds_dd)
    n_freqs = len(list(preds_dd.values())[0])

    targets_arr = load_q_cart_from_dir(targets_dir)
    target_i = targets_arr[sample_idxes]

    preds_i_dd = {}
    for k, v in preds_dd.items():
        preds_i_dd[k] = [load_q_cart_from_dir(x)[sample_idxes] for x in v]

    # fynet_preds = preds_i_dd.pop("FYNet")[0]

    # Compute the max and min values for the predictions and errors
    q_max = np.max([x for x in preds_i_dd.values()])
    q_max = max(np.max(target_i), q_max)
    q_min = 0.0

    # Loop through preds_i_dd and subtract from target_i to get errors_i_dd
    errors_i_dd = {}
    for k, v in preds_i_dd.items():
        errors_i_dd[k] = [target_i - x for x in v]

    # Compute max errors
    error_max = np.max([np.abs(x) for x in errors_i_dd.values()])
    error_min = 0.0

    if names_ordered_lst is None:
        names_ordered_lst = sorted(preds_i_dd.keys())

    fig = plt.figure()
    fig.set_size_inches(5 * (2 * n_samples), 5 * (1 + n_methods))

    gs = GridSpec(
        1 + n_methods, 2 * n_samples, top=TOP, bottom=BOTTOM, left=LEFT, right=RIGHT
    )

    ################################################################################
    # FIRST ROW: Ground-truth, FYNet preds and errors.
    for i in range(n_samples):
        ax_gt = fig.add_subplot(gs[0, 2 * i])

        im_1 = ax_gt.imshow(target_i[i], extent=EXTENT, aspect=ASPECT)
        im_1.set_clim(q_min, q_max)
        cb = plt.colorbar(im_1, ax=ax_gt)
        # increase the font size of the ticks and colorbar
        cb.ax.tick_params(labelsize=BIGTICK_SIZE)
        ax_gt.tick_params(axis="both", which="major", labelsize=BIGTICK_SIZE)
        if i == 0:
            ax_gt.set_ylabel("Ground-Truth", size=TITLE_SIZE)

    ################################################################################
    # OTHER ROWS: Predictions and errors for different methods. One method on each
    # row.

    for i in range(n_samples):
        target = target_i[i]
        for j in range(n_methods):
            method_name = names_ordered_lst[j]
            # print(f"Method name {method_name} and preds shape {preds_i_dd[method_name].shape}")
            method_preds = preds_i_dd[method_name][0][i]
            ax_preds = fig.add_subplot(gs[j + 1, 2 * i])
            ax_errors = fig.add_subplot(gs[j + 1, 2 * i + 1])
            _fill_ax_for_paper(
                [ax_preds, ax_errors],
                pred=method_preds,
                target=target,
                q_max=q_max,
                q_min=q_min,
                err_max=error_max,
                err_min=error_min,
                method_name=None,
                EXTENT=EXTENT,
                ASPECT=ASPECT,
                LABEL_SIZE=LABEL_SIZE,
                TITLE_SIZE=TITLE_SIZE,
                TICK_SIZE=BIGTICK_SIZE,
            )

            if i == 0:
                ax_preds.set_ylabel(method_name, size=TITLE_SIZE)

    plt.savefig(save_fp_format.format(0))
    plt.show()
    plt.clf()


def plot_diagonals(
    methods_dd: Dict, save_fp: str, sample_idx: int, names_ordered_lst: List[str] = None
) -> None:

    xvals = np.linspace(-0.5, 0.5, 192, endpoint=False)
    n_methods = len(methods_dd)

    q_gt = load_q_cart_from_dir(methods_dd["Ground Truth"])

    fig, ax = plt.subplots(1, 2)
    fig.set_size_inches(10, 5)

    if names_ordered_lst is None:
        names_ordered_lst = sorted(methods_dd.keys())

    for name in names_ordered_lst:
        if name == "Ground Truth":
            pred = q_gt[sample_idx]
        else:
            preds = load_q_cart_from_dir(methods_dd[name])
            pred = preds[sample_idx]

            diffs = pred - q_gt[sample_idx]
            ax[1].plot(xvals, np.diag(diffs), label=name)

        ax[0].plot(xvals, np.diag(pred), label=name)

    ax[0].legend()
    ax[1].legend()

    plt.show()


# (2024-12-11, OOT): New helper function for plotting groups at a time
def plot_row(
    obj_list: List[np.ndarray],
    title_list: List[np.ndarray],
    cmap_group_lens: List[int],
    group_cmaps: List[str],
    group_vmaxes: List[float] = None,
    group_vmins: List[float] = None,
    subplot_width: float = 1,
    subplot_height: float = 1,
    extra_fig_width: float = 0,
    extra_fig_height: float = 0,
    plt_subplots_kw: dict = dict(),
    fig: matplotlib.figure.Figure = None,
    axes: matplotlib.axes.Axes = None,
) -> Tuple:
    """Flexible helper function to plot a row of subplots
    Flexible in the sense that you can specify different groups to handle
    the colormap scaling. Within each group, the colormap min/max/coloring is
    shared, and a color bar is placed at the end of the group.

    Optionally pass in fig,axes if you want to fill in a row in a larger plot...

    Note: each group is contiguous
    """
    num_objs = len(obj_list)
    cmap_group_idcs = np.cumsum([0] + cmap_group_lens)
    num_cmap_groups = len(cmap_group_lens)
    vmin_grouped = [
        min(obj.min() for obj in 
            obj_list[cmap_group_idcs[gi]:cmap_group_idcs[gi+1]])
        for gi in range(num_cmap_groups)
    ]
    vmax_grouped = [
        max(obj.max() for obj in
            obj_list[cmap_group_idcs[gi]:cmap_group_idcs[gi+1]])
        for gi in range(num_cmap_groups)
    ]
    if group_vmaxes is not None:
        for gi, gvmax in enumerate(group_vmaxes):
            if gvmax is not None:
                vmax_grouped[gi] = gvmax
    if group_vmins is not None:
        for gi, gvmin in enumerate(group_vmins):
            if gvmin is not None:
                vmin_grouped[gi] = gvmin
    if fig is None or axes is None:
        fig, axes = plt.subplots(
            1, num_objs,
            figsize=(
                subplot_width*(num_objs+0.2*num_cmap_groups) + extra_fig_width,
                subplot_height+extra_fig_height
            ),
            **plt_subplots_kw,
        )

    group_start = 0
    group_idx   = 0 # index corresponding to the group
    for i in range(num_objs):
        ax = axes[i]
        if (i - group_start) == cmap_group_idcs[group_idx+1]:
            group_idx += 1
        # print(f"i={i}, group_idx={group_idx}")
        if title_list is not None:
            ax.set_title(f"{title_list[i]}")
        cb = ax.imshow(
            obj_list[i],
            vmin=vmin_grouped[group_idx],
            vmax=vmax_grouped[group_idx],
            cmap=group_cmaps[group_idx]
        )
        ax.set_xticks(ticks=[])
        ax.set_yticks(ticks=[])
        if (i - group_start + 1) == cmap_group_idcs[group_idx+1]:
            group_axes = axes[cmap_group_idcs[group_idx]:cmap_group_idcs[group_idx+1]]
            plt.colorbar(cb, ax=group_axes.ravel().tolist())

    return fig, axes
