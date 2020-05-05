
from os.path import join
import logging
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable


def main(name, data, transf, max_delta):
    """
    """
    Gm, BPRPm, Um, Bm, Vm, Im, UB, BV, VI = [np.array(data[_]) for _ in (
        'Gm', 'BPRPm', 'Um', 'Bm', 'Vm', 'Im', 'UBm', 'BVm', 'VIm')]

    plt.style.use('seaborn')
    plt.set_cmap('viridis')
    fig = plt.figure(figsize=(25, 25))
    gs = gridspec.GridSpec(4, 4)

    def polyPlot(gs_ax, mag, mag_name):
        plt.subplot(gs[gs_ax])
        plt.xlabel(r"$BP-RP$", fontsize=12)
        plt.ylabel(r"$G-{}$".format(mag_name), fontsize=12)
        y_poly = transf[mag_name + '_poly']
        plt.plot(transf['x'], y_poly, c='k', zorder=5)
        plt.scatter(BPRPm, Gm - mag, label="N={}".format(len(Gm)), c=Vm)
        plt.xlim(max(-.4, min(BPRPm) - .05), min(2.8, max(BPRPm) + .05))
        plt.ylim(min(y_poly) - .1, max(y_poly) + .1)

    # G-X vs BP-RP
    polyPlot(0, Um, 'U')
    polyPlot(1, Bm, 'B')
    polyPlot(2, Vm, 'V')
    polyPlot(3, Im, 'I')

    def magDiffs(gs_ax, mag, mag_name, BPRPm=BPRPm):
        plt.subplot(gs[gs_ax])
        plt.xlabel(r"${}$".format(mag_name), fontsize=12)  # _{{Gaia}}
        plt.ylabel(
            r"${}_{{Gaia}} - {}$".format(mag_name, mag_name), fontsize=12)
        delta_M = transf[mag_name + '_Gaia'] - mag

        msk = (-max_delta < delta_M) & (delta_M < max_delta)
        plt.title(
            "N={}, mask=({}, {})".format(msk.sum(), -max_delta, max_delta),
            fontsize=12)

        delta_M, BPRPm, mag = delta_M[msk], BPRPm[msk], mag[msk]
        mag_mean, mag_std = np.mean(delta_M), np.std(delta_M)
        plt.axhline(
            np.mean(delta_M), c='red', ls='--', lw=1.5, zorder=1,
            label=r"$\Delta {}_{{mean}}=${:.3f}$\pm${:.3f}".format(
                mag_name, mag_mean, mag_std))
        mag_median = np.nanmedian(delta_M)
        plt.axhline(
            y=mag_median, ls=':', c='k',
            label="Median = {:.3f}".format(mag_median))
        plt.scatter(mag, delta_M, c=BPRPm)
        plt.legend(fontsize=12)

        logging.info("Delta {} mean: {:.4f} +/- {:.4f}".format(
            mag_name, mag_mean, mag_std))
        logging.info("Delta {} median : {:.4f}".format(mag_name, mag_median))

    # Delta plots for magnitudes
    magDiffs(4, Um, 'U')
    magDiffs(5, Bm, 'B')
    magDiffs(6, Vm, 'V')
    magDiffs(7, Im, 'I')

    def colDiffs(gs_ax, col_data, col, Vm=Vm, BPRPm=BPRPm):
        ax = plt.subplot(gs[gs_ax])
        col_gaia = transf[col[0] + '_Gaia'] - transf[col[1] + '_Gaia']
        delta_col = col_gaia - col_data

        msk = (-max_delta < delta_col) & (delta_col < max_delta)
        plt.title(
            "N={}, mask=({}, {})".format(msk.sum(), -max_delta, max_delta),
            fontsize=12)
        plt.xlabel(r"$V}$", fontsize=12)
        plt.ylabel(r"${}_{{Gaia}} - {}$".format(col, col), fontsize=12)
        col_mean, col_std = np.mean(delta_col[msk]), np.std(delta_col[msk])
        plt.axhline(
            np.mean(delta_col[msk]), c='red', ls='--', lw=1.5, zorder=1,
            label=r"$\Delta {}_{{mean}}=${:.3f}$\pm${:.3f}".format(
                col, col_mean, col_std))
        col_median = np.nanmedian(delta_col[msk])
        plt.axhline(
            y=np.nanmedian(delta_col[msk]), ls=':', c='k',
            label="Median = {:.3f}".format(col_median))
        im = plt.scatter(Vm[msk], delta_col[msk], c=BPRPm[msk])
        plt.legend(fontsize=12)

        logging.info("Delta {} mean: {:.4f} +/- {:.4f}".format(
            col, col_mean, col_std))
        logging.info("Delta {} median: {:.3f}".format(col, col_median))

        if col == 'BV':
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='2%', pad=0.05)
            cbar = plt.colorbar(im, cax=cax)
            cbar.ax.set_ylabel('(BP-RP)', fontsize=10)
            cbar.ax.tick_params(labelsize=8)

    # Delta plots for colors
    colDiffs(8, UB, 'UB')
    colDiffs(9, BV, 'BV')
    colDiffs(10, VI, 'VI')

    fig.tight_layout()
    fig.savefig(
        join('out/gaia_transf_' + name + '.png'), dpi=150, bbox_inches='tight')
    plt.close()
