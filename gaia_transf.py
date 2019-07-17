
from os.path import join, isfile
from os.path import exists
from os import makedirs, listdir
from pathlib import Path
import logging

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from astropy.table import Table
import numpy as np
from scipy.spatial.distance import cdist


def main():
    """
    """

    # params_input()
    # Maximum error values for own UBVI photometry.
    eVmax, eBVmax, eVImax, V_id, eV_id, BV_id, eBV_id, VI_id, eVI_id, G_id,\
        eG_id, BPRP_id = params_input()

    # Generate output dir if it doesn't exist.
    if not exists('out'):
        makedirs('out')

    # Process all files inside 'in/' folder.
    clusters = get_files()
    if not clusters:
        print("No input cluster files found")

    rpath = Path().absolute()
    for cluster in clusters:
        # Extract name of file without extension
        cl_name = cluster[3:-4]

        # Set up logging module
        level = logging.INFO
        frmt = ' %(message)s'
        handlers = [
            logging.FileHandler(
                join(rpath, 'out', cl_name + '.log'), mode='w'),
            logging.StreamHandler()]
        logging.basicConfig(level=level, format=frmt, handlers=handlers)

        logging.info("\nProcessing: {}...".format(cl_name))

        # Read cluster photometry.
        logging.info("\nRead matched final photometry")
        t = Table.read(join(rpath, cluster), format='ascii')
        Vm_all, eVm, BVm_all, eBVm, VIm_all, eVIm, Gm_all, eGm, BPRPm_all =\
            t[V_id], t[eV_id], t[BV_id], t[eBV_id], t[VI_id], t[eVI_id],\
            t[G_id], t[eG_id], t[BPRP_id]

        logging.info("\nApply transformations")
        Vm, BVm, VIm, Gm, BPRPm, x1, y1, x2, y2, x3, y3, delta1, delta2,\
            delta3 = transfPlot(
                Gm_all, eGm, Vm_all, eVm, BVm_all, eBVm, VIm_all, eVIm,
                BPRPm_all, eVmax, eBVmax, eVImax)

        logging.info("\nPlot")
        makePlots(
            cl_name, Vm_all, BVm_all, VIm_all, Gm_all, BPRPm_all, Vm, eVm, BVm,
            eBVm, VIm, eVIm, Gm, eGm, BPRPm, x1, y1, x2, y2, x3, y3, delta1,
            delta2, delta3)

    logging.info("\nEnd")


def params_input():
    """
    Read input parameters from 'params_input.dat' file.
    """
    with open('params_input.dat', "r") as f_dat:
        # Iterate through each line in the file.
        for l, line in enumerate(f_dat):
            if not line.startswith("#") and line.strip() != '':
                reader = line.split()
                if reader[0] == 'EM':
                    eVmax, eBVmax, eVImax = list(map(float, reader[1:]))
                if reader[0] == 'CM':
                    V_id, eV_id, BV_id, eBV_id, VI_id, eVI_id, G_id, eG_id,\
                        BPRP_id = reader[1:]

    return eVmax, eBVmax, eVImax, V_id, eV_id, BV_id, eBV_id, VI_id, eVI_id,\
        G_id, eG_id, BPRP_id


def get_files():
    '''
    Store the paths and names of all the input clusters stored in the
    input folder.
    '''

    cl_files = []
    for f in listdir('in/'):
        if isfile(join('in/', f)) and f.endswith('.dat'):
            cl_files.append(join('in/', f))

    return cl_files


def PolyCoefficients(x, coeffs):
    """
    Returns a polynomial for ``x`` values for the ``coeffs`` provided.
    The coefficients must be in ascending order (``x**0`` to ``x**p``).
    """
    y = 0
    for p in range(len(coeffs)):
        y += coeffs[p] * x**p
    return y


def transfPlot(
    Gm, eGm, Vm, eVm, BVm, eBVm, VIm, eVIm, BPRPm, eVmax, eBVmax,
        eVImax):
    """
    Carrasco Gaia DR2 transformations:
    https://gea.esac.esa.int/archive/documentation/GDR2/Data_processing/
    chap_cu5pho/sec_cu5pho_calibr/ssec_cu5pho_PhotTransf.html
    """

    # Apply masks
    Gmsk = Gm < 13.
    eGmsk = eGm < 0.01
    BVmsk = (-0.3 < BVm) & (BVm < 2.4)
    VImsk = (-.3 < VIm) & (VIm < 2.7)
    BPRPmsk = (-.5 < BPRPm) & (BPRPm < 2.75)

    # Filters on our photometry
    eVmsk = eVm < eVmax
    eBVmsk = eBVm < eBVmax
    eVImsk = eVIm < eVImax
    msk = Gmsk & eGmsk & BVmsk & VImsk & eVmsk & eBVmsk & eVImsk & BPRPmsk
    Vm, BVm, VIm, Gm, BPRPm = Vm[msk], BVm[msk], VIm[msk],\
        Gm[msk], BPRPm[msk]

    Ninterp = 1000

    # G-V vs B-V
    x1 = np.linspace(min(BVm), max(BVm), Ninterp)
    coeffs = [-0.02907, -0.02385, -0.2297, -0.001768]
    y1 = PolyCoefficients(x1, coeffs)
    # Range of applicability
    delta1 = np.median(np.min(cdist(
        np.array([BVm, Gm - Vm]).T, np.array([x1, y1]).T), axis=1))
    logging.info("GV vs BV Delta_median: {:.4f}".format(delta1))

    # G-V vs V-I
    x2 = np.linspace(np.nanmin(VIm), np.nanmax(VIm), Ninterp)
    coeffs = [-0.01746, 0.008092, -0.2810, 0.03655]
    y2 = PolyCoefficients(x2, coeffs)
    delta2 = np.median(np.min(cdist(
        np.array([VIm, Gm - Vm]).T, np.array([x2, y2]).T), axis=1))
    logging.info("GV vs VI Delta_median: {:.4f}".format(delta2))

    # G-V vs G_BP-G_RP
    x3 = np.linspace(min(BPRPm), max(BPRPm), Ninterp)
    coeffs = [-0.01760, -0.006860, -0.1732]
    y3 = PolyCoefficients(x3, coeffs)
    delta3 = np.median(np.min(cdist(
        np.array([BPRPm, Gm - Vm]).T, np.array([x3, y3]).T), axis=1))
    logging.info("GV vs BPRP Delta_median: {:.4f}".format(delta3))

    return Vm, BVm, VIm, Gm, BPRPm, x1, y1, x2, y2, x3, y3, delta1, delta2,\
        delta3


def makePlots(
    name, Vm_all, BVm_all, VIm_all, Gm_all, BPRPm_all, Vm, eVm, BVm, eBVm,
    VIm, eVIm, Gm, eGm, BPRPm, x1, y1, x2, y2, x3, y3, delta1, delta2,
        delta3):
    """
    """
    dpi = 150
    fig = plt.figure(figsize=(20, 20))
    gs = gridspec.GridSpec(3, 3)

    ymin, ymax = min(Gm) - .5, max(Gm) + .5

    ax = plt.subplot(gs[3])
    ax.grid(which='major', axis='both', linestyle='--', color='grey', lw=.5)
    plt.xlabel(r"$B-V$", fontsize=14)
    plt.ylabel(r"$G$", fontsize=14)
    plt.scatter(BVm, Gm, label="N={}".format(len(Vm)), c=Vm, zorder=4)
    plt.scatter(
        BVm_all, Gm_all, label="N={}".format(len(Vm_all)), c='grey',
        zorder=0)
    plt.xlim(max(-.4, min(BVm) - .05), min(2.5, max(BVm) + .05))
    plt.ylim(ymin, ymax)
    plt.gca().invert_yaxis()
    plt.legend(fontsize=12)

    ax = plt.subplot(gs[4])
    ax.grid(which='major', axis='both', linestyle='--', color='grey', lw=.5)
    plt.xlabel(r"$V-I$", fontsize=14)
    plt.ylabel(r"$G$", fontsize=14)
    plt.scatter(VIm, Gm, label="N={}".format(len(Vm)), c=Vm, zorder=4)
    plt.scatter(VIm_all, Gm_all, c='grey', zorder=0)
    plt.xlim(max(-.4, min(VIm) - .05), min(2.5, max(VIm) + .05))
    plt.ylim(ymin, ymax)
    plt.gca().invert_yaxis()

    ax = plt.subplot(gs[5])
    ax.grid(which='major', axis='both', linestyle='--', color='grey', lw=.5)
    plt.xlabel(r"$BP-RP$", fontsize=14)
    plt.ylabel(r"$G$", fontsize=14)
    plt.scatter(BPRPm, Gm, label="N={}".format(len(Vm)), c=Vm, zorder=4)
    plt.scatter(BPRPm_all, Gm_all, c='grey', zorder=0)
    plt.xlim(max(-.4, min(BPRPm) - .05), min(2.5, max(BPRPm) + .05))
    plt.ylim(ymin, ymax)
    plt.gca().invert_yaxis()

    ax = plt.subplot(gs[6])
    ax.grid(which='major', axis='both', linestyle='--', color='grey', lw=.5)
    plt.title(
        r"GDR2 (G<13, $\sigma_{{G}}<0.01$) " +
        r"($\Delta_{{median}}\approx{:.4f}$)".format(delta1), fontsize=14)
    plt.xlabel(r"$B-V$", fontsize=14)
    plt.ylabel(r"$G-V$", fontsize=14)
    plt.plot(x1, y1, c='orange', label="Carrasco transformation", zorder=-1)
    plt.scatter(BVm, Gm - Vm, label="N={}".format(len(Vm)), c=Vm)
    plt.xlim(max(-.4, min(BVm) - .05), min(2.5, max(BVm) + .05))
    plt.legend(fontsize=12)

    ax = plt.subplot(gs[7])
    ax.grid(which='major', axis='both', linestyle='--', color='grey', lw=.5)
    plt.title(
        r"GDR2 (G<13, $\sigma_{{G}}<0.01$) " +
        r"($\Delta_{{median}}\approx{:.4f}$)".format(delta2), fontsize=14)
    plt.xlabel(r"$V-I$", fontsize=14)
    plt.ylabel(r"$G-V$", fontsize=14)
    plt.plot(x2, y2, c='orange', zorder=-1)
    plt.scatter(VIm, Gm - Vm, label="N={}".format(len(Vm)), c=Vm)
    plt.xlim(max(-.4, min(VIm) - .05), min(2.8, max(VIm) + .05))

    ax = plt.subplot(gs[8])
    ax.grid(which='major', axis='both', linestyle='--', color='grey', lw=.5)
    plt.title(
        r"GDR2 (G<13, $\sigma_{{G}}<0.01$) " +
        r"($\Delta_{{median}}\approx{:.4f}$)".format(delta3), fontsize=14)
    plt.xlabel(r"$BP-RP$", fontsize=14)
    plt.ylabel(r"$G-V$", fontsize=14)
    plt.plot(x3, y3, c='orange', zorder=-1)
    # cm = plt.cm.get_cmap('Reds')
    sc = plt.scatter(BPRPm, Gm - Vm, label="N={}".format(len(Vm)), c=Vm)
    plt.xlim(max(-.6, min(BPRPm) - .05), min(2.8, max(BPRPm) + .05))

    # create an axes on the right side of ax. The width of cax will be 5%
    # of ax and the padding between cax and ax will be fixed at 0.05 inch.
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="2%", pad=0.02)
    cbar = plt.colorbar(sc, cax=cax)
    # cbar = plt.colorbar(sc)
    cbar.ax.tick_params(labelsize=12)
    cbar.ax.invert_yaxis()
    cbar.set_label('V [mag]', fontsize=12)

    fig.tight_layout()
    fig.savefig(
        join('out', name + '_carrasco.png'), dpi=dpi, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    main()
