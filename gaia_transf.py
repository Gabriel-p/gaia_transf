
from os.path import join, isfile
from os.path import exists
from os import makedirs, listdir
from pathlib import Path
import logging
from datetime import datetime
from astropy.table import Table
from modules import transf
from modules import make_plots


def main(CarrascoStandards=False):
    """
    Compare UBVI photometry with Carrasco's transformations.

    CarrascoStandards: bool (hidden parameter)
    If True, it will use the list of Landolt standards provided by Carrasco.
    """

    # Set up logging module
    rpath = Path().absolute()
    level = logging.INFO
    frmt = ' %(message)s'
    handlers = [
        logging.FileHandler(join(rpath, 'out/gaia_transf.log'), mode='w'),
        logging.StreamHandler()]
    logging.basicConfig(level=level, format=frmt, handlers=handlers)

    logging.info(datetime.now())

    if CarrascoStandards:
        data_dict = loadCarrasco()
        transf_dict = transf.main(data_dict)
        make_plots.main('carrasco', data_dict, transf_dict)
        return

    # Input parameters
    eVmax, eUBmax, eBVmax, eVImax, V_id, eV_id, UB_id, eUB_id, BV_id, eBV_id,\
        VI_id, eVI_id, max_delta, plotAll = params_input()

    # Generate output dir if it doesn't exist.
    if not exists('out'):
        makedirs('out')

    # Process all files inside 'in/' folder.
    clusters = get_files()
    if not clusters:
        logging.info("No input cluster files found")

    data_dict_all = {
        'Gm': [], 'BPRPm': [], 'Um': [], 'Bm': [], 'Vm': [], 'Im': [],
        'UBm': [], 'BVm': [], 'VIm': []}
    for cluster in clusters:
        # Extract name of file without extension
        cl_name = cluster[3:-4]

        logging.info("\nProcessing: {}...".format(cl_name))

        # Read cluster photometry.
        logging.info("Read matched final photometry")
        t = Table.read(join(rpath, cluster), format='ascii',
                       fill_values=[('', '0'), ('NA', '0'), ('INDEF', '0')])
        Vm, eVm, UBm_all, eUBm, BVm_all, eBVm, VIm_all, eVIm, Gm, eGm,\
            BPRPm, e_BP, e_RP, eEBPRP = t[V_id], t[eV_id], t[UB_id],\
            t[eUB_id], t[BV_id], t[eBV_id], t[VI_id], t[eVI_id], t['Gmag'],\
            t['e_Gmag'], t['BP-RP'], t['e_BPmag'], t['e_RPmag'], t['E_BR_RP_']

        # Carrasco filters
        logging.info("Apply filters on photometry")
        data_dict = carrascoFilter(
            Vm, eVm, UBm_all, eUBm, BVm_all, eBVm, VIm_all, eVIm,
            Gm, eGm, BPRPm, e_BP, e_RP, eEBPRP, eVmax, eUBmax, eBVmax, eVImax)

        if plotAll:
            # Store for plotting of all the combined data
            for k, v in data_dict.items():
                data_dict_all[k] += list(v)
            continue
        else:
            logging.info("Apply transformations")
            transf_dict = transf.main(data_dict)

            logging.info("Plot")
            make_plots.main(cl_name, data_dict, transf_dict, max_delta)

    if plotAll:
        transf_dict = transf.main(data_dict_all)
        make_plots.main('all', data_dict_all, transf_dict, max_delta)

    print("\nFinished")


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
                    eVmax, eUBmax, eBVmax, eVImax = list(map(
                        float, reader[1:]))
                if reader[0] == 'CM':
                    V_id, eV_id, UB_id, eUB_id, BV_id, eBV_id, VI_id, eVI_id =\
                        reader[1:]
                if reader[0] == 'MP':
                    max_delta = float(reader[1])
                if reader[0] == 'PA':
                    plotAll = True if reader[1] == 'True' else False

    return eVmax, eUBmax, eBVmax, eVImax, V_id, eV_id, UB_id, eUB_id, BV_id,\
        eBV_id, VI_id, eVI_id, max_delta, plotAll


def loadCarrasco():
    """
    Load Carrasco's list of cross-matched Landolt standards.
    """
    t = Table.read(
        'modules/DR2G13xLandolt09-13.dat', format='ascii',
        fill_values=[('', '0'), ('NA', '0'), ('INDEF', '0')])

    G, eG, V, UB, BV, VR, RI, BP, e_BP, RP, e_RP =\
        t['phot_g_mean_mag'], t['phot_g_mean_mag_error'], t['Vmag'],\
        t['U-B'], t['B-V'], t['V-R'], t['R-I'], t['phot_bp_mean_mag'],\
        t['phot_bp_mean_mag_error'], t['phot_rp_mean_mag'],\
        t['phot_rp_mean_mag_error']

    B = BV + V
    VI = VR + RI
    U, Imag = UB + B, V - VI
    BPRP = BP - RP
    Rmag = RI + Imag

    # General mask
    msk = (G < 13.) & (eG < 0.01) & (e_BP < 0.01) & (e_RP < 0.01)
    Gm, Um, Bm, Vm, Im, Rm, BPRPm, UBm, BVm, VIm = [
        _[msk] for _ in (G, U, B, V, Imag, Rmag, BPRP, UB, BV, VI)]

    data_dict = {
        'Gm': Gm, 'BPRPm': BPRPm, 'Um': Um, 'Bm': Bm, 'Vm': Vm, 'Im': Im,
        'UBm': UBm, 'BVm': BVm, 'VIm': VIm}

    return data_dict


def get_files():
    """
    Store the paths and names of all the input clusters stored in the
    input folder.
    """

    cl_files = []
    for f in listdir('in/'):
        if isfile(join('in/', f)) and f.endswith('.dat'):
            cl_files.append(join('in/', f))

    return cl_files


def carrascoFilter(
    Vm, eVm, UBm, eUBm, BVm, eBVm, VIm, eVIm, Gm, eGm, BPRPm, e_BP, e_RP,
        eEBPRP, eVmax, eUBmax, eBVmax, eVImax):
    """
    """
    # Apply mask on our photometry
    msk_o = (eVm < eVmax) & (eUBm < eUBmax) & (eBVm < eBVmax) & (eVIm < eVImax)

    # Carrasco mask
    msk_G, msk_eG, msk_eEBPRP = Gm < 13., eGm < 0.01,\
        eEBPRP < 1.5 + 0.03 * BPRPm**2
    msk_BPRP = (e_BP < 0.01) & (e_RP < 0.01)

    # Full mask
    msk = msk_o & msk_G & msk_eG & msk_eEBPRP & msk_BPRP

    Bm = BVm + Vm
    Um = UBm + Bm
    Im = Vm - VIm

    data_dict = {
        'Gm': Gm[msk], 'BPRPm': BPRPm[msk], 'Um': Um[msk], 'Bm': Bm[msk],
        'Vm': Vm[msk], 'Im': Im[msk], 'UBm': UBm[msk], 'BVm': BVm[msk],
        'VIm': VIm[msk]}

    return data_dict


if __name__ == '__main__':
    main()
