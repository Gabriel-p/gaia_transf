
import numpy as np


def main(data, Ninterp=1000):
    """
    Transformations for G-U and G-B are not provided by Carrasco in the link
    below. We obtained those using Carrasco's list of Landolt standards and
    the 'UB_coeffs()' function (below).

    Carrasco Gaia DR2 transformations:
    https://gea.esac.esa.int/archive/documentation/GDR2/Data_processing/
    chap_cu5pho/sec_cu5pho_calibr/ssec_cu5pho_PhotTransf.html
    """
    Gm, BPRPm = data['Gm'], data['BPRPm']

    # UB_coeffs(data['Bm'], Gm, BPRPm)

    x = np.linspace(np.nanmin(BPRPm), np.nanmax(BPRPm), Ninterp)
    transf_dict = {'x': x}

    # G-U vs BP-RP
    U_coeffs = [-0.34971232, 2.26474081, -4.66713317, 2.69990085, -1.45412643,
                0.12850817]
    # [0.30457846, -1.14012079, 0.64847919, -1.5527081, 0.34591795]
    # [0.21240278, -1.01742603, -1.14593491, 0.48686377]
    U_poly = np.polyval(U_coeffs, x)
    U_Gaia = Gm - np.polyval(U_coeffs, BPRPm)
    transf_dict.update({'U_poly': U_poly, 'U_Gaia': U_Gaia})

    # G-B vs BP-RP
    B_coeffs = [0.06719282, -0.42375974, -0.63802333, 0.00282631]
    B_poly = np.polyval(B_coeffs, x)
    B_Gaia = Gm - np.polyval(B_coeffs, BPRPm)
    transf_dict.update({'B_poly': B_poly, 'B_Gaia': B_Gaia})

    # G-V vs BP-RP
    V_coeffs = [-0.1732, -0.006860, -0.01760]
    V_poly = np.polyval(V_coeffs, x)
    V_Gaia = Gm - np.polyval(V_coeffs, BPRPm)
    transf_dict.update({'V_poly': V_poly, 'V_Gaia': V_Gaia})

    # G-I vs BP-RP
    I_coeffs = [-0.09631, 0.7419, 0.02085]
    I_poly = np.polyval(I_coeffs, x)
    I_Gaia = Gm - np.polyval(I_coeffs, BPRPm)
    transf_dict.update({'I_poly': I_poly, 'I_Gaia': I_Gaia})

    return transf_dict


def UB_coeffs(Bm, Gm, BPRPm, poly_order=4):
    """
    """
    from scipy.optimize import least_squares

    def fun(coeffs, yd):
        """
        Obtain transformation coefficients for the U,B,I magnitude as:
        G - X = f(BP-RP)
        """
        v = (Gm - yd) - np.polyval(coeffs, BPRPm)
        return v

    for yd in (Bm,):  # Um, Vm, Rm, Im
        res = least_squares(fun, [.1] * poly_order, args=([yd]))

        # Parameters errors. See:
        # https://stackoverflow.com/a/21844726/1391441
        # https://stackoverflow.com/a/14857441/1391441

        # Modified Jacobian
        J = res.jac
        # Reduced covariance matrix
        red_cov = np.linalg.inv(J.T.dot(J))
        # RMS of the residuals
        RMS = (res.fun**2).mean()
        # Covariance Matrix
        cov = red_cov * RMS
        # Standard deviation of the parameters
        p_std = np.sqrt(np.diagonal(cov))

        print("Coeffs:", np.array(res.x))
        print(p_std)
        print(np.sqrt((res.fun**2).mean()))


# OLD coefficients
#
# def PolyCoefficients(x, coeffs):
#     """
#     Returns a polynomial for ``x`` values for the ``coeffs`` provided.
#     The coefficients must be in ascending order (``x**0`` to ``x**p``).
#     """
#     y = 0
#     for p in range(len(coeffs)):
#         y += coeffs[p] * x**p
#     return y


# def transf(Gm, Vm, BVm, VIm, BPRPm, Ninterp=1000):
#     """
#     Carrasco Gaia DR2 transformations:
#     https://gea.esac.esa.int/archive/documentation/GDR2/Data_processing/
#     chap_cu5pho/sec_cu5pho_calibr/ssec_cu5pho_PhotTransf.html
#     """

#     def colorTransf(coeffs, col):
#         return coeffs[0] + coeffs[1] * col + coeffs[2] * col**2 + coeffs[3] *\
#             col**3

#     # G-V vs B-V
#     x1 = np.linspace(min(BVm), max(BVm), Ninterp)
#     coeffs = [-0.02907, -0.02385, -0.2297, -0.001768]
#     y1 = PolyCoefficients(x1, coeffs)
#     # Range of applicability
#     delta1 = np.median(np.min(cdist(
#         np.array([BVm, Gm - Vm]).T, np.array([x1, y1]).T), axis=1))
#     logging.info("GV vs BV Delta_median: {:.4f}".format(delta1))

#     # G_Gaia vs G_Transf
#     G_BV_trnsf = Vm + colorTransf(coeffs, BVm)
#     GG_BV_mean, GG_BV_median = np.mean(Gm - G_BV_trnsf),\
#         np.median(Gm - G_BV_trnsf)
#     logging.info("BV, G_Gaia-G_Transf Delta_mean: {:.4f}".format(
#         GG_BV_mean))
#     logging.info("BV, G_Gaia-G_Transf Delta_median: {:.4f}".format(
#         GG_BV_median))

#     # G-V vs V-I
#     x2 = np.linspace(np.nanmin(VIm), np.nanmax(VIm), Ninterp)
#     coeffs = [-0.01746, 0.008092, -0.2810, 0.03655]
#     y2 = PolyCoefficients(x2, coeffs)
#     delta2 = np.median(np.min(cdist(
#         np.array([VIm, Gm - Vm]).T, np.array([x2, y2]).T), axis=1))
#     logging.info("GV vs VI Delta_median: {:.4f}".format(delta2))

#     # G_Gaia vs G_Transf
#     G_VI_trnsf = Vm + colorTransf(coeffs, VIm)
#     GG_VI_mean, GG_VI_median = np.mean(Gm - G_VI_trnsf),\
#         np.median(Gm - G_VI_trnsf)
#     logging.info("VI, G_Gaia-G_Transf Delta_mean: {:.4f}".format(
#         GG_VI_mean))
#     logging.info("VI, G_Gaia-G_Transf Delta_median: {:.4f}".format(
#         GG_VI_median))

#     # G-V vs G_BP-G_RP
#     x3 = np.linspace(min(BPRPm), max(BPRPm), Ninterp)
#     coeffs = [-0.01760, -0.006860, -0.1732, 0.]
#     y3 = PolyCoefficients(x3, coeffs)
#     delta3 = np.median(np.min(cdist(
#         np.array([BPRPm, Gm - Vm]).T, np.array([x3, y3]).T), axis=1))
#     logging.info("GV vs BPRP Delta_median: {:.4f}".format(delta3))

#     # G_Gaia vs G_Transf
#     G_BR_trnsf = Vm + colorTransf(coeffs, BPRPm)
#     GG_BR_mean, GG_BR_median = np.mean(Gm - G_BR_trnsf),\
#         np.median(Gm - G_BR_trnsf)
#     logging.info("BR, G_Gaia-G_Transf Delta_mean: {:.4f}".format(
#         GG_BR_mean))
#     logging.info("BR, G_Gaia-G_Transf Delta_median: {:.4f}".format(
#         GG_BR_median))

#     return G_BV_trnsf, GG_BV_mean, GG_BV_median, G_VI_trnsf, GG_VI_mean,\
#         GG_VI_median, G_BR_trnsf, GG_BR_mean, GG_BR_median, x1, y1, x2, y2,\
#         x3, y3, delta1, delta2, delta3
