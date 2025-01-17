import numpy as np
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.neighborlist import NeighborList
import matplotlib.pyplot as plt


def read_n2p2(filename='output.data', index=':', with_energy_and_forces='auto'):
    fd = open(filename, 'r')  # @reader decorator ensures this is a file descriptor???
    images = list()
    lineindexlist = []
    lineindex = 0
    line = fd.readline()
    lineindex += 1
    while 'begin' in line:
        lineindexlist.append(lineindex)
        line = fd.readline()
        lineindex += 1
        if 'comment' in line:
            comment = line[7:]
            line = fd.readline()
            lineindex += 1

        cell = np.zeros((3, 3))
        for ii in range(3):
            cell[ii] = [float(jj) for jj in line.split()[1:4]]
            line = fd.readline()
            lineindex += 1

        positions = []
        symbols = []
        charges = []  # not used yet
        nn = []  # not used
        forces = []
        energy = 0.0
        charge = 0.0

        while 'atom' in line:
            sline = line.split()
            positions.append([float(pos) for pos in sline[1:4]])
            symbols.append(sline[4])
            nn.append(float(sline[5]))
            charges.append(float(sline[6]))
            forces.append([float(pos) for pos in sline[7:10]])
            line = fd.readline()
            lineindex += 1

        while 'end' not in line:
            if 'energy' in line:
                energy = float(line.split()[-1])
            if 'charge' in line:
                charge = float(line.split()[-1])
            line = fd.readline()
            lineindex += 1

        image = Atoms(symbols=symbols, positions=positions, cell=cell)

        sorted_indices = np.argsort(image.numbers)
        image = image[sorted_indices]
        store_energy_and_forces = False
        if with_energy_and_forces == True:
            store_energy_and_forces = True
        elif with_energy_and_forces == 'auto':
            if energy != 0.0 or np.absolute(forces).sum() > 1e-8:
                store_energy_and_forces = True

        if store_energy_and_forces:
            image.calc = SinglePointCalculator(
                atoms=image,
                energy=energy,
                forces=forces,
                charges=charges)
            # charge  = charge)
        images.append(image)
        # to start the next section
        line = fd.readline()
        lineindex += 1

    if index == ':' or index is None:
        return images
    else:
        return images[index]


class Cutoff:
    """
    A class for computing cutoff functions (including derivatives) for
    atom-centered descriptors. The existing cutoff functions in PyXtal_FF
    can be found in:
        Singraber, A. (2019). J. Chem. Theory Comput., 15, 1827-1840.

    Parameters
    ----------
    function: str
        The type of cutoff function:
            1. cosine
                f(x) = 0.5*(cos(pi*x)+1)
            2. tanh
                f(x) = (tanh(1-x))**3
            3. exponent
                f(x) = exp(1-(1/(1-x**2)))
            4. poly1
                f(x) = x**2(2*x-3)+1
            5. poly2
                f(x) = x**3(x(15-6*x)-10)+1
            6. poly3
                f(x) = x**4(x(x(20*x-70)+84)-35)+1
            7. poly4
                f(x) = x**5(x(x(x(315-70*x)-540)+420)-126)+1

        where x = R_ij/R_c
    """

    def __init__(self, function):
        self.function = function

    def calculate(self, R, Rc):
        if self.function == 'cosine':
            cutoff = Cosine(R, Rc)
        elif self.function == 'tanh':
            cutoff = Tanh(R, Rc)
        elif self.function == 'poly1':
            cutoff = Poly1(R, Rc)
        elif self.function == 'poly2':
            cutoff = Poly2(R, Rc)
        elif self.function == 'poly3':
            cutoff = Poly3(R, Rc)
        elif self.function == 'poly4':
            cutoff = Poly4(R, Rc)
        elif self.function == 'exponent':
            cutoff = Exponent(R, Rc)
        else:
            msg = f"The {self.function} function is not implemented."
            raise NotImplementedError(msg)
        return cutoff

    def calculate_derivative(self, R, Rc):
        if self.function == 'cosine':
            cutoff_prime = CosinePrime(R, Rc)
        elif self.function == 'tanh':
            cutoff_prime = TanhPrime(R, Rc)
        elif self.function == 'poly1':
            cutoff_prime = Poly1Prime(R, Rc)
        elif self.function == 'poly2':
            cutoff_prime = Poly2Prime(R, Rc)
        elif self.function == 'poly3':
            cutoff_prime = Poly3Prime(R, Rc)
        elif self.function == 'poly4':
            cutoff_prime = Poly4Prime(R, Rc)
        elif self.function == 'exponent':
            cutoff_prime = ExponentPrime(R, Rc)
        else:
            msg = f"The {self.function} function is not implemented."
            raise NotImplementedError(msg)
        return cutoff_prime


def Cosine(Rij, Rc):
    # Rij is the norm
    ids = (Rij > Rc)
    result = 0.5 * (np.cos(np.pi * Rij / Rc) + 1.)
    result[ids] = 0
    return result


def CosinePrime(Rij, Rc):
    # Rij is the norm
    ids = (Rij > Rc)
    result = -0.5 * np.pi / Rc * np.sin(np.pi * Rij / Rc)
    result[ids] = 0
    return result


def Tanh(Rij, Rc):
    # ids = (Rij > Rc)
    if Rij < Rc:
        result = np.tanh(1 - Rij / Rc) ** 3
    else:
        result = 0
    # result[ids] = 0
    return result


def TanhPrime(Rij, Rc):
    if Rij < Rc:
        tanh_square = np.tanh(1 - Rij / Rc) ** 2
        result = - (3 / Rc) * tanh_square * (1 - tanh_square)
    else:
        result = 0
    return result


def Poly1(Rij, Rc):
    if Rij < Rc:
        x = Rij / Rc
        x_square = x ** 2
        result = x_square * (2 * x - 3) + 1
    else:
        result = 0
    return result


def Poly1Prime(Rij, Rc):
    if Rij < Rc:
        term1 = (6 / Rc ** 2) * Rij
        term2 = Rij / Rc - 1
        result = term1 * term2
    else:
        result = 0
    return result


def Poly2(Rij, Rc):
    if Rij < Rc:
        x = Rij / Rc
        result = x ** 3 * (x * (15 - 6 * x) - 10) + 1
    else:
        result = 0
    return result


def Poly2Prime(Rij, Rc):
    if Rij < Rc:
        x = Rij / Rc
        result = (-30 / Rc) * (x ** 2 * (x - 1) ** 2)
    else:
        result = 0
    return result


def Poly3(Rij, Rc):
    if Rij < Rc:
        x = Rij / Rc
        result = x ** 4 * (x * (x * (20 * x - 70) + 84) - 35) + 1
    else:
        result = 0
    return result


def Poly3Prime(Rij, Rc):
    ids = (Rij > Rc)
    x = Rij / Rc
    result = (140 / Rc) * (x ** 3 * (x - 1) ** 3)
    result[ids] = 0
    return result


def Poly4(Rij, Rc):
    if Rij < Rc:
        x = Rij / Rc
        result = x ** 5 * (x * (x * (x * (315 - 70 * x) - 540) + 420) - 126) + 1
    else:
        result = 0
    return result


def Poly4Prime(Rij, Rc):
    if Rij < Rc:
        x = Rij / Rc
        result = (-630 / Rc) * (x ** 4 * (x - 1) ** 4)
    else:
        result = 0
    return result


def Exponent(Rij, Rc):
    result = np.zeros_like(Rij)
    ids = (Rij < Rc)
    x = Rij[ids] / Rc
    result[ids] = np.exp(1 - 1 / (1 - x ** 2))
    # result[ids] = 0
    return result


def ExponentPrime(Rij, Rc):
    ids = (Rij > Rc)
    x = Rij / Rc
    result = -2 * x * np.exp(1 - 1 / (1 - x ** 2)) / (1 - x ** 2) ** 2
    result[ids] = 0
    return result


def get_G9(R1ij0, R1ik0, theta, rs, zetas, lamBdas, etas, Rc, cutoff):
    results = 0
    R2ij = (R1ij0 - rs) ** 2
    R2ik = (R1ik0 - rs) ** 2
    term4 = cutoff.calculate(R1ij0, Rc) * cutoff.calculate(R1ik0, Rc)
    if term4 > 0:
        powers = 2. ** (1. - zetas)
        # cos_ijk = np.dot(rij, rik) / R1ij0 / R1ik0
        cos_ijk = np.cos(np.radians(theta))
        term1 = 1. + lamBdas * cos_ijk
        term2 = np.power(term1, zetas)
        term3 = np.exp(-etas * (R2ij + R2ik))
        results = powers * term2 * term3 * term4
    return results  # ,thetalist


def get_G3(R1ij0, R1ik0, theta, rs, zetas, lamBdas, etas, Rc, cutoff):
    results = 0
    R1jk0 = np.sqrt(R1ij0 ** 2 + R1ik0 ** 2 - 2 * R1ij0 * R1ik0 * np.cos(np.radians(theta)))
    R2ij = (R1ij0 - rs) ** 2
    R2ik = (R1ik0 - rs) ** 2
    R2jk = (R1jk0 - rs) ** 2

    term4 = cutoff.calculate(R1ij0, Rc) * cutoff.calculate(R1ik0, Rc) * cutoff.calculate(R1jk0, Rc)
    if term4 > 0:
        powers = 2. ** (1. - zetas)
        cos_ijk = np.cos(np.radians(theta))
        term1 = 1. + lamBdas * cos_ijk
        term2 = np.power(term1, zetas)
        term3 = np.exp(-etas * (R2ij + R2ik + R2jk))
        results = powers * term2 * term3 * term4

    return results


if __name__ == '__main__':
    # atomslist = read_n2p2('input.data')
    zetas = 6
    lamBdas = -1
    etas = 0.007
    rs = 0
    Rc = 6.
#   rc=2
    cutoff = Cutoff('tanh')
    resultslist1 = []
    thetalist1 = []
#    resultslist2 = []
#    thetalist2 = []

    for R1ij0 in np.arange(0.1, 6, 0.5):
        for R1ik0 in np.arange(0.1, 6, 0.5):
            for theta in np.arange(0, 361, 1):
                resultslist = get_G3(R1ij0, R1ik0, theta, rs, zetas, lamBdas, etas, Rc, cutoff)
                resultslist1.append(resultslist)
                thetalist1.append(theta)

    # for R1ij0 in np.arange(0.1, 6, 0.5):
    #     for R1ik0 in np.arange(0.1, 6, 0.5):
    #         for theta in np.arange(0, 361, 1):
    #             resultslist = get_G3(R1ij0, R1ik0, theta, rs, zetas, lamBdas, etas, rc, cutoff)
    #             resultslist2.append(resultslist)
    #             thetalist2.append(theta)

    # resultsss=np.array([item for sublist in resultslist1 for item in sublist])
    # thetalistsss=np.array([item for sublist in thetalist1 for item in sublist])

    # 绘制极坐标图
    plt.figure()
    ax = plt.subplot(111, projection='polar')
    # ax.plot(thetalistsss, resultsss)
    thetalist_radians = np.radians(thetalist1)
#    thetalist_radians2 = np.radians(thetalist2)
    ax.plot(thetalist_radians, resultslist1)
    # ax.plot(thetalist_radians2, resultslist2,color='red')
    # 显示极坐标图
    plt.show()
