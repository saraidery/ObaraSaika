import numpy as np

from obara_saika.angular_momentum import get_n_cartesian

class ShellData:

    def __init__(self, exp, coeff, l, center, start, stop):
        self.exp = np.array(exp)
        self.coeff = np.array(coeff)
        self.l = l
        self.center = np.array(center)
        self.start = start
        self.stop = stop


class ShellDataPWGTO:

    def __init__(self, exp, coeff, l, k, center, start, stop):
        self.exp = np.array(exp)
        self.coeff = np.array(coeff)
        self.l = l
        self.k = k
        self.center = np.array(center)
        self.start = start
        self.stop = stop

class PointCharge:

    def __init__(self, center, charge):
        self.center = center
        self.charge = charge

class QchemBasis:
    def __init__(self, filename):

        self.n_shells, self.exponents, self.angular_momentum, self.coefficients, self.center_index, self.centers = self.read_qchem_basis(filename)
        self._current_index = 0

        self.sh_dim = 0
        self.offsets = np.zeros(self.n_shells + 1, dtype=int)

        for i, l in enumerate(self.angular_momentum):

            self.offsets[i] = self.sh_dim
            self.sh_dim += get_n_cartesian(l)

        self.offsets[self.n_shells] = self.sh_dim


    def __iter__(self):

        n = 0
        while n < self.n_shells:

            center = self.centers[self.center_index[n]]
            exp = self.exponents[n]
            coeff = self.coefficients[n]
            l = self.angular_momentum[n]
            start = self.offsets[n]
            stop = self.offsets[n + 1]

            shell_data = ShellData(exp, coeff, l, center, start, stop)

            n += 1
            yield shell_data

    def read_qchem_basis(self, filename):

        file = open(filename, 'r')
        lines = file.readlines()

        lines = [i for i in lines if i != "\n"]
        lines = [i.strip("\n") for i in lines]

        offset = 0

        # first line
        current_line = lines[offset]
        offset += 1

        n_centers = int(current_line.split(" ")[0])
        n_exponents = int(current_line.split(" ")[1])
        n_coefficients = int(current_line.split(" ")[2])
        n_shells = int(current_line.split(" ")[3])

        centers      = np.zeros(n_centers);
        exponents    = np.zeros(n_exponents);
        coefficients = np.zeros(n_coefficients);
        shells       = np.zeros(n_shells);

        centers = np.zeros([n_centers, 3])
        for i in np.arange(n_centers):

            current_line = lines[i + offset].strip()
            coordinates = [i for i in current_line.split(" ") if i != ""]

            centers[i, 0] = float(coordinates[0])
            centers[i, 1] = float(coordinates[1])
            centers[i, 2] = float(coordinates[2])

        offset += n_centers

        for i in np.arange(n_exponents):
            current_line = lines[offset + i].strip()
            exponents[i] = float(current_line)

        offset += n_exponents

        for i in np.arange(n_coefficients):
            current_line = lines[offset + i].strip()
            coefficients[i] = float(current_line)

        offset += n_coefficients

        exponent_offset = np.zeros(n_shells, dtype=int)
        shared_exponents = np.zeros(n_shells, dtype=int)
        degree_of_contraction = np.zeros(n_shells, dtype=int)

        coefficient_offset = []
        angular_momentum = []
        center_index = []

        for i in np.arange(n_shells):
            current_line = lines[offset].strip()
            line_1 = [i for i in current_line.split(" ") if i != ""]

            exponent_offset[i] = int(line_1[3])
            shared_exponents[i] = int(line_1[4])

            current_line = lines[offset + 1].strip()
            offsets = [i for i in current_line.split(" ") if i != ""]

            for j in np.arange(shared_exponents[i]):
                center_index.append(int(line_1[0]))
                coefficient_offset.append(int(offsets[j*2]))
                angular_momentum.append(int(offsets[j*2+1]))

            current_line = lines[offset + 2].strip()
            offsets = [i for i in current_line.split(" ") if i != ""]

            degree_of_contraction[i] = int(offsets[0])

            offset += 3

        exponents_expanded = []
        exponents_offset_expanded = []
        coefficients_expanded = []

        count_shells = 0
        for i in np.arange(n_shells):
            for j in np.arange(shared_exponents[i]):
                exponents_expanded.append(exponents[exponent_offset[i]:exponent_offset[i]+degree_of_contraction[i]])
                coefficients_expanded.append(coefficients[coefficient_offset[count_shells]:coefficient_offset[count_shells]+degree_of_contraction[i]])

                count_shells += 1

        n_shells = count_shells

        return n_shells, exponents_expanded, angular_momentum, coefficients_expanded, center_index, centers

    @property
    def n_aos(self):

        n_ao = 0
        for i in np.arange(self.n_shells):
            l = self.angular_momentum[i]

            for x in range(l, -1, -1):
                for y in (range(l - x, -1, -1)):
                    n_ao += 1
        return n_ao


class QchemBasisPWGTO(QchemBasis):

    def __init__(self, filename, k):

        super().__init__(filename)
        self.k = k


    def __iter__(self):

        n = 0
        while n < self.n_shells:

            center = self.centers[self.center_index[n]]
            exp = self.exponents[n]
            coeff = self.coefficients[n]
            l = self.angular_momentum[n]
            k = self.k
            start = self.offsets[n]
            stop = self.offsets[n + 1]

            shell_data = ShellDataPWGTO(exp, coeff, l, k, center, start, stop)

            n += 1
            yield shell_data

