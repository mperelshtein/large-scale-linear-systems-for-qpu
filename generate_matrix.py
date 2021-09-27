import random
from qiskit import QuantumCircuit
from math import pi
import numpy as np
import scipy as sc


Id = np.eye(2).astype(complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.diag([1, -1]).astype(complex)
H = np.array([[1, 1], [1, -1]], dtype=complex)/np.sqrt(2)
S = np.diag([1, 1j])
Rz = lambda theta: sc.linalg.expm(-1j*theta*Z/2)
U3 = lambda theta, phi, lamda: np.array([
    [np.cos(theta/2), -np.exp(1j*lamda)*np.sin(theta/2)],
    [np.exp(1j*phi)*np.sin(theta/2), np.exp(1j*(lamda + phi))*np.cos(theta/2)]
], dtype=complex)
CNOT = np.kron(Id, np.diag([1, 0])) + np.kron(X, np.diag([0, 1]))


def n_tensor_product(*matrices):
    if len(matrices) == 1:
        return matrices[0]
    else:
        return np.kron(matrices[0], n_tensor_product(*matrices[1:]))


def add_hermitian_single_qubit_gate(circuit, register):
    theta, lamda = random.random() * 2 * pi, random.random() * 2 * pi
    sign = 1 if random.random() > 0.5 else -1
    phi = (sign * pi - lamda) % (2 * pi)
    circuit.u3(theta, phi, lamda, register)

    return theta, phi, lamda


def generate_random_U_circ(registers, circ=None, circ_type='TP1', seed=None):
    if seed != None:
        random.seed(seed)
    if circ_type == 'TP1':
        if circ is None:
            circ = QuantumCircuit(*([max(registers) + 1] * 2))

        number_qubits = len(registers)
        circ.u3(pi / 2, 0, 0, registers[0])
        for i in range(1):
            for j in range(1, number_qubits):
                add_hermitian_single_qubit_gate(circ, registers[j])
        return circ

    elif circ_type == 'TP2':
        if circ is None:
            circ = QuantumCircuit(*([max(registers) + 1] * 2))
        number_qubits = len(registers)
        circ.u3(pi / 2, 0, 0, registers[0])
        last_set = {}

        for j in range(1, number_qubits):
            last_set[j] = add_hermitian_single_qubit_gate(circ, registers[j])

        for j in range(1, number_qubits - 1, 2):
            circ.cx(registers[j], registers[j + 1])

        for j in range(1, number_qubits if number_qubits % 2 == 1 else number_qubits - 1):
            circ.u3(*(last_set[j] + (registers[j], ) ))

        return circ

    elif circ_type == 'NTP':
        if circ is None:
            circ = QuantumCircuit(*([max(registers) + 1] * 2))

        number_qubits = len(registers)

        circ.u3(pi / 2, 0, 0, registers[0])

        if number_qubits == 1:
            pass
        elif number_qubits == 2:
            add_hermitian_single_qubit_gate(circ, registers[1])
        elif number_qubits == 3:
            circ.cx(registers[1], registers[2])
            add_hermitian_single_qubit_gate(circ, registers[1])
            add_hermitian_single_qubit_gate(circ, registers[2])
            circ.cx(registers[1], registers[2])
        else:
            last_set = {}
            for i in range(1, number_qubits - 1, 2):
                circ.cx(registers[i], registers[i + 1])

            for i in range(1, number_qubits):
                last_set[i] = add_hermitian_single_qubit_gate(circ, registers[i])

            for i in range(2, number_qubits - 1, 2):
                circ.cx(registers[i], registers[i + 1])

            for i in range(2, number_qubits if number_qubits % 2 == 0 else number_qubits - 1):
                circ.u3(*(last_set[i] + (registers[i], ) ))

            for i in range(1, number_qubits - 1, 2):
                circ.cx(registers[i], registers[i + 1])

        return circ
    else:
        raise NotImplementedError


def get_matrix_from_circ(circ: QuantumCircuit, circ_registers):
    U_direct = np.eye(2**len(circ_registers))

    for gate in circ.data:
        U_gate = [Id for _ in range(len(circ_registers))]
        if len(gate[1]) == 1:
            if gate[0].name == 'x':
                n_gate = X
            elif gate[0].name == 'y':
                n_gate = Y
            elif gate[0].name == 'z':
                n_gate = Z
            elif gate[0].name == 'rz':
                n_gate = Rz(*gate[0].params)
            elif gate[0].name == 'u3':
                n_gate = U3(*list(map(lambda x: float(x), gate[0].params)))
            elif gate[0].name == 'h':
                n_gate = H
            elif gate[0].name == 's':
                n_gate = S
            else:
                raise NotImplementedError

            index = circ_registers.index(gate[1][0].index)
            U_gate[-1 - index] = n_gate

            U_direct = n_tensor_product(*U_gate) @ U_direct

        elif len(gate[1]) == 2:
            if gate[0].name == 'cx':
                cU = X
            elif gate[0].name == 'cu3':
                cU = U3(*list(map(lambda x: float(x), gate[0].params)))
            else:
                raise NotImplementedError

            U_step = np.zeros([2**len(circ_registers)]*2, dtype=complex)
            index_c = circ_registers.index(gate[1][0].index)
            index_t = circ_registers.index(gate[1][1].index)

            U_gate = [Id for _ in range(len(circ_registers))]
            U_gate[-1 - index_c] = np.diag([1, 0])
            U_gate[-1 - index_t] = Id
            U_step += n_tensor_product(*U_gate)

            U_gate = [Id for _ in range(len(circ_registers))]
            U_gate[-1 - index_c] = np.diag([0, 1])
            U_gate[-1 - index_t] = cU
            U_step += n_tensor_product(*U_gate)

            U_direct = U_step@U_direct
        else:
            raise NotImplementedError

    return U_direct