import math

from qiskit import QuantumCircuit, execute, Aer, IBMQ
from qiskit.compiler import transpile, assemble
# from qiskit.tools.jupyter import *
# from qiskit.visualization import *


def code_to_sign_pattern(code):
    signs = [1]*4
    modulus = code
    for idx in range(4):
        if modulus >= 2**(3-idx):
            signs[idx] = -1
            modulus -= 2**(3-idx)

    return signs

def implement_HSGS_2qubit(circuit, signs, count_map):


    # Define the binary representation of each state
    # binary_reps = ["00", "01", "10", "11"]
    binary_reps = ["11", "10", "01", "00"]

    # Initialize array to track which sign we have applied to each state
    implemented_signs = [1]*4

    # Flip all of the signs if coefficient for |00> is -1
    if signs[3] == -1:
        signs = [s*-1 for s in signs]

    # Apply the sign changes for states with a single "1", eg |01> and |10>
    for idx in count_map.get(1, []):
        # See if the sign that is implemented is not correct
        if signs[idx] != implemented_signs[idx]:
            # Get the binary representation of this index
            bin_rep = binary_reps[idx]
            # The qubit to flip is the one that is a "1"
            apply_qbits = [i for i, x in enumerate(bin_rep) if x == "1"]
            apply_qbit = apply_qbits[0]
            # Flip the sign of that qubit
            circuit.z(apply_qbit)

            # Update the sign as to which is currently implemented.
            for i, bin_rep in enumerate(binary_reps):
                if bin_rep[apply_qbit] == "1":
                    implemented_signs[i] *= -1


    # Apply the sign change for the states with 2 "1" qubits, eg |11>
    for idx in count_map.get(2, []):
        # See if the sign that is implemented is not correct
        if signs[idx] != implemented_signs[idx]:
            # Get the binary representation of this index
            bin_rep = binary_reps[idx]
            # The qubit to flip is the one that is a "1"
            apply_qbits = [i for i, x in enumerate(bin_rep) if x == "1"]

            # With a controlled Z gate, it doesn't matter which is control or target
            circuit.cz(apply_qbits[0], apply_qbits[1])

    return circuit

def create_perceptron_circuit(data_code, weight_code):
    # Initialize the quantum circuit: 3 qubits (2 state, 1 work), 1 cbit
    circuit = QuantumCircuit(3, 1)
    # Set the 4 pixel qubits to equisuperposition state
    circuit.h(0)
    circuit.h(1)

    # Convert the codes to the patterns of +1 and -1
    data_signs = code_to_sign_pattern(data_code)
    weight_signs = code_to_sign_pattern(weight_code)

    count_to_idxs = {
        1: [1, 2],
        2: [0],
    }

    # Apply the unitary transform to initialize the data
    circuit = implement_HSGS_2qubit(circuit, data_signs, count_to_idxs)
    # Apply the unitary transform to impliment dot product with weight vector
    circuit = implement_HSGS_2qubit(circuit, weight_signs, count_to_idxs)

    # Do the final H and X gates:
    circuit.h(0)
    circuit.h(1)
    circuit.x(0)
    circuit.x(1)

    # Final Toffoli gate
    circuit.ccx(0,1,2)

    # Measure the ancilla qubit
    circuit.measure([2],[0])


    return circuit


# Define map of the #of ones in the binary rep to the indexes:
# count_to_idxs = {
#     1: [1, 2, 4, 8],
#     2: [3, 5, 6, 9, 10, 12],
#     3: [7, 11, 13, 14],
#     4: [15],
# }


def main():

    # Use Aer's qasm_simulator
    simulator = Aer.get_backend('qasm_simulator')

    data_code = 0
    for weight_code in range(16):
        circuit = create_perceptron_circuit(data_code, weight_code)
        # import ipdb; ipdb.set_trace()
        # circuit.draw()

        # Execute the circuit on the qasm simulator
        job = execute(circuit, simulator, shots=8192)

        # Grab results from the job
        result = job.result()

        # Return counts
        counts = result.get_counts(circuit)
        print("Total [0,1] counts for i={} and w={} are:".format(data_code, weight_code),counts)


if __name__ == '__main__':
    main()
