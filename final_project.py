
import numpy as np
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit, execute, Aer


def code_to_sign_pattern(code):
    """
    Converts an integer code to its binary vector representation.
    For instance, 7 = 0111 -> (-1)^[0,1,1,1] = [1, -1, -1, -1]

    Currently only works for the 2 qubit case

    inputs: integer,
    outputs: list of +-1 values
    """

    signs = [1]*4
    modulus = code
    for idx in range(4):
        if modulus >= 2**(3-idx):
            signs[idx] = -1
            modulus -= 2**(3-idx)

    return signs

def implement_HSGS_2qubit(circuit, signs, count_map):
    """
    Implementation of the hypergraph states generation subroutine for 2 qubits

    HSGS is an efficient method for implementing the unitary transforms required
    to both initialize a data vector, as well as the rotation which implements
    the dot product of it with a weight vector

    inputs:
        circuit: work in progress IBM QuantumCircuit object
        signs: list of signs on the computational basis states
        count_map: (dict) maps the count of "1" qubits to
            a list of the comp basis indexes with that number of "1" qubits

    outputs:
        circuit: IBM QuantumCircuit object with the appropriate unitary transform added to it
    """

    # Define the binary representation of each state
    binary_reps = ["00", "01", "10", "11"]

    # Initialize array to track which sign we have applied to each state
    implemented_signs = [1]*4

    # Flip all of the signs if coefficient for |00> is -1
    if signs[0] == -1:
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
    """
    Implements a quantum circuit to compute the perceptron output a data and weight vector

    The data and weight vectors are binary, and specified by their integer code.
    The circuit returned  is suitable for simulation or execution with IBM qiskit

    inputs:
        data_code (int)
        weight_code (int)

    outputs:
        circuit qiskit QuantumCircuit Type
    """

    # Initialize the quantum circuit: 3 qubits (2 state, 1 work), 1 cbit
    circuit = QuantumCircuit(3, 1)
    # Set the 4 pixel qubits to equi-superposition state
    circuit.h(0)
    circuit.h(1)

    # Convert the codes to the patterns of +1 and -1
    data_signs = code_to_sign_pattern(data_code)
    weight_signs = code_to_sign_pattern(weight_code)

    count_to_idxs = {
        1: [1, 2],
        2: [3],
    }

    # Apply the unitary transform to initialize the data
    circuit = implement_HSGS_2qubit(circuit, data_signs, count_to_idxs)
    # Apply the unitary transform to implement dot product with weight vector
    circuit = implement_HSGS_2qubit(circuit, weight_signs, count_to_idxs)

    # Do the final H and X gates:
    circuit.h(0)
    circuit.h(1)
    circuit.x(0)
    circuit.x(1)

    # Final Toffoli gate
    circuit.ccx(0, 1, 2)

    # Measure the ancilla qubit
    circuit.measure([2], [0])


    return circuit


def main():

    n_shots = 8192  # Match the # of shots used in the paper

    # Use Aer's qasm_simulator
    simulator = Aer.get_backend('qasm_simulator')

    results_matrix = np.zeros((16,16))

    for data_code in range(16):
        for weight_code in range(16):
            circuit = create_perceptron_circuit(data_code, weight_code)

            # Pause simulation to check circuit is implemented as in the paper
            if data_code == 11 and weight_code == 7:
                # circuit.draw()
                import ipdb; ipdb.set_trace()

            # Execute the circuit on the qasm simulator
            job = execute(circuit, simulator, shots=n_shots)

            # Grab results from the job
            result = job.result()

            # Return counts
            counts = result.get_counts(circuit)
            print("Total counts for i={} and w={} are:".format(data_code, weight_code),counts)

            # Store results in the matrix
            results_matrix[data_code, weight_code] = counts.get('1', 0)/n_shots

    # Visualize the results as an image
    plt.imshow(results_matrix)
    plt.xlabel('Weight Code ($w_j$)')
    plt.ylabel('Data Code ($i_k$)')
    plt.colorbar()
    plt.title('Simulated Probabilities')
    plt.show()


if __name__ == '__main__':
    main()
