import argparse
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit, execute, Aer, IBMQ


def code_to_sign_pattern(code, n_qubits=2):
    """
    Converts an integer code to its binary vector representation.
    For instance, 7 = 0111 -> (-1)^[0,1,1,1] = [1, -1, -1, -1]

    Currently only works for the 2 qubit case

    inputs: integer,
    outputs: list of +-1 values
    """

    m = 2**n_qubits

    signs = [1]*m
    modulus = code
    for idx in range(m):
        if modulus >= 2**(m-1-idx):
            signs[idx] = -1
            modulus -= 2**(m-1-idx)

    return signs

def implement_HSGS_nqubits(circuit, signs, n_qubits=2):
    """
    Implementation of the hypergraph states generation subroutine for N qubits

    HSGS is an efficient method for implementing the unitary transforms required
    to both initialize a data vector, as well as the rotation which implements
    the dot product of it with a weight vector

    inputs:
        circuit: work in progress IBM QuantumCircuit object
        signs: list of signs on the computational basis states
        count_map: (dict) maps the count of "1" qubits to
            a list of the comp basis indexes with that number of "1" qubits

    outputs:
        circuit: IBM QuantumCircuit object with the HSGS unitary transform added to it
    """

    m = 2**n_qubits
    # Define the binary representation of each state
    # e.g. for 2 qubits binary_reps = ["00", "01", "10", "11"]
    str_format = "{:0" + str(n_qubits) + "b}"
    binary_reps = [str_format.format(j) for j in range(m)]

    # Need to generate the count map
    count_map = defaultdict(list)
    for idx, bin_rep in enumerate(binary_reps):
        n_ones = sum(s == "1" for s in bin_rep)
        count_map[n_ones].append(idx)
    count_map = dict(count_map)

    # Initialize array to track which sign we have applied to each state
    implemented_signs = [1]*m

    # Flip all of the signs if coefficient for |00> is -1
    if signs[0] == -1:
        signs = [s*-1 for s in signs]

    # Increment over the # of "1" qubits in the basis state
    for count in range(1,n_qubits+1):
        # Loop over all of the states which have count # of "1" qubits
        for idx in count_map.get(count, []):
            # See if the sign that is implemented is not correct
            if signs[idx] != implemented_signs[idx]:
                # Get the binary representation of this index
                bin_rep = binary_reps[idx]
                # The flip is controlled by the qubits that are a "1"
                apply_qbits = [i for i, x in enumerate(bin_rep) if x == "1"]

                # Flip the sign of those qubits, using the appropriate gate
                if count == 1:
                    circuit.z(apply_qbits[0])
                elif count == 2:
                    # With a controlled Z gate, it doesn't matter which is control or target
                    circuit.cz(apply_qbits[0], apply_qbits[1])
                else:
                    ctrl_qubits = [circuit.qubits[i] for i in apply_qbits[1:]]
                    targ_qubit = circuit.qubits[apply_qbits[0]]
                    # CRZ gate with theta=-pi is equiv to a CZ gate up to global phase
                    circuit.mcrz(-np.pi, ctrl_qubits, targ_qubit)

                # Update the signs to be what is currently implemented.
                for i, bin_rep in enumerate(binary_reps):
                    # Convert to numpy array so we can do vector ops
                    bool_rep = np.array([s == '1' for s in bin_rep])
                    if np.all(bool_rep[apply_qbits]):
                        implemented_signs[i] *= -1

    return circuit

def create_perceptron_circuit(data_code, weight_code, n_qubits=2):
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

    # Initialize the quantum circuit: N+1 qubits (N state, 1 work), 1 cbit
    circuit = QuantumCircuit(n_qubits+1, 1)
    # Set the 4 pixel qubits to equi-superposition state
    for qubit_idx in range(n_qubits):
        circuit.h(qubit_idx)

    # Convert the codes to the patterns of +1 and -1
    data_signs = code_to_sign_pattern(data_code, n_qubits)
    weight_signs = code_to_sign_pattern(weight_code, n_qubits)

    # Apply the unitary transform to initialize the data
    circuit = implement_HSGS_nqubits(circuit, data_signs, n_qubits)
    # Apply the unitary transform to implement dot product with weight vector
    circuit = implement_HSGS_nqubits(circuit, weight_signs, n_qubits)

    # Do the final H and X gates:
    for qubit_idx in range(n_qubits):
        circuit.h(qubit_idx)
    for qubit_idx in range(n_qubits):
        circuit.x(qubit_idx)

    # Final Toffoli gate
    circuit.mcx(control_qubits=list(range(n_qubits)), target_qubit=n_qubits)
    # Measure the ancilla qubit
    circuit.measure(
        qubit=[n_qubits],
        cbit=[0]
    )

    return circuit


def replicate_2qbit_results(backend, n_shots=8192):
    """
    Code to replicate HSGS results in Figure 3 of original paper
    """

    results_matrix = np.zeros((16,16))

    for data_code in range(16):
        for weight_code in range(16):
            circuit = create_perceptron_circuit(data_code, weight_code)

            # Pause simulation to check circuit is implemented as in the paper
            if data_code == 11 and weight_code == 7:
                # circuit.draw()
                import pdb; pdb.set_trace()

            # Execute the circuit on the qasm simulator
            job = execute(circuit, backend, shots=n_shots)

            # Grab results from the job and get counts
            result = job.result()
            counts = result.get_counts(circuit)
            print("Total counts for i={} and w={} are:".format(data_code, weight_code),counts)

            # Store normalized result in the matrix
            results_matrix[data_code, weight_code] = counts.get('1', 0)/n_shots

    # Visualize the results as an image
    plt.imshow(results_matrix)
    plt.xlabel('Weight Code ($w_j$)')
    plt.ylabel('Data Code ($i_k$)')
    plt.colorbar()
    plt.clim([0, 1])
    plt.title('Experimental Probabilities')
    plt.show()


def bit_string_to_img(bit_string):
    """
    Coverts from the bit string representation to its image
    """
    m = len(bit_string)
    img = np.zeros((m))
    for ii, bit in enumerate(bit_string):
        if bit == '0':
            img[ii] = 255

    hm = int(np.sqrt(m))
    return img.reshape((hm, hm))

def plot_bit_string(bit_string):
    """
    Plots the image associated with a data bit string
    """
    ax = plt.imshow(bit_string_to_img(bit_string), cmap='gray')
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)

    return ax

def replicate_4qbit_results(backend, n_shots=8192):
    """
    Code to replicate the 4 qubit results from the original paper
    """
    # Read off the pixel values from Fig 4 of the paper
    w = '0000001001110010'
    i_bit_strings = [
        '0000001001110000',
        '1111110110101101',
        '0000001001000010',
        '1111110110011101',
        '1100000001001000',
        '0011110111111111',
        '1001000100000000',
        '1000100001000010'
    ]
    # Convert these to the integer codes
    weight_code = int(w,2)

    # Plot the weight image first
    sub = plt.subplot(3, 3, 1)
    plot_bit_string(w)
    sub.set_title('w')

    results = []
    for ii, i_bit_str in enumerate(i_bit_strings):
        data_code = int(i_bit_str, 2)
        circuit = create_perceptron_circuit(data_code, weight_code, n_qubits=4)

        # Run the quantum circuit and grab results
        job = execute(circuit, backend, shots=n_shots)
        result = job.result()

        # Get counts and normalize to get a probability
        counts = result.get_counts(circuit)
        prob = counts.get('1', 0)/n_shots
        results.append(prob)

        # Plot the data image and the associated probability
        sub = plt.subplot(3, 3, ii+2)
        plot_bit_string(i_bit_str)
        sub.set_title("q.alg.={:0.4f}".format(prob))

    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--token", "-t",  help="Auth Token from IBM Website")
    parser.add_argument(
        "--backend", "-b",
        help="Hardware backend to use, eg 'ibmq_5_yorktown'. Uses Aer simulator if unprovided"
    )
    parser.add_argument(
        "-e", "--experiment", type=str, default='2qubits',
        help="Which experiment to run, either '2qubits' or '4qubits'"
    )
    cli_args = parser.parse_args()

    if cli_args.token is not None and cli_args.backend is not None:
        provider = IBMQ.enable_account(cli_args.token)
        backend = provider.get_backend(cli_args.backend)
        print("Using IBM Backend ", cli_args.backend)
    else:
        print("Backend not properly specified. Using Aer simulator.")
        backend = Aer.get_backend('qasm_simulator')

    if cli_args.experiment.lower() == "2qubits":
        print("Running the 2 qubit experiment")
        replicate_2qbit_results(backend)
    elif cli_args.experiment.lower() == "4qubits":
        print("Running the 4 qubit experiment")
        replicate_4qbit_results(backend)
    else:
        raise RuntimeError("Unknown experiment type ", cli_args.experiment)

