# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
#
# Copied and adapted from https://gist.github.com/Strilanc/a4a5f2f9410f84212f6b2c26d9e46e24


from dataclasses import dataclass
from typing import List, Dict, Set, FrozenSet, Iterable, Tuple
import stim
import surface_code_decoder
import pymatching
import numpy as np

# Define some data for working with the three edge orientations.
@dataclass
class EdgeType:
    pauli: str
    hex_to_hex_delta: complex
    hex_to_qubit_delta: complex
EDGE_TYPES = [
    EdgeType(pauli="X", hex_to_hex_delta=4 - 9j, hex_to_qubit_delta=2 - 3j),
    EdgeType(pauli="Y", hex_to_hex_delta=4 + 9j, hex_to_qubit_delta=2 + 3j),
    EdgeType(pauli="Z", hex_to_hex_delta=8, hex_to_qubit_delta=2),
]
EDGES_AROUND_HEX: List[Tuple[complex, complex]] = [
    (-2-3j, +2-3j),
    (+2-3j, +2),
    (+2,    +2+3j),
    (+2+3j, -2+3j),
    (-2+3j, -2),
    (-2,    -2-3j),
]

class HeavyHexHoneycombFloquetCode:
    def __init__(self,
                d: int,
                T: int = 0,
                logical_observable: str = "Z",
                gate_error_1q: float = 0,
                gate_error_2q: float = 0,
                idle_error_T1: float = 0,
                idle_error_T2: float = 0,
                measurement_error_rate: float = 0,
                noSWAP: bool = True
                ):
        self.d = d
        self.stim_circuit = self.generate_circuit(distance = d, cycles = 2*T,
                                                  gate_error_1q = gate_error_1q, gate_error_2q = gate_error_2q,
                                                  idle_error_T1 = idle_error_T1, idle_error_T2 = idle_error_T2,
                                                  measurement_error_rate = measurement_error_rate,
                                                  T2_X_basis = (logical_observable != "Z"), noSWAP = noSWAP)
    def generate_circuit_cycle(self,
                            q2i: Dict[complex, int],
                            ancA2i: Dict[complex, int],
                            ancB2i: Dict[complex, int],
                            gate_error_1q: float = 0,
                            gate_error_2q: float = 0,
                            idle_error_T1: float = 0,
                            idle_error_T2: float = 0,
                            measurement_error_rate: float = 0,
                            T2_X_basis: bool = False,
                            hex_centers: Dict[complex, int] = {},
                            distance: int = 2,
                            detectors: bool = True,
                            noSWAP: bool = False) -> stim.Circuit:
        round_circuits = []
        measurement_times: Dict[FrozenSet[int], int] = {}
        current_time = 0
        measurements_per_round: int
        for r in range(3):

            relevant_hexes = [h for h, category in hex_centers.items() if category == r]

            # Find the edges between the relevant hexes, grouped as X/Y/Z.
            edge_groups: Dict[str, List[FrozenSet[complex]]] = {"X": [], "Y": [], "Z": []}
            for h in relevant_hexes:
                for edge_type in EDGE_TYPES:
                    q1 = self.torus(h + edge_type.hex_to_qubit_delta, distance=distance)
                    q2 = self.torus(h + edge_type.hex_to_hex_delta - edge_type.hex_to_qubit_delta, distance=distance)
                    edge_groups[edge_type.pauli].append(frozenset([q1, q2]))
            x_qubits = [q2i[q] for pair in edge_groups["X"] for q in self.sorted_complex(pair)]
            y_qubits = [q2i[q] for pair in edge_groups["Y"] for q in self.sorted_complex(pair)]

            pair_targets = [
                q2i[q]
                for group in edge_groups.values()
                for pair in group
                for q in self.sorted_complex(pair)
            ]

            q0_to_ancA = [
                qind
                for group in edge_groups.values()
                for pair in group
                for qind in [q2i[self.sorted_complex(pair)[0]],
                            ancA2i[self.torus(self.sorted_complex(pair)[0]+(np.diff(self.sorted_complex(pair))[0].real>0)-2*(np.diff(self.sorted_complex(pair))[0].real>0 and self.sorted_complex(pair)[0].real==2 and self.sorted_complex(pair)[0].imag%6!=0)+(np.diff(self.sorted_complex(pair))[0].imag>0)*1j,distance=distance)]]
            ]
            q1_to_ancB = [
                qind
                for group in edge_groups.values()
                for pair in group
                for qind in [q2i[self.sorted_complex(pair)[1]],
                            ancB2i[self.torus(self.sorted_complex(pair)[1]-(np.diff(self.sorted_complex(pair))[0].real>0)+2*(np.diff(self.sorted_complex(pair))[0].real>0 and self.sorted_complex(pair)[1].real==8*distance-2 and self.sorted_complex(pair)[1].imag%6!=0)-(np.diff(self.sorted_complex(pair))[0].imag>0)*1j,distance=distance)]]
            ]

            ancA_to_ancB = [
                qind
                for group in edge_groups.values()
                for pair in group
                for qind in [ancA2i[self.torus(self.sorted_complex(pair)[0]+(np.diff(self.sorted_complex(pair))[0].real>0)-2*(np.diff(self.sorted_complex(pair))[0].real>0 and self.sorted_complex(pair)[0].real==2 and self.sorted_complex(pair)[0].imag%6!=0)+(np.diff(self.sorted_complex(pair))[0].imag>0)*1j,distance=distance)],
                            ancB2i[self.torus(self.sorted_complex(pair)[1]-(np.diff(self.sorted_complex(pair))[0].real>0)+2*(np.diff(self.sorted_complex(pair))[0].real>0 and self.sorted_complex(pair)[1].real==8*distance-2 and self.sorted_complex(pair)[1].imag%6!=0)-(np.diff(self.sorted_complex(pair))[0].imag>0)*1j,distance=distance)]]
            ]

            ancA_qubits = [
                ancA2i[self.torus(self.sorted_complex(pair)[0]+(np.diff(self.sorted_complex(pair))[0].real>0)-2*(np.diff(self.sorted_complex(pair))[0].real>0 and self.sorted_complex(pair)[0].real==2 and self.sorted_complex(pair)[0].imag%6!=0)+(np.diff(self.sorted_complex(pair))[0].imag>0)*1j,distance=distance)]
                for group in edge_groups.values()
                for pair in group
            ]

            ancB_qubits = [
                ancB2i[self.torus(self.sorted_complex(pair)[1]-(np.diff(self.sorted_complex(pair))[0].real>0)+2*(np.diff(self.sorted_complex(pair))[0].real>0 and self.sorted_complex(pair)[1].real==8*distance-2 and self.sorted_complex(pair)[1].imag%6!=0)-(np.diff(self.sorted_complex(pair))[0].imag>0)*1j,distance=distance)]
                for group in edge_groups.values()
                for pair in group
            ]
            
            circuit = stim.Circuit()      
            
            ##init error of the ancillas
            if measurement_error_rate > 0:
                circuit.append("DEPOLARIZE2", ancA_to_ancB, measurement_error_rate)  
            if noSWAP:
                ##XX-ZZ eigenstates
                circuit.append_operation("H", ancA_to_ancB[::2])
                circuit.append_operation("CNOT", ancA_to_ancB)


            # Make all the parity operations Z basis parities.
            circuit.append_operation("H", x_qubits)
            if gate_error_1q > 0:
                circuit.append_operation("DEPOLARIZE1", x_qubits, gate_error_1q)
            circuit.append_operation("H_YZ", y_qubits)
            if gate_error_1q > 0:
                circuit.append_operation("DEPOLARIZE1", y_qubits, gate_error_1q)

            circuit.append_operation("CNOT", q0_to_ancA)
            if gate_error_2q > 0:
                circuit.append_operation("DEPOLARIZE2", q0_to_ancA, gate_error_2q)

            ### SWAPs
            if not noSWAP:
                circuit.append_operation("SWAP", ancA_to_ancB)
                if gate_error_2q > 0:
                    circuit.append_operation("DEPOLARIZE2", ancA_to_ancB, gate_error_2q)

            circuit.append_operation("CNOT", q1_to_ancB)
            if gate_error_2q > 0:
                circuit.append_operation("DEPOLARIZE2", q1_to_ancB, gate_error_2q)
            
            # Restore qubit bases.
            circuit.append_operation("H", x_qubits)
            if gate_error_1q > 0:
                circuit.append_operation("DEPOLARIZE1", x_qubits, gate_error_1q)
            circuit.append_operation("H_YZ", y_qubits)
            if gate_error_1q > 0:
                circuit.append_operation("DEPOLARIZE1", y_qubits, gate_error_1q)

            # Measure TO BE UPDATED
            for k in range(0, len(pair_targets), 2):
                edge_key = frozenset([pair_targets[k], pair_targets[k + 1]])
                measurement_times[edge_key] = current_time
                current_time += 1
            circuit.append_operation("CNOT", ancA_to_ancB)
            if measurement_error_rate > 0:
                circuit.append("DEPOLARIZE2", ancA_to_ancB, measurement_error_rate)
                circuit.append("X_ERROR", ancB_qubits, measurement_error_rate)
            circuit.append_operation("MR", ancB_qubits) # ancB_qubits follows the same ordering as pair_targets
            circuit.append_operation("R", ancA_qubits) #init error comes at the beginning of the round

            # qubit idling errors during measurment
            if T2_X_basis:
                if idle_error_T1>0:
                    circuit.append('PAULI_CHANNEL_1', q2i.values(), [idle_error_T1/2,idle_error_T1/2,0])
                if idle_error_T2>0:
                    circuit.append('X_ERROR', q2i.values(), idle_error_T2)
            else:
                if idle_error_T1>0:
                    circuit.append('PAULI_CHANNEL_1', q2i.values(), [0,idle_error_T1/2,idle_error_T1/2])
                if idle_error_T2>0:
                    circuit.append('Z_ERROR', q2i.values(), idle_error_T2)

            # Multiply relevant measurements into the observable.
            included_measurements = []
            for group in edge_groups.values():
                for pair in group:
                    a, b = pair
                    if a.real == b.real == 2:
                        edge_key = frozenset([q2i[a], q2i[b]])
                        included_measurements.append(stim.target_rec(measurement_times[edge_key] - current_time))
            circuit.append_operation("OBSERVABLE_INCLUDE", included_measurements, 0)

            round_circuits.append(circuit)
        measurements_per_cycle = current_time
        measurements_per_round = measurements_per_cycle // 3

        # Determine which sets of measurements to compare in order to get detection events in the bulk.
        if detectors:
            for r in range(3):
                circuit = stim.Circuit()
                relevant_hexes = [h for h, category in hex_centers.items() if category == (r + 1) % 3]
                end_time = (r + 1) * measurements_per_round
                for h in relevant_hexes:
                    record_targets = []
                    for a, b in EDGES_AROUND_HEX:
                        q1 = self.torus(h + a, distance=distance)
                        q2 = self.torus(h + b, distance=distance)
                        edge_key = frozenset([q2i[q1], q2i[q2]])
                        relative_index = (measurement_times[edge_key] - end_time) % measurements_per_cycle - measurements_per_cycle
                        record_targets.append(stim.target_rec(relative_index))
                        record_targets.append(stim.target_rec(relative_index - measurements_per_cycle))
                    circuit.append_operation("DETECTOR", record_targets, [h.real, h.imag, 0])
                circuit.append_operation("SHIFT_COORDS", [], [0, 0, 1])
                round_circuits[r] += circuit

        full_circuit = stim.Circuit()
        full_circuit += round_circuits[0] + round_circuits[1] + round_circuits[2]
        return full_circuit


    def generate_circuit(self,
                        distance: int, cycles: int,
                        gate_error_1q: float = 0,
                        gate_error_2q: float = 0,
                        idle_error_T1: float = 0,
                        idle_error_T2: float = 0,
                        measurement_error_rate: float = 0,
                        T2_X_basis: bool = False,
                        noSWAP: bool = False) -> stim.Circuit:

        # Generate and categorize the hexes defining the circuit.
        hex_centers: Dict[complex, int] = {}
        for row in range(3 * distance):
            for col in range(2 * distance):
                center = row * 6j + 4 * col - 3j * (col % 2)
                category = (-row - col % 2) % 3
                hex_centers[self.torus(center, distance=distance)] = category

        # Find all the qubit positions around the hexes.
        qubit_coordinates: Set[complex] = set()
        for h in hex_centers:
            for edge_type in EDGE_TYPES:
                for sign in [-1, +1]:
                    q = h + edge_type.hex_to_qubit_delta * sign
                    qubit_coordinates.add(self.torus(q, distance=distance))
        # print_2d(hex_centers)
        # print(qubit_coordinates)
        ancillaA_coordinates: Set[complex] = set()
        for q in qubit_coordinates:
            ancillaA_coordinates.add(self.torus(q+1j+(q.imag==18*distance-3)*1j, distance=distance))
            if self.torus(q + 2, distance=distance) not in hex_centers:
                ancillaA_coordinates.add(self.torus(q+1+2*(q.real==8*distance-2), distance=distance))
        
        ancillaB_coordinates: Set[complex] = set()
        for q in qubit_coordinates:
            ancillaB_coordinates.add(self.torus(q-1j-(q.imag==0)*1j, distance=distance))
            if self.torus(q - 2, distance=distance) not in hex_centers:
                ancillaB_coordinates.add(self.torus(q-1-2*(q.real==2), distance=distance))

        # Assign integer indices to the qubit positions.
        q2i: Dict[complex, int] = {q: i for i, q in enumerate(self.sorted_complex(qubit_coordinates))}
        ancA2i: Dict[complex, int] = {a: i for i, a in enumerate(self.sorted_complex(ancillaA_coordinates),start=len(q2i))}
        ancB2i: Dict[complex, int] = {a: i for i, a in enumerate(self.sorted_complex(ancillaB_coordinates),start=len(q2i)+len(ancA2i))}

        # Generate a circuit performing the parity measurements that are part of each round.
        # Also keep track of the exact order the measurements occur in.
        round_circuit_no_noise_no_detectors = self.generate_circuit_cycle(
            q2i=q2i,
            ancA2i=ancA2i,
            ancB2i=ancB2i,
            gate_error_1q = 0,
            gate_error_2q = 0,
            idle_error_T1 = 0,
            idle_error_T2 = 0,
            measurement_error_rate = 0,
            hex_centers=hex_centers,
            distance=distance,
            detectors=False,
            noSWAP = noSWAP
        )
        round_circuit_no_noise_yes_detectors = self.generate_circuit_cycle(
            q2i=q2i,
            ancA2i=ancA2i,
            ancB2i=ancB2i,
            gate_error_1q = 0,
            gate_error_2q = 0,
            idle_error_T1 = 0,
            idle_error_T2 = 0,
            measurement_error_rate = 0,
            hex_centers=hex_centers,
            distance=distance,
            detectors=True,
            noSWAP = noSWAP
        )
        round_circuit_yes_noise_yes_detectors = self.generate_circuit_cycle(
            q2i=q2i,
            ancA2i=ancA2i,
            ancB2i=ancB2i,
            gate_error_1q = gate_error_1q,
            gate_error_2q = gate_error_2q,
            idle_error_T1 = idle_error_T1,
            idle_error_T2 = idle_error_T2,
            measurement_error_rate = measurement_error_rate,
            T2_X_basis = T2_X_basis,
            hex_centers=hex_centers,
            distance=distance,
            detectors=True,
            noSWAP = noSWAP
        )

        # Put together the pieces to get a correctable noisy circuit with noiseless time padding
        # (since the time boundaries are not implemented yet).
        full_circuit = stim.Circuit()
        for q, i in q2i.items():
            full_circuit.append_operation("QUBIT_COORDS", [i], [q.real, q.imag])
        for a, i in ancA2i.items():
            full_circuit.append_operation("QUBIT_COORDS", [i], [a.real, a.imag])
        for a, i in ancB2i.items():
            full_circuit.append_operation("QUBIT_COORDS", [i], [a.real, a.imag])

        # Initialize data qubits along logical observable column into correct basis for observable to be deterministic.
        qubits_along_column = sorted([q for q in qubit_coordinates if q.real == 2], key=lambda v: v.imag)
        initial_bases_along_column = "ZY_ZX_" * distance
        x_initialized = [q2i[q] for q, b in zip(qubits_along_column, initial_bases_along_column) if b == "X"]
        y_initialized = [q2i[q] for q, b in zip(qubits_along_column, initial_bases_along_column) if b == "Y"]
        full_circuit.append_operation("H", x_initialized)
        full_circuit.append_operation("H_YZ", y_initialized)

        full_circuit += (
                round_circuit_no_noise_no_detectors * 2
                + round_circuit_no_noise_yes_detectors * 2
        )

        full_circuit += (
                round_circuit_yes_noise_yes_detectors * cycles
                + round_circuit_no_noise_yes_detectors * 2
                + round_circuit_no_noise_no_detectors * 2
        )

        # Finish circuit with data measurements.
        qubit_coords_to_measure = [q for q, b in zip(qubits_along_column, initial_bases_along_column) if b != "_"]
        qubit_indices_to_measure= [q2i[q] for q in qubit_coords_to_measure]
        order = {q: i for i, q in enumerate(qubit_indices_to_measure)}
        assert cycles % 2 == 0
        full_circuit.append_operation("H_YZ", y_initialized)
        full_circuit.append_operation("H", x_initialized)
        full_circuit.append_operation("M", qubit_indices_to_measure)

        full_circuit.append_operation("OBSERVABLE_INCLUDE",
                                    [stim.target_rec(i - len(qubit_indices_to_measure)) for i in order.values()],
                                    0)

        return full_circuit

    def torus(self, c: complex, *, distance: int) -> complex:
        r = c.real % (distance * 8)
        i = c.imag % (distance * 18)
        return r + i*1j

    def sorted_complex(self, xs: Iterable[complex]) -> List[complex]:
        return sorted(xs, key=lambda v: (v.real, v.imag))
    
    def matching(self):
        return surface_code_decoder.detector_error_model_to_pymatching_graph(self.stim_circuit.detector_error_model(decompose_errors=True,approximate_disjoint_errors=True))
    
    def PredictedObservableOutcome(self, sample: List[bool], m: pymatching.Matching):
        return m.decode(sample)[0]
