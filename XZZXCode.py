# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

from typing import List, Iterable, Dict
import pymatching
import numpy as np
import matplotlib.pyplot as plt

import rustworkx as rx
from rustworkx.visualization import mpl_draw
import stim ## version = '1.11.dev0'
import surface_code_decoder

class XZZXCode:
    """XZZXCode class."""

    def __init__(
        self,
        d: int,
        logical_observable: str = "Z",
        T: int = 0,
        gate_error_1q: float = 0,
        gate_error_2q: float = 0,
        idle_error_T1: float = 0,
        idle_error_T2: float = 0,
        measurement_error_rate: float = 0,
        XZZX: bool =True
    ):
        """
        Creates the circuits corresponding to a logical X or Z eigenstate encoded using the surface code.

        Implementation of a distance d surface code, implemented over
        T syndrome measurement rounds.

        Args:
            d (int): Number of code qubits (and hence repetitions) used.
            logical_observable (str): X or Z logical to be initalized ("Z" by default).
            T (int): Number of rounds of ancilla-assisted syndrome measurement (T = 0 by default).
            gate_error_1q (float): single-qubit depolarizing noise
            gate_error_2q (float): two-qubit depolarizing noise
            idle_error_T1 (float): bit-flip errors (X and Y Paulis) by idling (during measurement)
            idle_error_T2 (float): phase-flip errors
            measurement_error_rate (bool): two-qubit depolarizing error after initialization and during the parity measurement, as well as bit-flip error right before measurement.
        """

        self.d = d
        self.T = T
        self.logical_observable = logical_observable
        self.XZZX = XZZX
        
        self.logind = self.pauli_to_index(logical_observable)       
        
        self.q2i, self.a2i = self.qubit_indices()
        self.i2q = {v: k for k,v in self.q2i.items()}

        self.qubit_index_list = [i for i in self.q2i.values()]
        self.qubit_pos = [q for q in self.q2i.keys()]
        
        self.even_ancilla_pos = [a for a in self.a2i.keys() if a.imag%3==2]
        self.even_ancilla_index_list = [self.a2i[a] for a in self.even_ancilla_pos]
        self.even_x_ancilla_pos = [a for a in self.even_ancilla_pos if (a.real-1+a.imag-2)%6 == 3]
        self.even_x_ancilla_ind = [self.a2i[a] for a in self.even_x_ancilla_pos]
        self.even_z_ancilla_pos = [a for a in self.even_ancilla_pos if (a.real-1+a.imag-2)%6 == 0]
        self.even_z_ancilla_ind = [self.a2i[a] for a in self.even_z_ancilla_pos]

        self.odd_ancilla_pos = [a for a in self.a2i.keys() if a.imag%3==1]
        self.odd_ancilla_index_list = [self.a2i[a] for a in self.odd_ancilla_pos]
        self.odd_x_ancilla_pos = [a for a in self.odd_ancilla_pos if (a.real-2+a.imag-1)%6 == 3]
        self.odd_x_ancilla_ind = [self.a2i[a] for a in self.odd_x_ancilla_pos]
        self.odd_z_ancilla_pos = [a for a in self.odd_ancilla_pos if (a.real-2+a.imag-1)%6 == 0]
        self.odd_z_ancilla_ind = [self.a2i[a] for a in self.odd_z_ancilla_pos]

        self.ancilla_pairs = [self.a2i[a] for ae in self.even_ancilla_pos for a in [ae,ae+1-1j]]
        self.x_ancilla_pairs = [self.a2i[a] for ae in self.even_ancilla_pos if (ae.real-1+ae.imag-2)%6 == 3 for a in [ae,ae+1-1j]]

        self.pair_targets_round0 = [qind for anc_pos in self.even_x_ancilla_pos if anc_pos-1+1j in self.qubit_pos for qind in [self.q2i[anc_pos-1+1j], self.a2i[anc_pos]]]
        self.pair_targets_round0.extend([qind for anc_pos in self.even_z_ancilla_pos if anc_pos-1+1j in self.qubit_pos for qind in [self.q2i[anc_pos-1+1j],self.a2i[anc_pos]]])
        self.hadamards_targets_round0 = [self.q2i[anc_pos-1+1j] for anc_pos in self.even_x_ancilla_pos if anc_pos-1+1j in self.qubit_pos]
        if self.XZZX:
            self.hadamards_targets_round0.extend([self.q2i[anc_pos-1+1j] for anc_pos in self.even_z_ancilla_pos if anc_pos-1+1j in self.qubit_pos])

        self.pair_targets_round1 = [qind for anc_pos in self.odd_x_ancilla_pos if anc_pos+1+2j in self.qubit_pos for qind in [self.q2i[anc_pos+1+2j], self.a2i[anc_pos]]]
        self.pair_targets_round1.extend([qind for anc_pos in self.even_z_ancilla_pos if anc_pos-1-2j in self.qubit_pos for qind in [self.q2i[anc_pos-1-2j],self.a2i[anc_pos]]])
        if self.XZZX:
            self.hadamards_targets_round1 = []
        else:
            self.hadamards_targets_round1 = [self.q2i[anc_pos+1+2j] for anc_pos in self.odd_x_ancilla_pos if anc_pos+1+2j in self.qubit_pos]

        self.pair_targets_round2 = [qind for anc_pos in self.even_x_ancilla_pos if anc_pos-1-2j in self.qubit_pos for qind in [self.q2i[anc_pos-1-2j], self.a2i[anc_pos]]]
        self.pair_targets_round2.extend([qind for anc_pos in self.odd_z_ancilla_pos if anc_pos+1+2j in self.qubit_pos for qind in [self.q2i[anc_pos+1+2j],self.a2i[anc_pos]]])
        if self.XZZX:
            self.hadamards_targets_round2 = []
        else:
            self.hadamards_targets_round2 = [self.q2i[anc_pos-1-2j] for anc_pos in self.even_x_ancilla_pos if anc_pos-1-2j in self.qubit_pos]

        self.pair_targets_round3 = [qind for anc_pos in self.odd_x_ancilla_pos if anc_pos+1-1j in self.qubit_pos for qind in [self.q2i[anc_pos+1-1j],self.a2i[anc_pos]]]
        self.pair_targets_round3.extend([qind for anc_pos in self.odd_z_ancilla_pos if anc_pos+1-1j in self.qubit_pos for qind in [self.q2i[anc_pos+1-1j],self.a2i[anc_pos]]])
        self.hadamards_targets_round3 = [self.q2i[anc_pos+1-1j] for anc_pos in self.odd_x_ancilla_pos if anc_pos+1-1j in self.qubit_pos]
        if self.XZZX:
            self.hadamards_targets_round3.extend([self.q2i[anc_pos+1-1j] for anc_pos in self.odd_z_ancilla_pos if anc_pos+1-1j in self.qubit_pos])

        # edge qubits
        if self.logical_observable == 'Z':
            self.edge_qubits = [self.q2i[k+3j] for k in range(3,3*self.d+1,3)]
        elif self.logical_observable == 'X':
            self.edge_qubits = [self.q2i[3+k*1j] for k in range(3,3*self.d+1,3)]
        else:
            print('This logical observable is not implemented')
            return

        self.stim_circuit = stim.Circuit()
        self.stim_circuit += self.initialize_stim_circuit(gate_error_1q = gate_error_1q, gate_error_2q = gate_error_2q, measurement_error_rate = measurement_error_rate)
        self.stim_circuit += 1 * self.generate_stim_circuit_cycle(gate_error_1q = gate_error_1q, gate_error_2q = gate_error_2q, 
            idle_error_T1 = idle_error_T1, idle_error_T2 = idle_error_T2, measurement_error_rate = measurement_error_rate,detectors = False)
        self.stim_circuit += T * self.generate_stim_circuit_cycle(gate_error_1q = gate_error_1q, gate_error_2q = gate_error_2q,
            idle_error_T1 = idle_error_T1, idle_error_T2 = idle_error_T2, measurement_error_rate = measurement_error_rate)
        self.stim_circuit += self.final_measurement_stim_circuit(gate_error_2q = gate_error_2q,measurement_error_rate = measurement_error_rate)

    def pauli_to_index(self, paulixyz):
        pauliind = -1
        for i,s in enumerate(["Z","X","Y"]):
            if s == paulixyz:
                pauliind = i
        if pauliind == -1:
            print('Unexpected value for "logical_observable" or the "dedicated_link". The available options are "X", "Y", and "Z".')
            return
        return pauliind

    def sorted_complex(self, xs: Iterable[complex]) -> List[complex]:
        return sorted(xs, key=lambda v: (v.real, v.imag))

    def qubit_indices(self) -> List[Dict[complex, int]]:
        qubit_coordinates = [k + i*1j for k in range(3, 3*(self.d)+1, 3) for i in range(3, 3*(self.d)+1, 3)]

        unit_cell_coordinates = [k + i*1j for k in range(0, 3*(self.d)+1, 3) for i in range(0, 3*(self.d)+1, 3) 
                                if k%(3*self.d)!=0 or (k==0 and i%6==0) or (k==3*self.d and i%6==3) if  i%(3*self.d)!=0 or (i==0 and k%6==3) or (i==3*self.d and k%6==0)] 
        ancilla_coordinates = [q + a for q in unit_cell_coordinates for a in [1+2j,2+1j]]
        
        # preparing the dictionary 
        q2i: Dict[complex, int] = {q: i for i, q in enumerate(self.sorted_complex(qubit_coordinates))}
        a2i: Dict[complex, int] = {a: i for i, a in enumerate(self.sorted_complex(ancilla_coordinates), start=self.d**2)}

        return [q2i, a2i]

    def initialize_stim_circuit(self, gate_error_1q: float = 0, gate_error_2q: float = 0, measurement_error_rate: float = 0) -> stim.Circuit:
        circuit = stim.Circuit()
        for q, i in self.q2i.items():
                circuit.append("QUBIT_COORDS", [i], [q.real, q.imag])
        for q, i in self.a2i.items():
                circuit.append("QUBIT_COORDS", [i], [q.real, q.imag])
        
        circuit.append("R", self.qubit_index_list) # the logical Z is initialized
        if measurement_error_rate > 0:
            circuit.append("DEPOLARIZE1", self.qubit_index_list, 12/15*measurement_error_rate) # effective error from ancilla initialization
        if gate_error_2q > 0:
            circuit.append("DEPOLARIZE1", self.qubit_index_list, 12/15*gate_error_2q) # effective error from swapping the ancilla with the code qubit

        circuit.append("H", self.qubit_index_list[1::2])
        if gate_error_1q > 0:
            circuit.append("DEPOLARIZE1", self.qubit_index_list[1::2], gate_error_1q)

        if self.logical_observable == 'X':
            circuit.append("H", self.qubit_index_list) # the logical X is initialized
            if gate_error_1q > 0:
                circuit.append("DEPOLARIZE1", self.qubit_index_list, gate_error_1q)
        
        circuit.append("R", self.ancilla_pairs)
        if measurement_error_rate > 0:
            circuit.append("DEPOLARIZE2", self.ancilla_pairs, measurement_error_rate) # initialization error
 
        return circuit

    def generate_stim_circuit_cycle(self,
                            gate_error_1q: float = 0,
                            gate_error_2q: float = 0,
                            idle_error_T1: float = 0,
                            idle_error_T2: float = 0,
                            measurement_error_rate: float = 0, 
                            detectors: bool = True) -> stim.Circuit:
    
        measurement_times: Dict[str, int] = {}
        current_time = 0
 
        circuit = stim.Circuit()

        # circuit.append("H",self.even_x_ancilla_ind)
        # if gate_error_1q>0:
        #     circuit.append('DEPOLARIZE1', self.even_x_ancilla_ind, gate_error_1q)

        circuit.append("H",self.hadamards_targets_round0)
        if gate_error_1q>0:
            circuit.append('DEPOLARIZE1', self.hadamards_targets_round0, gate_error_1q)  
        circuit.append("CNOT", self.pair_targets_round0)
        if gate_error_2q>0:
            circuit.append('DEPOLARIZE2', self.pair_targets_round0, gate_error_2q)
        circuit.append("H",self.hadamards_targets_round0)
        if gate_error_1q>0:
            circuit.append('DEPOLARIZE1', self.hadamards_targets_round0, gate_error_1q)  
        circuit.append('SWAP', self.x_ancilla_pairs)
        if gate_error_2q>0:
            circuit.append('DEPOLARIZE2', self.x_ancilla_pairs, gate_error_2q)

        circuit.append("H",self.hadamards_targets_round1)
        if gate_error_1q>0:
            circuit.append('DEPOLARIZE1', self.hadamards_targets_round1, gate_error_1q)  
        circuit.append("CNOT", self.pair_targets_round1)
        if gate_error_2q>0:
            circuit.append('DEPOLARIZE2', self.pair_targets_round1, gate_error_2q)        
        circuit.append("H",self.hadamards_targets_round1)
        if gate_error_1q>0:
            circuit.append('DEPOLARIZE1', self.hadamards_targets_round1, gate_error_1q)  
        circuit.append('SWAP', self.ancilla_pairs)
        if gate_error_2q>0:
            circuit.append('DEPOLARIZE2', self.ancilla_pairs, gate_error_2q)
        
        circuit.append("H",self.hadamards_targets_round2)
        if gate_error_1q>0:
            circuit.append('DEPOLARIZE1', self.hadamards_targets_round2, gate_error_1q)  
        circuit.append("CNOT", self.pair_targets_round2)
        if gate_error_2q>0:
            circuit.append('DEPOLARIZE2', self.pair_targets_round2, gate_error_2q)
        circuit.append("H",self.hadamards_targets_round2)
        if gate_error_1q>0:
            circuit.append('DEPOLARIZE1', self.hadamards_targets_round2, gate_error_1q)  
        circuit.append('SWAP', self.x_ancilla_pairs)
        if gate_error_2q>0:
            circuit.append('DEPOLARIZE2', self.x_ancilla_pairs, gate_error_2q)
        
        circuit.append("H",self.hadamards_targets_round3)
        if gate_error_1q>0:
            circuit.append('DEPOLARIZE1', self.hadamards_targets_round3, gate_error_1q)  
        circuit.append("CNOT", self.pair_targets_round3)
        if gate_error_2q>0:
            circuit.append('DEPOLARIZE2', self.pair_targets_round3, gate_error_2q)
        circuit.append("H",self.hadamards_targets_round3)
        if gate_error_1q>0:
            circuit.append('DEPOLARIZE1', self.hadamards_targets_round3, gate_error_1q)  
        
        # circuit.append("H",self.odd_x_ancilla_ind)
        # if gate_error_1q>0:
        #     circuit.append('DEPOLARIZE1', self.odd_x_ancilla_ind, gate_error_1q)        

        #Measure ancilla
        for k in self.odd_ancilla_index_list:
            edge_key = 'plaquette_'+str(k)
            measurement_times[edge_key] = current_time
            current_time += 1 
        circuit.append("CNOT", self.ancilla_pairs) #controlled on the evens and targeted on the odds
        if measurement_error_rate > 0:
            circuit.append("DEPOLARIZE2", self.ancilla_pairs, measurement_error_rate) # measurement error (ramping in the PSB)
            circuit.append("X_ERROR", self.odd_ancilla_index_list, measurement_error_rate) # readout assignment error
        circuit.append("MR", self.odd_ancilla_index_list)
        circuit.append("R", self.even_ancilla_index_list)
        if measurement_error_rate > 0:
            circuit.append("DEPOLARIZE2", self.ancilla_pairs, measurement_error_rate) # initialization error

        #qubit idling errors during measurment
        if idle_error_T1>0:
            circuit.append('PAULI_CHANNEL_1', self.qubit_index_list, [idle_error_T1/2,idle_error_T1/2,0])
        if idle_error_T2>0:
            circuit.append('Z_ERROR', self.qubit_index_list, idle_error_T2)

        measurements_per_cycle = current_time
        det_circuit = stim.Circuit()
        if detectors:
            ## Determine which sets of measurements to compare in order to get detection events for plaquettes.
            for k in range(len(self.odd_ancilla_index_list)):
                edge_key = 'plaquette_'+str(self.odd_ancilla_index_list[k])
                record_targets = []
                relative_index = measurement_times[edge_key] - measurements_per_cycle
                record_targets.append(stim.target_rec(relative_index))
                record_targets.append(stim.target_rec(relative_index - measurements_per_cycle))
                circuit.append("DETECTOR", record_targets, [self.odd_ancilla_pos[k].real, self.odd_ancilla_pos[k].imag, 0])
        else: 
            ## In the first round after the noisy initialization we can already compare one type of plaquettes 
            ## with the value (+1) in which they are supposed bo be initialized.
            if self.logical_observable == 'Z':
                odd_log_ancilla_pos = self.odd_z_ancilla_pos
                odd_log_ancilla_ind = self.odd_z_ancilla_ind
            elif self.logical_observable == 'X':
                odd_log_ancilla_pos = self.odd_x_ancilla_pos
                odd_log_ancilla_ind = self.odd_x_ancilla_ind
            for k in range(len(odd_log_ancilla_ind)):
                edge_key = 'plaquette_'+str(odd_log_ancilla_ind[k])
                record_targets = []
                relative_index = measurement_times[edge_key] - measurements_per_cycle
                record_targets.append(stim.target_rec(relative_index))
                circuit.append("DETECTOR", record_targets, [odd_log_ancilla_pos[k].real, odd_log_ancilla_pos[k].imag, 0])
        circuit.append("SHIFT_COORDS", [], [0, 0, 1])
        circuit += det_circuit
        
        return circuit

    def final_measurement_stim_circuit(self, gate_error_1q: float = 0, gate_error_2q: float = 0, measurement_error_rate: float = 0) -> stim.Circuit:
        circuit = stim.Circuit()

        circuit.append("H", self.qubit_index_list[1::2])
        if gate_error_1q > 0:
            circuit.append("DEPOLARIZE1", self.qubit_index_list[1::2], gate_error_1q)

        if self.logical_observable == 'X':
            circuit.append("H", self.qubit_index_list) # the logical X is also initialized
            if gate_error_1q > 0:
                circuit.append("DEPOLARIZE1", self.qubit_index_list, gate_error_1q)   
        
        ## There should be a 'd'th ancilla and every qubit should be mapped to the nearest ancilla via a noisy CNOT before the ancillas are read out
        ## here we only account for the respective noise components that would enter the measurement 
        if gate_error_2q > 0:
            circuit.append("DEPOLARIZE1", self.qubit_index_list, 12/15*gate_error_2q) # effective error during the CNOT: IX IY and IZ do not affect the measurement outcome
        if measurement_error_rate > 0:
            circuit.append("DEPOLARIZE1", self.qubit_index_list, 12/15*measurement_error_rate) # effective error during the parity measurement
        if measurement_error_rate > 0:
            circuit.append("X_ERROR", self.qubit_index_list, measurement_error_rate) # readout assignment error
        circuit.append("M", self.qubit_index_list)
        measurements_since_last_cycle = len(self.qubit_index_list)
        
        if self.logical_observable == 'Z':
            even_log_ancilla_pos = self.even_z_ancilla_pos
            odd_log_ancilla_pos = self.odd_z_ancilla_pos
            odd_log_ancilla_ind = self.odd_z_ancilla_ind
        elif self.logical_observable == 'X':
            even_log_ancilla_pos = self.even_x_ancilla_pos
            odd_log_ancilla_pos = self.odd_x_ancilla_pos
            odd_log_ancilla_ind = self.odd_x_ancilla_ind

        detector_qubits = [[self.q2i[even_log_ancilla_pos[i]+r] for r in [-1+1j,-1-2j,2+1j,2-2j] if even_log_ancilla_pos[i]+r in self.qubit_pos]
                             for i in range(len(even_log_ancilla_pos))]

        measurements_per_cycle = self.d**2-1   

        det_circuit = stim.Circuit()
        for k in range(len(even_log_ancilla_pos)):
            record_targets = []
            for i in range(len(detector_qubits[k])):
                record_targets.append(stim.target_rec(detector_qubits[k][i]-measurements_since_last_cycle))
            anc_time = 0
            while self.odd_ancilla_index_list[anc_time] != odd_log_ancilla_ind[k]:
                anc_time+=1
            record_targets.append(stim.target_rec(anc_time-measurements_since_last_cycle-measurements_per_cycle))
            circuit.append("DETECTOR", record_targets, [odd_log_ancilla_pos[k].real, odd_log_ancilla_pos[k].imag, 1])
        circuit.append("SHIFT_COORDS", [], [0, 0, 1])
        circuit += det_circuit

        circuit.append("OBSERVABLE_INCLUDE",
                                [stim.target_rec(q - measurements_since_last_cycle) for q in sorted(self.edge_qubits)],
                                0)
        return circuit

    def draw_lattice(self, boundaries: bool = True, connectivity: bool = False, schedule: bool = False):
        if self.d <= 5:
            lattice = rx.PyGraph()
            q2i, a2i = self.qubit_indices()
            pos =[]
            node_color=[]

            for q in q2i:
                lattice.add_node(q)
                pos.append([q.real,q.imag])
                node_color.append('red')
            
            for a in a2i:
                lattice.add_node(a)
                pos.append([a.real,a.imag])
                node_color.append('gray')
            
            z_plaquettes = [[self.q2i[self.even_z_ancilla_pos[i]+r] for r in [-1+1j,2+1j,2-2j,-1-2j] if self.even_z_ancilla_pos[i]+r in self.qubit_pos]
                        for i in range(len(self.even_z_ancilla_pos))]
            
            x_plaquettes = [[self.q2i[self.even_x_ancilla_pos[i]+r] for r in [-1+1j,2+1j,2-2j,-1-2j] if self.even_x_ancilla_pos[i]+r in self.qubit_pos]
                        for i in range(len(self.even_x_ancilla_pos))]
            
            e = 0
            edge_cols = []
            edge_widths = []
            
            if connectivity:
                if schedule:
                    round = 0
                    for pair_color, pair_targets in zip(['red','green','blue','purple'],[self.pair_targets_round0,self.pair_targets_round1,self.pair_targets_round2,self.pair_targets_round3]):
                        pind = 0
                        for pairs in [[pair_targets[j], pair_targets[j+1]] for j in range(0,len(pair_targets),2)]:
                            lattice.add_edge(pairs[0],pairs[1],e)
                            e += 1
                            pind += 1
                        edge_cols.extend([pair_color]*pind)
                        edge_widths.extend([1.0]*pind)
                        print(pair_color,': round', round)   
                        round += 1           
                else:
                    pair_targets_all = []
                    pair_targets_all.append(self.pair_targets_round0)
                    pair_targets_all.append(self.pair_targets_round1)
                    pair_targets_all.append(self.pair_targets_round2)
                    pair_targets_all.append(self.pair_targets_round3)
                    for pairs in [[pair_targets_all[i][j], pair_targets_all[i][j+1]] for i in range(len(pair_targets_all)) for j in range(0,len(pair_targets_all[i]),2)]:
                        lattice.add_edge(pairs[0],pairs[1],e)
                        e += 1
                    e_connectivity = e
                    edge_cols.extend(['blue']*e_connectivity)
                    edge_widths.extend([1.0]*e_connectivity)
                e_connectivity = e
                for i in range(0,len(self.ancilla_pairs),2):
                    lattice.add_edge(self.ancilla_pairs[i],self.ancilla_pairs[i+1],e)
                    e += 1
                edge_cols.extend(['gray']*(e-e_connectivity))
                edge_widths.extend([3.0]*(e-e_connectivity))
                e_connectivity = e

            fig, ax = plt.subplots()
            mpl_draw(lattice, ax = ax, pos = pos, with_labels=True, node_color = node_color, edge_color = edge_cols, style = 'solid', width = edge_widths)   

            if boundaries:    
                for plind,pl in enumerate(z_plaquettes):
                    if len(pl) == 4:
                        for pairs in [(pl[i],pl[(i+1)%len(pl)]) for i in range(len(pl))]:
                            lattice.add_edge(pairs[0],pairs[1],e)
                            e += 1
                    else:
                        lattice.add_edge(pl[0],pl[1],e)
                        lattice.add_edge(pl[0],self.even_z_ancilla_ind[plind],e+1)
                        lattice.add_edge(pl[1],self.even_z_ancilla_ind[plind],e+2)
                        e += 3
                for plind,pl in enumerate(x_plaquettes):
                    if len(pl) == 4:
                        for pairs in [(pl[i],pl[(i+1)%len(pl)]) for i in range(len(pl))]:
                            lattice.add_edge(pairs[0],pairs[1],e)
                            e += 1
                    else:
                        lattice.add_edge(pl[0],pl[1],e)
                        lattice.add_edge(pl[0],self.even_x_ancilla_ind[plind],e+1)
                        lattice.add_edge(pl[1],self.even_x_ancilla_ind[plind],e+2)
                        e += 3
                e_boundary = e - e_connectivity
                edge_cols.extend(['gray']*e_boundary)
                edge_widths.extend([':']*e_boundary)
            
            mpl_draw(lattice, ax = ax, pos = pos, node_list = [], edge_color = edge_cols, style = 'dotted')

        else:
            print("Code distance is too large. Try d <= 5.")
   
    def matching(self):
        return surface_code_decoder.detector_error_model_to_pymatching_graph(self.stim_circuit.detector_error_model(decompose_errors=True,approximate_disjoint_errors=True))
    
    def PredictedObservableOutcome(self, sample: List[bool], m: pymatching.Matching):
        return m.decode(sample)[0]
