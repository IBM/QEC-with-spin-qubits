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
import math
import pymatching
import numpy as np
import matplotlib.pyplot as plt

import networkx as nx
import retworkx as rx
from retworkx.visualization import mpl_draw
import stim ## version = '1.11.dev0'


class XYZ2Code:
    """XYZ2Code class."""
    def __init__(
        self,
        d: int,
        T: int = 0,
        logical_observable: str = "Z",
        gate_error_1q: float = 0,
        gate_error_2q: float = 0,
        idle_error_T1: float = 0,
        idle_error_T2: float = 0,
        measurement_error_rate: float = 0,
        dedicated_link: str = "Z",
        noSWAP: bool = False
    ):
        """
        Creates the circuits corresponding to a logical X or Z eigenvalue encoded
        using the XYZ2 code. 
        
        In the versions "_noisyLogRO" the initialization and readout of the
        logical observable is fault tolerant. 
        
        In version"_noisyLogRO_v2" the layout and the assumed connectivity is consistent
        for the logical initialization and readout.

        Implementation of a distance d XYZ^2 code, implemented over
        T syndrome measurement rounds.

        Args:
            d (int): Number of code qubits (and hence repetitions) used.
            T (int): Number of rounds of ancilla-assisted syndrome measurement (T = 0 by default).
            dedicated_link (str): the link operator "X", "Y", or "Z", to be measured together 
            with the plaquettes (Z links are used by default)
            logical_observable (str): logical observable "X", "Y", or "Z" (Z basis used by default)
            T (int): number of syndrome measurement rounds
            gate_error_1q (float): single-qubit depolarizing noise
            gate_error_2q (float): two-qubit depolarizing noise
            idle_error_T1 (float): bit-flip errors (X and Y Paulis) by idling (during measurement)
            idle_error_T2 (float): phase-flip errors
            measurement_error_rate (bool): two-qubit depolarizing error after initialization and 
            during the parity measurement, as well as bit-flip error right before measurement.
            noSWAP (bool): prepares the ancillas in a Bell-state and measures their parity without
            requiring any SWAPs in the syndrome measurements (False by default).
        """

        self.d = d
        self.T = T
        self.dedicated_link = dedicated_link
        self.logical_observable = logical_observable
        self.noSWAP = noSWAP
        self.gate_error_1q = gate_error_1q
        self.gate_error_2q = gate_error_2q
        self.idle_error_T1 = idle_error_T1
        self.idle_error_T2 = idle_error_T2
        self.measurement_error_rate = measurement_error_rate

        qubits_around_center = [2, 1-1j, -1j, -1, 1j, 1+1j] # wrt. tha ancillas with even real part
        # self.type1_XYZschedule = [0,1,2,5,4,3] # the order for ZXYZXY for the Z-link version
        # self.type2_XYZschedule = [0,3,4,5,2,1] # the order for ZXYZXY for the Z-link version
        self.type2_XYZschedule = [0,1,2,5,4,3] # the order for ZXYZXY for the Z-link version
        self.type1_XYZschedule = [0,3,4,5,2,1] # the order for ZXYZXY for the Z-link version
        
        self.num_pl_rounds = max(max(self.type1_XYZschedule),max(self.type2_XYZschedule))+1

        self.linkind = self.pauli_to_index(dedicated_link)
        
        self.logind = self.pauli_to_index(logical_observable)

        if self.linkind > 1:
            print('This link is not impllemented use X or Z.')
            return            

        if self.logind == (self.linkind + 1) % 3:
            print('This observable is not implemented. For Z links use Y, for X links use Z')
            return

        self.Hi_stim = [["I","H","H_YZ"][i % 3] for i in range(self.linkind,self.linkind+3)]
        
        self.q2i, self.a2i = self.qubit_indices()
        self.i2q = {v: k for k,v in self.q2i.items()}

        self.qubit_index_list = [i for i in self.q2i.values()]  
        self.qubit_pos = [q for q in self.q2i.keys()]
        self.ancilla_pos = [a for a in self.a2i.keys()]

        self.even_ancilla_pos = [a for a in self.a2i.keys() if a.real%2==0]
        self.even_ancilla_index_list = [self.a2i[a] for a in self.even_ancilla_pos]
        self.odd_ancilla_index_list = [self.a2i[a+1] for a in self.even_ancilla_pos]
        self.ancilla_pairs = [self.a2i[a] for ae in self.even_ancilla_pos for a in [ae,ae+1]]

        self.pl_ancilla_pos = [a for a in self.even_ancilla_pos if a != self.d*1j]
        self.pl_ancilla_index_list = [self.a2i[a] for a in self.pl_ancilla_pos]
        self.odd_pl_ancilla_index_list = [self.a2i[a+1] for a in self.pl_ancilla_pos]
        self.type1_even_ancilla_pos = [a for a in self.even_ancilla_pos if (a.real%4 == 0)]
        self.type1_even_pl_ancilla_pos = [a for a in self.even_ancilla_pos if (a.real%4 == 0) and (a != self.d*1j)]
        self.type1_pl_ancilla_index_list = [self.a2i[a] for a in self.type1_even_ancilla_pos]
        self.type2_even_ancilla_pos = [a for a in self.even_ancilla_pos if (a.real%4 == 2)]
        self.type2_pl_ancilla_index_list = [self.a2i[a] for a in self.type2_even_ancilla_pos]

        self.type1_ancilla_pos = [a for a in self.ancilla_pos if (a.real%4 == 0) or (a.real%4 == 1)]
        self.type1_ancilla_index_list = [self.a2i[a] for a in self.type1_ancilla_pos]
        self.type1_even_ancilla_index_list = [self.a2i[a] for a in self.type1_ancilla_pos if a in self.even_ancilla_pos]
        self.type1_odd_ancilla_pos = [a for a in self.type1_ancilla_pos if a not in self.even_ancilla_pos]
        self.type1_odd_ancilla_index_list = [self.a2i[a] for a in self.type1_ancilla_pos if a not in self.even_ancilla_pos]
        self.type1_ancilla_pairs = [self.a2i[a] for ae in self.type1_even_ancilla_pos for a in [ae,ae+1]]

        self.type2_ancilla_pos = [a for a in self.ancilla_pos if (a.real%4 == 2) or (a.real%4 == 3)]
        self.type2_ancilla_index_list = [self.a2i[a] for a in self.type2_ancilla_pos]
        self.type2_even_ancilla_index_list = [self.a2i[a] for a in self.type2_ancilla_pos if a in self.even_ancilla_pos]
        self.type2_odd_ancilla_pos = [a for a in self.type2_ancilla_pos if a not in self.even_ancilla_pos]
        self.type2_odd_ancilla_index_list = [self.a2i[a] for a in self.type2_ancilla_pos if a not in self.even_ancilla_pos]
        self.type2_ancilla_pairs = [self.a2i[a] for ae in self.type2_even_ancilla_pos for a in [ae,ae+1]]

        # plaquette qubits and the corresponding schedule, ordered the same way as the ancillas
        self.type1_plaquettes = [[(self.q2i[center+relpos], self.type1_XYZschedule[relposind],relposind) for relposind,relpos in enumerate(qubits_around_center) 
                            if center + relpos in self.q2i] for center in self.type1_even_pl_ancilla_pos]
        self.type2_plaquettes = [[(self.q2i[center+relpos], self.type2_XYZschedule[relposind],relposind) for relposind,relpos in enumerate(qubits_around_center) 
                            if center + relpos in self.q2i] for center in self.type2_even_ancilla_pos]
        
        self.type1_plaquette_qubits_per_round_per_pauli = [[[i[0] for pl in range(len(self.type1_plaquettes)) for i in self.type1_plaquettes[pl] if i[1]==round if i[2]%3==pauli] 
                                                            for pauli in range(3)] for round in range(self.num_pl_rounds)]
        self.type2_plaquette_qubits_per_round_per_pauli = [[[i[0] for pl in range(len(self.type2_plaquettes)) for i in self.type2_plaquettes[pl] if i[1]==round if i[2]%3==pauli] 
                                                            for pauli in range(3)] for round in range(self.num_pl_rounds)]
        
        self.type1_pair_targets_all = [[i for pl in range(len(self.type1_plaquettes)) for qubit in range(len(self.type1_plaquettes[pl])) 
                                    for i in [self.type1_plaquettes[pl][qubit][0],self.a2i[self.type1_even_pl_ancilla_pos[pl]+(self.type1_plaquettes[pl][qubit][2]<3)]]
                                    if self.type1_plaquettes[pl][qubit][1] == t] for t in range(self.num_pl_rounds)] 
        
        self.type2_pair_targets_all = [[i for pl in range(len(self.type2_plaquettes)) for qubit in range(len(self.type2_plaquettes[pl])) 
                                    for i in [self.type2_plaquettes[pl][qubit][0],self.a2i[self.type2_even_ancilla_pos[pl]+(self.type2_plaquettes[pl][qubit][2]<3)]]
                                    if self.type2_plaquettes[pl][qubit][1] == t] for t in range(self.num_pl_rounds)] 
        
        # link qubits
        self.links = [[self.q2i[qpos], self.q2i[qpos+1]] for qpos in self.qubit_pos if qpos+1 in self.qubit_pos]# if qpos+1 not in self.even_ancilla_pos]
        self.type1_links = [[self.q2i[qpos], self.q2i[qpos+1]] for qpos in self.qubit_pos if (qpos+1 in self.qubit_pos) and (qpos.real%4==0)]
        self.type2_links = [[self.q2i[qpos], self.q2i[qpos+1]] for qpos in self.qubit_pos if (qpos+1 in self.qubit_pos) and (qpos.real%4==2)]

        # edge qubits
        self.edge1_qubits = [self.q2i[k -int((2*self.d-2-abs(k)+(k > 0))/2)*1j] for k in range(-2*self.d+2, 1, 1) if k%4 !=1]
        self.edge2_qubits = [self.q2i[k + int((2*self.d-2-abs(k)+(k > 0))/2)*1j] for k in range(-2*self.d+2, 2, 1) if k%4 !=3]

        # qubit lists for the logical initialization and readout
        ## generating 5-qubit groups for each link
        ## connectivity q0-a0, q1-a0, q1-q2, a0-a1, measurements q1-q2 and a0-a1
        q0_qubits = [self.links[l][0] for l in range(len(self.links))] 
        q1_qubits = [self.links[l][1] for l in range(len(self.links))] 
        q2_qubits = []
        for l in range(len(self.links)):
            if self.i2q[self.links[l][1]]+1-1j in self.qubit_pos:
                q2_qubits.append(self.q2i[self.i2q[self.links[l][1]]+1-1j])
            else:
                q2_qubits.append(-1)
        a0_ancillas = [self.a2i[self.even_ancilla_pos[l]+((self.even_ancilla_pos[l] in self.type1_ancilla_pos)+(self.even_ancilla_pos[l].real<0))%2] for l in range(len(self.links))]
        a1_ancillas = [self.a2i[self.even_ancilla_pos[l]+((self.even_ancilla_pos[l] in self.type2_ancilla_pos)+(self.even_ancilla_pos[l].real<0))%2] for l in range(len(self.links))]
        self.q1_noq2_typeab = [q1_qubits[l] for l in range(len(self.links)) if q2_qubits[l]==-1] # independent from logind
        if self.logind == (self.linkind - 1)%3: 
            self.q0_typea = [q0_qubits[l] for l in range(len(self.links)) if a0_ancillas[l] in self.type1_ancilla_pairs]
            self.q1_typea = [q1_qubits[l] for l in range(len(self.links)) if a0_ancillas[l] in self.type1_ancilla_pairs]
            self.q2_typea = [q2_qubits[l] for l in range(len(self.links)) if a0_ancillas[l] in self.type1_ancilla_pairs]
            self.q2_nonex_listind = [ind for ind,q in enumerate(self.q2_typea) if q==-1]
            self.a0_typea = [a0_ancillas[l] for l in range(len(self.links)) if a0_ancillas[l] in self.type1_ancilla_pairs]
            self.a1_typea = [a1_ancillas[l] for l in range(len(self.links)) if a0_ancillas[l] in self.type1_ancilla_pairs]
            self.q2_to_q1_typea = [ind for l in range(len(self.type1_links)) for ind in [self.q2_typea[l],self.q1_typea[l]] if self.q2_typea[l]!=-1]
            self.q0_to_anc0_typea = [ind for l in range(len(self.type1_links)) for ind in [self.q0_typea[l],self.a0_typea[l]]]
            self.q1_to_anc0_typea = [ind for l in range(len(self.type1_links)) for ind in [self.q1_typea[l],self.a0_typea[l]]]
            self.q1_to_anc0_noq2_typea = [ind for l in range(len(self.type1_links)) for ind in [self.q1_typea[l],self.a0_typea[l]] if self.q2_typea[l]!=-1]
            self.anc0_to_anc1_typea = [ind for l in range(len(self.type1_links)) for ind in [self.a0_typea[l],self.a1_typea[l]]]
            self.typea_odd_ancilla_index_list = self.type1_odd_ancilla_index_list
            self.typea_odd_ancilla_pos = self.type1_odd_ancilla_pos
            self.typea_pl_ancilla_index_list = self.type1_pl_ancilla_index_list
            
            self.q0_typeb = [q0_qubits[l] for l in range(len(self.links)) if a0_ancillas[l] in self.type2_ancilla_pairs]
            self.q1_typeb = [q1_qubits[l] for l in range(len(self.links)) if a0_ancillas[l] in self.type2_ancilla_pairs]
            self.q2_typeb = [q2_qubits[l] for l in range(len(self.links)) if a0_ancillas[l] in self.type2_ancilla_pairs]
            self.a0_typeb = [a0_ancillas[l] for l in range(len(self.links)) if a0_ancillas[l] in self.type2_ancilla_pairs]
            self.a1_typeb = [a1_ancillas[l] for l in range(len(self.links)) if a0_ancillas[l] in self.type2_ancilla_pairs]
            self.q0_to_anc0_typeb = [ind for l in range(len(self.type2_links)) for ind in [self.q0_typeb[l],self.a0_typeb[l]]]
            self.q2_to_q1_typeb = [ind for l in range(len(self.type2_links)) for ind in [self.q2_typeb[l],self.q1_typeb[l]] if self.q2_typeb[l]!=-1]
            self.anc0_to_anc1_typeb = [ind for l in range(len(self.type2_links)) for ind in [self.a0_typeb[l],self.a1_typeb[l]]]
            self.typeb_odd_ancilla_index_list = self.type2_odd_ancilla_index_list
            self.typeb_ancilla_index_list = self.type2_ancilla_index_list
        elif self.logind == self.linkind:
            self.q0_typea = [q0_qubits[l] for l in range(len(self.links)) if a0_ancillas[l] in self.type2_ancilla_pairs]
            self.q1_typea = [q1_qubits[l] for l in range(len(self.links)) if a0_ancillas[l] in self.type2_ancilla_pairs]
            self.q2_typea = [q2_qubits[l] for l in range(len(self.links)) if a0_ancillas[l] in self.type2_ancilla_pairs]
            self.q2_nonex_listind = [ind for ind,q in enumerate(self.q2_typea) if q==-1]
            self.a0_typea = [a0_ancillas[l] for l in range(len(self.links)) if a0_ancillas[l] in self.type2_ancilla_pairs]
            self.a1_typea = [a1_ancillas[l] for l in range(len(self.links)) if a0_ancillas[l] in self.type2_ancilla_pairs]
            self.q2_to_q1_typea = [ind for l in range(len(self.type2_links)) for ind in [self.q2_typea[l],self.q1_typea[l]] if self.q2_typea[l]!=-1]
            self.q0_to_anc0_typea = [ind for l in range(len(self.type2_links)) for ind in [self.q0_typea[l],self.a0_typea[l]]]
            self.q1_to_anc0_typea = [ind for l in range(len(self.type2_links)) for ind in [self.q1_typea[l],self.a0_typea[l]]]
            self.q1_to_anc0_noq2_typea = [ind for l in range(len(self.type2_links)) for ind in [self.q1_typea[l],self.a0_typea[l]] if self.q2_typea[l]!=-1]
            self.anc0_to_anc1_typea = [ind for l in range(len(self.type2_links)) for ind in [self.a0_typea[l],self.a1_typea[l]]]
            self.typea_odd_ancilla_index_list = self.type2_odd_ancilla_index_list
            self.typea_odd_ancilla_pos = self.type2_odd_ancilla_pos
            self.typea_pl_ancilla_index_list = self.type2_pl_ancilla_index_list

            self.q0_typeb = [q0_qubits[l] for l in range(len(self.links)) if a0_ancillas[l] in self.type1_ancilla_pairs]
            self.q1_typeb = [q1_qubits[l] for l in range(len(self.links)) if a0_ancillas[l] in self.type1_ancilla_pairs]
            self.q2_typeb = [q2_qubits[l] for l in range(len(self.links)) if a0_ancillas[l] in self.type1_ancilla_pairs]
            self.a0_typeb = [a0_ancillas[l] for l in range(len(self.links)) if a0_ancillas[l] in self.type1_ancilla_pairs]
            self.a1_typeb = [a1_ancillas[l] for l in range(len(self.links)) if a0_ancillas[l] in self.type1_ancilla_pairs]
            self.q0_to_anc0_typeb = [ind for l in range(len(self.type1_links)) for ind in [self.q0_typeb[l],self.a0_typeb[l]]]
            self.q2_to_q1_typeb = [ind for l in range(len(self.type1_links)) for ind in [self.q2_typeb[l],self.q1_typeb[l]] if self.q2_typeb[l]!=-1]
            self.anc0_to_anc1_typeb = [ind for l in range(len(self.type1_links)) for ind in [self.a0_typeb[l],self.a1_typeb[l]]]
            self.typeb_odd_ancilla_index_list = self.type1_odd_ancilla_index_list
            self.typeb_ancilla_index_list = self.type1_ancilla_index_list

        

        self.stim_circuit = stim.Circuit()
        self.stim_circuit += self.initialize_stim_circuit(gate_error_1q = gate_error_1q, gate_error_2q = gate_error_2q,measurement_error_rate = measurement_error_rate)
        self.stim_circuit += 1 * self.generate_stim_circuit_cycle(gate_error_1q = gate_error_1q, gate_error_2q = gate_error_2q,
            idle_error_T1 = idle_error_T1, idle_error_T2 = idle_error_T2, measurement_error_rate = measurement_error_rate,detectors = False)
        self.stim_circuit += T * self.generate_stim_circuit_cycle(gate_error_1q = gate_error_1q, gate_error_2q = gate_error_2q,
            idle_error_T1 = idle_error_T1, idle_error_T2 = idle_error_T2, measurement_error_rate = measurement_error_rate)
        self.stim_circuit += self.final_measurement_stim_circuit(gate_error_1q = gate_error_1q, gate_error_2q = gate_error_2q,measurement_error_rate = measurement_error_rate)

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
        # lattice coordinates without semiplaquettes on the boundaries
        lattice_coordinates = [k + i*1j for k in range(-2*self.d+2, 2*self.d, 1) 
                            for i in range(-int((2*self.d-2-abs(k)+(k > 0))/2) - (k%4<=1 and k<0) - (k%4>1 and k>=0),
                                            int((2*self.d-2-abs(k)+(k > 0))/2)+1 + (k%4>1 and k<0) + (k%4<=1 and k>=0))] 

        # centers of hexagons, including the auxiliary triangles (semiplaquettes) i.e., ancilla qubit coordinates in terms of complex numbers
        ancilla_coordinates = [coord for coord in lattice_coordinates if (coord.real%4<=1 and coord.imag%2==1) or (coord.real%4>1 and coord.imag%2==0)]
        # physical qubit coordinates in terms of complex numbers
        qubit_coordinates = [coord for coord in lattice_coordinates if (coord.real%4<=1 and coord.imag%2==0) or (coord.real%4>1 and coord.imag%2==1)]
        
        # preparing the dictionary 
        q2i: Dict[complex, int] = {q: i for i, q in enumerate(self.sorted_complex(qubit_coordinates))}
        a2i: Dict[complex, int] = {a: i for i, a in enumerate(self.sorted_complex(ancilla_coordinates), start=max([i for i in q2i.values()])+1)}

        return [q2i, a2i]

    def initialize_stim_circuit(self,
                            gate_error_1q: float = 0,
                            gate_error_2q: float = 0,
                            measurement_error_rate: float = 0) -> stim.Circuit:
        circuit = stim.Circuit()
        for q, i in self.q2i.items():
                circuit.append("QUBIT_COORDS", [i], [q.real, q.imag])
        for q, i in self.a2i.items():
                circuit.append("QUBIT_COORDS", [i], [q.real, q.imag])
        
        self.stim_circuit.append("R", self.qubit_index_list)
        if measurement_error_rate>0:
            circuit.append('DEPOLARIZE2', self.q2_to_q1_typea, measurement_error_rate)
            circuit.append('DEPOLARIZE2', self.q2_to_q1_typeb, measurement_error_rate)
            circuit.append('DEPOLARIZE1', self.q1_noq2_typeab, measurement_error_rate)
        circuit.append("R", self.ancilla_pairs)
        if measurement_error_rate>0:
            circuit.append('DEPOLARIZE2', self.ancilla_pairs, measurement_error_rate)

        # prepare XX-ZZ mutual +1 eigenstate first
        circuit.append("H", self.q0_typea)
        if gate_error_1q>0:
            circuit.append('DEPOLARIZE1', self.q0_typea, gate_error_1q)
        circuit.append("CNOT", self.q0_to_anc0_typea)
        if gate_error_2q>0:
            circuit.append('DEPOLARIZE2', self.q0_to_anc0_typea, gate_error_2q)
        if self.dedicated_link == "X":
            circuit.append("H_XY", self.q0_to_anc0_typea) # all of them on which the CNOT acted
            if gate_error_1q>0:
                circuit.append('DEPOLARIZE1', self.q0_to_anc0_typea, gate_error_1q)


        if self.logind != (self.linkind + 1) % 3:
            circuit.append(self.Hi_stim[2], self.q0_typea)
            circuit.append(self.Hi_stim[1], self.a0_typea)
            circuit.append(self.Hi_stim[2], self.a0_typea)
            circuit.append(self.Hi_stim[0], self.q0_typeb)
            circuit.append(self.Hi_stim[0], self.q1_typeb)
            if gate_error_1q>0:
                circuit.append('DEPOLARIZE1', self.a0_typea, gate_error_1q)
                if self.Hi_stim[2] != "I":
                    circuit.append('DEPOLARIZE1', self.q0_typea, gate_error_1q)
                if self.Hi_stim[0] != "I":
                    circuit.append('DEPOLARIZE1', self.q0_typeb, gate_error_1q)
                    circuit.append('DEPOLARIZE1', self.q1_typeb, gate_error_1q)

        circuit.append("SWAP",self.q1_to_anc0_typea)
        if gate_error_2q>0:
            circuit.append('DEPOLARIZE2', self.q1_to_anc0_typea, gate_error_2q)
        
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
        
        ## Plaquette measurements
        
        #Map XYZ parity to ancilla
        if self.noSWAP:
            circuit.append("H",self.even_ancilla_index_list) # initialization error accouns for everything...
            circuit.append("CNOT", self.ancilla_pairs) # from even to odd
        for round in range(self.num_pl_rounds):
            for pauli in range(3):
                if self.type1_plaquette_qubits_per_round_per_pauli[round][pauli]!=[]:
                    circuit.append(self.Hi_stim[pauli], self.type1_plaquette_qubits_per_round_per_pauli[round][pauli])
                    if gate_error_1q>0 and self.Hi_stim[pauli]!="I":
                        circuit.append('DEPOLARIZE1', self.type1_plaquette_qubits_per_round_per_pauli[round][pauli], gate_error_1q)
                if self.type2_plaquette_qubits_per_round_per_pauli[round][pauli]!=[]:
                    circuit.append(self.Hi_stim[pauli], self.type2_plaquette_qubits_per_round_per_pauli[round][pauli])
                    if gate_error_1q>0 and self.Hi_stim[pauli]!="I":
                        circuit.append('DEPOLARIZE1', self.type2_plaquette_qubits_per_round_per_pauli[round][pauli], gate_error_1q)
            
            pair_targets_both = self.type1_pair_targets_all[round].copy()
            pair_targets_both.extend(self.type2_pair_targets_all[round]) # makes sure that the scheduling is correct
            circuit.append("CNOT", pair_targets_both)
            if gate_error_2q>0:
                circuit.append('DEPOLARIZE2', pair_targets_both, gate_error_2q)

            for pauli in range(3):
                if self.type1_plaquette_qubits_per_round_per_pauli[round][pauli]!=[]:
                    circuit.append(self.Hi_stim[pauli], self.type1_plaquette_qubits_per_round_per_pauli[round][pauli])
                    if gate_error_1q>0 and self.Hi_stim[pauli]!="I":
                        circuit.append('DEPOLARIZE1', self.type1_plaquette_qubits_per_round_per_pauli[round][pauli], gate_error_1q)
                if self.type2_plaquette_qubits_per_round_per_pauli[round][pauli]!=[]:
                    circuit.append(self.Hi_stim[pauli], self.type2_plaquette_qubits_per_round_per_pauli[round][pauli])
                    if gate_error_1q>0 and self.Hi_stim[pauli]!="I":
                        circuit.append('DEPOLARIZE1', self.type2_plaquette_qubits_per_round_per_pauli[round][pauli], gate_error_1q)

            if self.noSWAP == False:
                if round == 2:
                    circuit.append('SWAP', self.ancilla_pairs) # only type two needs swap
                    if gate_error_2q>0:
                        circuit.append('DEPOLARIZE2', self.ancilla_pairs, gate_error_2q)
                # if round == 0 or round == 4:
                #     circuit.append('SWAP', self.type2_ancilla_pairs) # only type two needs swap
                #     if gate_error_2q>0:
                #         circuit.append('DEPOLARIZE2', self.type2_ancilla_pairs, gate_error_2q)
                if round == 0 or round == 4:
                    circuit.append('SWAP', self.type1_ancilla_pairs) # only type two needs swap
                    if gate_error_2q>0:
                        circuit.append('DEPOLARIZE2', self.type1_ancilla_pairs, gate_error_2q)

        #Measure ancilla
        for k in self.pl_ancilla_index_list:
            edge_key = 'plaquette_'+str(k)
            measurement_times[edge_key] = current_time
            current_time += 1 
        circuit.append("CNOT", self.ancilla_pairs) # from even to odd ancillas
        if measurement_error_rate>0:
            circuit.append('DEPOLARIZE2', self.ancilla_pairs, measurement_error_rate)
        circuit.append("M", self.odd_pl_ancilla_index_list)
        circuit.append("R", self.ancilla_pairs)
        if measurement_error_rate>0:
            circuit.append('DEPOLARIZE2', self.ancilla_pairs, measurement_error_rate)

        # qubit idling errors during measurment
        if idle_error_T1>0:
            circuit.append('PAULI_CHANNEL_1', self.qubit_index_list, [idle_error_T1/2,idle_error_T1/2,0])

        if idle_error_T2>0:
            circuit.append('Z_ERROR', self.qubit_index_list, idle_error_T2)

        # ## Link measurements

        # Map i parity to one of the link qubits
        circuit.append(self.Hi_stim[0], self.qubit_index_list)
        if gate_error_1q>0 and self.Hi_stim[0]!="I":
            circuit.append('DEPOLARIZE1', self.qubit_index_list, gate_error_1q)
        if self.noSWAP:
            circuit.append("H",self.even_ancilla_index_list) # initialization error accouns for everything...
            circuit.append("CNOT", self.ancilla_pairs) # from even to odd
            link0_to_even_anc = [ind for l in range(len(self.links)) for ind in [self.links[l][0],self.even_ancilla_index_list[l]]]
            link1_to_odd_anc = [ind for l in range(len(self.links)) for ind in [self.links[l][1],self.odd_ancilla_index_list[l]]]
            circuit.append("CNOT", link0_to_even_anc)
            if gate_error_2q>0:
                circuit.append('DEPOLARIZE2', link0_to_even_anc, gate_error_2q)
            circuit.append("CNOT", link1_to_odd_anc)
            if gate_error_2q>0:
                circuit.append('DEPOLARIZE2', link1_to_odd_anc, gate_error_2q) 
        else:
            link_to_anc = [ind for l in range(len(self.links)) for i in self.links[l] for ind in [i,self.even_ancilla_index_list[l]]]
            circuit.append("CNOT", link_to_anc)
            if gate_error_2q>0:
                circuit.append('DEPOLARIZE2', link_to_anc, gate_error_2q)        
        circuit.append(self.Hi_stim[0], self.qubit_index_list)
        if gate_error_1q>0 and self.Hi_stim[0]!="I":
            circuit.append('DEPOLARIZE1', self.qubit_index_list, gate_error_1q)
        
        # Measure parity-link qubit
        for k in self.even_ancilla_index_list:
            edge_key = 'link_'+str(k)
            measurement_times[edge_key] = current_time
            current_time += 1 
        circuit.append("CNOT", self.ancilla_pairs) # from even to odd ancillas
        if measurement_error_rate>0:
            circuit.append('DEPOLARIZE2', self.ancilla_pairs, measurement_error_rate)
        circuit.append("M", self.odd_ancilla_index_list)
        circuit.append("R", self.ancilla_pairs)
        if measurement_error_rate>0:
            circuit.append('DEPOLARIZE2', self.ancilla_pairs, measurement_error_rate)

    
        # qubit idling errors during measurment
        if idle_error_T1>0:
            circuit.append('PAULI_CHANNEL_1', self.qubit_index_list, [idle_error_T1/2,idle_error_T1/2,0])
        if idle_error_T2>0:
            circuit.append('Z_ERROR', self.qubit_index_list, idle_error_T2)
        
        measurements_per_cycle = current_time

        
        if detectors:
            ## Determine which sets of measurements to compare in order to get detection events for plaquettes.
            det_circuit = stim.Circuit()
            for k in range(len(self.pl_ancilla_index_list)):
                edge_key = 'plaquette_'+str(self.pl_ancilla_index_list[k])
                record_targets = []
                relative_index = measurement_times[edge_key] - measurements_per_cycle
                record_targets.append(stim.target_rec(relative_index))
                record_targets.append(stim.target_rec(relative_index - measurements_per_cycle))
                circuit.append("DETECTOR", record_targets, [self.pl_ancilla_pos[k].real, self.pl_ancilla_pos[k].imag, 0])
            #Determine which sets of measurements to compare in order to get detection events for links.
            link0_pos = [qpos for qpos in self.qubit_pos if qpos+1 in self.qubit_pos]
            for k in range(0, len(self.even_ancilla_index_list)):
                edge_key = 'link_'+str(self.even_ancilla_index_list[k])
                record_targets = []
                relative_index = measurement_times[edge_key] - measurements_per_cycle
                record_targets.append(stim.target_rec(relative_index))
                record_targets.append(stim.target_rec(relative_index - measurements_per_cycle))
                circuit.append("DETECTOR", record_targets, [link0_pos[k].real, link0_pos[k].imag, 0])
            circuit.append("SHIFT_COORDS", [], [0, 0, 1])
            circuit += det_circuit
        else:
            ## Only compare the measurements that are deterministic
            det_circuit = stim.Circuit()
            for k in range(len(self.pl_ancilla_index_list)):
                edge_key = 'plaquette_'+str(self.pl_ancilla_index_list[k])
                record_targets = []
                relative_index = measurement_times[edge_key] - measurements_per_cycle
                if self.pl_ancilla_index_list[k] in self.typea_pl_ancilla_index_list:                
                    record_targets.append(stim.target_rec(relative_index))
                elif self.pl_ancilla_index_list[k] in self.typeb_ancilla_index_list:
                    ## compare it with itself... to keep the indexing of the detectors correct
                    record_targets.append(stim.target_rec(relative_index))
                    record_targets.append(stim.target_rec(relative_index))
                circuit.append("DETECTOR", record_targets, [self.pl_ancilla_pos[k].real, self.pl_ancilla_pos[k].imag, 0])
            ## every link is deterministic initially
            link0_pos = [qpos for qpos in self.qubit_pos if qpos+1 in self.qubit_pos]
            for k in range(0, len(self.even_ancilla_index_list)):
                edge_key = 'link_'+str(self.even_ancilla_index_list[k])
                record_targets = []
                relative_index = measurement_times[edge_key] - measurements_per_cycle
                record_targets.append(stim.target_rec(relative_index))
                circuit.append("DETECTOR", record_targets, [link0_pos[k].real, link0_pos[k].imag, 0])
            circuit.append("SHIFT_COORDS", [], [0, 0, 1])
            circuit += det_circuit            
        
        return circuit

    def final_measurement_stim_circuit(self,
                            gate_error_1q: float = 0,
                            gate_error_2q: float = 0,
                            measurement_error_rate: float = 0) -> stim.Circuit:
        circuit = stim.Circuit()

        measurement_times: Dict[str, int] = {}
        current_time = 0

        ## part of type2 Z0 and Z1 measurements on type2s
        ## Z1s are measured on the qubits, Z0s on the ancillas        
        circuit.append(self.Hi_stim[0], self.q0_typeb)
        if gate_error_1q>0 and self.Hi_stim[0]!="I":
            circuit.append('DEPOLARIZE1', self.q0_typeb, gate_error_1q)
        circuit.append(self.Hi_stim[0], self.q1_typeb)
        if gate_error_1q>0 and self.Hi_stim[0]!="I":
            circuit.append('DEPOLARIZE1', self.q1_typeb, gate_error_1q)
        ## part of type1 XY and ZZ measurements on type1s
        ## XYs are measured on the qubits, ZZs on the ancillas
        circuit.append(self.Hi_stim[0], self.q0_typea)
        if gate_error_1q>0 and self.Hi_stim[0]!="I":
            circuit.append('DEPOLARIZE1', self.q0_typea, gate_error_1q)
        circuit.append(self.Hi_stim[0], self.q1_typea)
        if gate_error_1q>0 and self.Hi_stim[0]!="I":
            circuit.append('DEPOLARIZE1', self.q1_typea, gate_error_1q)        

        circuit.append("CNOT",self.q0_to_anc0_typea)
        if gate_error_2q>0:
            circuit.append('DEPOLARIZE2', self.q0_to_anc0_typea, gate_error_2q)
        circuit.append("CNOT",self.q1_to_anc0_typea)
        if gate_error_2q>0:
            circuit.append('DEPOLARIZE2', self.q1_to_anc0_typea, gate_error_2q)
        circuit.append(self.Hi_stim[0], self.q0_typea)
        circuit.append(self.Hi_stim[1], self.q0_typea)
        if gate_error_1q>0:
            circuit.append('DEPOLARIZE1', self.q0_typea, gate_error_1q)
        circuit.append(self.Hi_stim[0], self.q1_typea)
        circuit.append(self.Hi_stim[2], self.q1_typea)
        if gate_error_1q>0:
            circuit.append('DEPOLARIZE1', self.q1_typea, gate_error_1q)
        circuit.append("SWAP", self.anc0_to_anc1_typea) # order does not matter for the SWAPs
        if gate_error_2q>0:
            circuit.append('DEPOLARIZE2', self.anc0_to_anc1_typea, gate_error_2q)
        circuit.append("SWAP", self.q0_to_anc0_typea)
        if gate_error_2q>0:
            circuit.append('DEPOLARIZE2', self.q0_to_anc0_typea, gate_error_2q)
        ## part of type2 Z0 and Z1 measurements on type2s
        circuit.append("SWAP", self.q0_to_anc0_typeb) # makes sure that q2 on typea-s is just an ancilla
        if gate_error_2q>0:
            circuit.append('DEPOLARIZE2', self.q0_to_anc0_typeb, gate_error_2q)
        ## type1 XY and ZZ measurements on type1s
        circuit.append("SWAP", self.q2_to_q1_typea) # provided q2 exists, it is empty due to the exchange with an ancilla
        if gate_error_2q>0:
            circuit.append('DEPOLARIZE2', self.q2_to_q1_typea, gate_error_2q)
            circuit.append('DEPOLARIZE1', [self.q1_typea[i] for i in self.q2_nonex_listind], 12/15*gate_error_2q)
        circuit.append("SWAP", self.q1_to_anc0_typea)
        if gate_error_2q>0:
            circuit.append('DEPOLARIZE2', self.q1_to_anc0_typea, gate_error_2q) 
        circuit.append("CNOT", self.q2_to_q1_typea) # provided q2 exists
        if measurement_error_rate>0:
            circuit.append('DEPOLARIZE2', self.q2_to_q1_typea, measurement_error_rate)
        circuit.append("CNOT", [q for i in self.q2_nonex_listind for q in [self.a0_typea[i],self.q1_typea[i]]])
        if measurement_error_rate>0:
            circuit.append('DEPOLARIZE2', [q for i in self.q2_nonex_listind for q in [self.a0_typea[i],self.q1_typea[i]]], measurement_error_rate)
        circuit.append("R", [self.a0_typea[i] for i in self.q2_nonex_listind]) # reset ancillas where there was no q2 as a 'buffer'
        
        for k in self.q1_typea:
            edge_key = 'XY_on_qubit_'+str(k)
            measurement_times[edge_key] = current_time
            current_time += 1         
        if measurement_error_rate>0:
            circuit.append("X_ERROR", self.q1_typea, measurement_error_rate) # readout assignment error
        circuit.append("M", self.q1_typea)        

        circuit.append("CNOT", self.anc0_to_anc1_typea) # from even to odd ancillas
        if measurement_error_rate>0:
            circuit.append('DEPOLARIZE2', self.anc0_to_anc1_typea, measurement_error_rate)        
        for k in self.typea_odd_ancilla_index_list: # only for the indexing
            edge_key = 'ZZ_on_ancilla_'+str(k)
            measurement_times[edge_key] = current_time
            current_time += 1 
        if measurement_error_rate>0:
            circuit.append("X_ERROR", self.a1_typea, measurement_error_rate) # readout assignment error
        circuit.append("M", self.a1_typea)

        circuit.append("CNOT", self.anc0_to_anc1_typeb) # from even to odd ancillas
        if measurement_error_rate>0:
            circuit.append('DEPOLARIZE2', self.anc0_to_anc1_typeb, measurement_error_rate)
        for k in self.typeb_odd_ancilla_index_list:
            edge_key = 'Z0_on_ancilla_'+str(k)
            measurement_times[edge_key] = current_time
            current_time += 1         
        if measurement_error_rate>0:
            circuit.append("X_ERROR", self.a1_typeb, measurement_error_rate) # readout assignment error    
        circuit.append("M", self.a1_typeb)

        circuit.append("CNOT", self.q2_to_q1_typeb) # from even to odd qubits
        if measurement_error_rate>0:
            circuit.append('DEPOLARIZE2', self.q2_to_q1_typeb, measurement_error_rate)
            circuit.append('DEPOLARIZE1', [q for i,q in enumerate(self.q1_typeb) if self.q2_typeb[i]==-1], 12/15*gate_error_2q)        
        for k in self.q1_typeb:
            edge_key = 'Z1_on_qubit_'+str(k)
            measurement_times[edge_key] = current_time
            current_time += 1 
        if measurement_error_rate>0:
            circuit.append("X_ERROR", self.q1_typeb, measurement_error_rate) # readout assignment error
        circuit.append("M", self.q1_typeb)
        
        measurements_since_last_cycle = current_time

        ## Inferring type1 plaquette and every link stabilizer eigenvalue
        # print(measurement_times)
        typea_plaquette_rec_targets = []
        for apos in self.typea_odd_ancilla_pos:
            if apos.imag != self.d:
                plaquette_rec_targets=[]
                if apos-2 in self.qubit_pos:
                    edge_key = 'Z1_on_qubit_'+str(self.q2i[apos-2])
                    # print(edge_key)
                    relative_index = measurement_times[edge_key] - measurements_since_last_cycle
                    plaquette_rec_targets.append(stim.target_rec(relative_index))
                if apos-1j in self.qubit_pos: #neads conversion to YX
                    qind = self.q2i[apos-1j]
                    edge_key = 'XY_on_qubit_'+str(qind)
                    # print(edge_key)
                    relative_index = measurement_times[edge_key] - measurements_since_last_cycle
                    plaquette_rec_targets.append(stim.target_rec(relative_index))
                    indqind = 0
                    while qind != self.q1_typea[indqind]:
                        indqind+=1
                    edge_key = 'ZZ_on_ancilla_'+str(self.typea_odd_ancilla_index_list[indqind])
                    # print(edge_key)
                    relative_index = measurement_times[edge_key] - measurements_since_last_cycle
                    plaquette_rec_targets.append(stim.target_rec(relative_index))
                if apos+1j in self.qubit_pos:
                    edge_key = 'XY_on_qubit_'+str(self.q2i[apos+1j])
                    # print(edge_key)
                    relative_index = measurement_times[edge_key] - measurements_since_last_cycle
                    plaquette_rec_targets.append(stim.target_rec(relative_index))
                if apos+2 in self.qubit_pos:
                    q1_typeb_index = 0
                    while self.q2i[apos+2] != self.q1_typeb[q1_typeb_index]:
                        q1_typeb_index+=1
                    aindZ0 = self.typeb_odd_ancilla_index_list[q1_typeb_index]    
                    edge_key = 'Z0_on_ancilla_'+str(aindZ0)
                    # print(edge_key)
                    relative_index = measurement_times[edge_key] - measurements_since_last_cycle
                    plaquette_rec_targets.append(stim.target_rec(relative_index))
                typea_plaquette_rec_targets.append(plaquette_rec_targets)
                # print()
        
        typea_link_rec_targets = []
        for aind in self.odd_ancilla_index_list:
            link_rec_targets=[]
            if aind in self.typea_odd_ancilla_index_list:
                edge_key = 'ZZ_on_ancilla_'+str(aind)
                relative_index = measurement_times[edge_key] - measurements_since_last_cycle
                link_rec_targets.append(stim.target_rec(relative_index))
            if aind in self.typeb_odd_ancilla_index_list:
                edge_key = 'Z0_on_ancilla_'+str(aind)
                relative_index = measurement_times[edge_key] - measurements_since_last_cycle
                link_rec_targets.append(stim.target_rec(relative_index))
                indaind = 0
                while aind != self.typeb_odd_ancilla_index_list[indaind]:
                    indaind+=1
                qindZ1 = self.q1_typeb[indaind]
                edge_key = 'Z1_on_qubit_'+str(qindZ1)
                relative_index = measurement_times[edge_key] - measurements_since_last_cycle
                link_rec_targets.append(stim.target_rec(relative_index))
            typea_link_rec_targets.append(link_rec_targets)
        
        # Inferring logical observable
        if self.logind == (self.linkind - 1)%3: 
            observable_rec_targets = []
            for qind in self.edge2_qubits:
                if qind in self.q1_typea:
                    edge_key = 'XY_on_qubit_'+str(qind)
                    relative_index = measurement_times[edge_key] - measurements_since_last_cycle
                    observable_rec_targets.append(stim.target_rec(relative_index))
                    indqind = 0
                    while qind != self.q1_typea[indqind]:
                        indqind+=1
                    ancindZZ = self.typea_odd_ancilla_index_list[indqind]                
                    edge_key = 'ZZ_on_ancilla_'+str(ancindZZ)
                    relative_index = measurement_times[edge_key] - measurements_since_last_cycle
                    observable_rec_targets.append(stim.target_rec(relative_index))
                if qind in self.q0_typeb:
                    indqind = 0
                    while qind != self.q0_typeb[indqind]:
                        indqind+=1
                    ancindZ0 = self.typeb_odd_ancilla_index_list[indqind]
                    edge_key = 'Z0_on_ancilla_'+str(ancindZ0)
                    relative_index = measurement_times[edge_key] - measurements_since_last_cycle
                    observable_rec_targets.append(stim.target_rec(relative_index))
        elif self.logind == self.linkind:
            observable_rec_targets = []
            for qind in self.edge1_qubits:
                if qind in self.q1_typea:
                    edge_key = 'XY_on_qubit_'+str(qind)
                    relative_index = measurement_times[edge_key] - measurements_since_last_cycle
                    observable_rec_targets.append(stim.target_rec(relative_index))
                    # indqind = 0
                    # while qind != q1_typea[indqind]:
                    #     indqind+=1
                    # ancindZZ = typea_odd_ancilla_index_list[indqind]                
                    # edge_key = 'ZZ_on_ancilla_'+str(ancindZZ)
                    # relative_index = measurement_times[edge_key] - measurements_since_last_cycle
                    # observable_rec_targets.append(stim.target_rec(relative_index))
                if qind in self.q0_typeb:
                    indqind = 0
                    while qind != self.q0_typeb[indqind]:
                        indqind+=1
                    ancindZ0 = self.typeb_odd_ancilla_index_list[indqind]
                    edge_key = 'Z0_on_ancilla_'+str(ancindZ0)
                    relative_index = measurement_times[edge_key] - measurements_since_last_cycle
                    observable_rec_targets.append(stim.target_rec(relative_index))

        ## Comparing plaquette and link stabilizers to the previous round
        measurement_times: Dict[str, int] = {} #we restore the measurement_time dictionary stabilizer cycles
        current_time = 0
        for k in self.pl_ancilla_index_list:
            edge_key = 'plaquette_'+str(k)
            measurement_times[edge_key] = current_time
            current_time += 1
        for k in self.even_ancilla_index_list:
            edge_key = 'link_'+str(k)
            measurement_times[edge_key] = current_time
            current_time += 1
        measurements_per_cycle = current_time
        
        det_circuit = stim.Circuit()
        typeaind=0
        for k in range(len(self.pl_ancilla_index_list)):
            record_targets = []
            if (self.pl_ancilla_index_list[k] in self.typea_pl_ancilla_index_list):
                edge_key = 'plaquette_'+str(self.pl_ancilla_index_list[k])
                record_targets.extend(typea_plaquette_rec_targets[typeaind])
                typeaind+=1
                relative_index = measurement_times[edge_key] - measurements_per_cycle
                record_targets.append(stim.target_rec(relative_index - measurements_since_last_cycle))
            else:
                edge_key = 'plaquette_'+str(self.pl_ancilla_index_list[k])
                relative_index = measurement_times[edge_key] - measurements_per_cycle
                record_targets.append(stim.target_rec(relative_index - measurements_since_last_cycle))
                record_targets.append(stim.target_rec(relative_index - measurements_since_last_cycle))
            circuit.append("DETECTOR", record_targets, [self.pl_ancilla_pos[k].real, self.pl_ancilla_pos[k].imag, 0])
        link0_pos = [qpos for qpos in self.qubit_pos if qpos+1 in self.qubit_pos]
        for k in range(0, len(self.even_ancilla_index_list)):
            edge_key = 'link_'+str(self.even_ancilla_index_list[k])
            record_targets = []
            relative_index = measurement_times[edge_key] - measurements_per_cycle
            record_targets.extend(typea_link_rec_targets[k])
            record_targets.append(stim.target_rec(relative_index - measurements_since_last_cycle))
            circuit.append("DETECTOR", record_targets, [link0_pos[k].real, link0_pos[k].imag, 0])
        circuit.append("SHIFT_COORDS", [], [0, 0, 1])
        circuit += det_circuit

        ## Adding the observable
        circuit.append("OBSERVABLE_INCLUDE",observable_rec_targets,0)
        
        return circuit

    def syndrome_samples(self, num_shots: int = 1):
        """"Returns one or more syndrome samples of the code circuit"""
        return self.stim_circuit.compile_detector_sampler().sample(num_shots, append_observables=True)

    def decoding_graph2(self) -> nx.Graph:
        """Returns the decoding graph of the code circuit
           Args: matched_edges (bool): if 'True' the edges receive individual labels ('fault_id's) such that decoding by pymatching returns the matched edges, not the logical failure.
        """        
        obsind = self.pauli_to_index(self.logical_observable)
        if obsind == self.linkind:
            logical_to_sign = 1
            relevant_plaquette_pos = [a for a in self.type2_even_ancilla_pos]
            bulk_semiplaquette_pos = [a for i,a in enumerate(self.type2_even_ancilla_pos) if len(self.type2_plaquettes[i])==3 and i!=0 and i!=len(self.type2_even_ancilla_pos)-1]
            link_pos_typea = [l for l in self.qubit_pos if l+1 in self.qubit_pos and [self.q2i[l],self.q2i[l+1]] in self.type2_links]
        elif obsind == (self.linkind + 2) % 3:
            logical_to_sign = -1
            relevant_plaquette_pos = [a for a in self.type1_even_pl_ancilla_pos]
            bulk_semiplaquette_pos = [a for i,a in enumerate(self.type1_even_pl_ancilla_pos) if len(self.type1_plaquettes[i])==3 and i!=0 and i!=len(self.type1_even_pl_ancilla_pos)-1]
            link_pos_typea = [l for l in self.qubit_pos if l+1 in self.qubit_pos and [self.q2i[l],self.q2i[l+1]] in self.type1_links]
        else:
            raise NotImplementedError()

        relevant_plaquette_pos_A = [a for a in relevant_plaquette_pos if a.real < -2.1*logical_to_sign*a.imag]
        # relevant_plaquette_pos_A = [a for a in relevant_plaquette_pos if a.real<0  or (a.real==0 and logical_to_sign*a.imag<0)] # connected to the logical boundary
        
        link_pos = [l for l in self.qubit_pos if l+1 in self.qubit_pos]
        linkA_pos = [l for l in self.qubit_pos if  (l+1 in self.qubit_pos) and l.real < -2.1*logical_to_sign*l.imag]
        
        # print(link_pos,link_pos_typea)

        d = self.d
        T = self.T

        """There should be a way to find the detector index for a given time and stabilizer"""
        num_synd = (2*d**2-1)
        def det_plaq_ind(time, a_pos):
            if a_pos not in self.ancilla_pos or time not in range(T+2):
                return []
            plaq_index = [plind for plind,ai in enumerate(self.pl_ancilla_index_list) if ai==self.a2i[a_pos]]
            if plaq_index==[]:
                return []
            else:
                return time*num_synd + plaq_index[0]
        def det_link_ind(time, q_pos):
            if q_pos not in self.qubit_pos or time not in range(T+2):
                return []
            else:
                link_index = [linkind for linkind,link in enumerate(self.links) if link[0]==self.q2i[q_pos]]
                return time*num_synd + (d**2-1) + link_index[0]

        """Define probabilities for different detection event pairs, independently. 
           Detection events which appear in several single-error-induced syndromes will have a higher probability."""

        prob_XY_1Q = sum([math.comb(2,i)*(self.idle_error_T1)**i*(1-self.idle_error_T1)**(2-i)* 
                            math.comb(4,j)*(2/3*self.gate_error_1q)**j*(1-2/3*self.gate_error_1q)**(4-j)*
                            math.comb(4,k)*(8/15*self.gate_error_2q)**k*(1-8/15*self.gate_error_2q)**(4-k)
                            for i in range(3) for j in range(5) for k in range(5) if (i+j+k)%2==1])
        no2Qgates = 4 - 2*logical_to_sign
        prob_Z_1Q = sum([math.comb(2,i)*(self.idle_error_T2)**i*(1-self.idle_error_T2)**(2-i)* 
                            math.comb(4,j)*(1/3*self.gate_error_1q)**j*(1-1/3*self.gate_error_1q)**(4-j)*
                            math.comb(4,k)*(4/15*self.gate_error_2q)**k*(1-4/15*self.gate_error_2q)**(4-k)*
                            math.comb(no2Qgates,l)*(8/15*self.gate_error_2q)**l*(1-8/15*self.gate_error_2q)**(no2Qgates-l)*
                            math.comb(1,m)*(4/15*self.gate_error_2q)**m*(1-4/15*self.gate_error_2q)**(1-m)
                            for i in range(3) for j in range(5) for k in range(5) for l in range(no2Qgates+1) for m in range(2) if (i+j+k+l+m)%2==1])

        #error propagation from the ancilla to 2 or 3 data qubits
        prob_ancilla_YZ_2Q = 2*(8/15*self.gate_error_2q)*(1-8/15*self.gate_error_2q) #the CNOT fails after round-1 or round-3
        prob_ancilla_YZ_3Q = 2*(8/15*self.gate_error_2q)*(1-8/15*self.gate_error_2q) #either the SWAP or the CNOT fails after round-2
        
        no2Qgates = 8 - logical_to_sign # this is when type1 has the 2 SWAPs
        prob_pl_meas_flicker = sum([math.comb(1,i)*(self.measurement_error_rate)**i*(1-self.measurement_error_rate)**(1-i)* 
                                    math.comb(2,j)*(8/15*self.measurement_error_rate)**j*(1-8/15*self.measurement_error_rate)**(2-j)*
                                    math.comb(no2Qgates,k)*(8/15*self.gate_error_2q)**k*(1-8/15*self.gate_error_2q)**(no2Qgates-k)
                                    for i in range(2) for j in range(3) for k in range(no2Qgates+1) if (i+j+k)%2==1])
        prob_link_meas_flicker = sum([math.comb(1,i)*(self.measurement_error_rate)**i*(1-self.measurement_error_rate)**(1-i)* 
                                    math.comb(2,j)*(8/15*self.measurement_error_rate)**j*(1-8/15*self.measurement_error_rate)**(2-j)*
                                    math.comb(2,k)*(8/15*self.gate_error_2q)**k*(1-8/15*self.gate_error_2q)**(2-k)
                                    for i in range(2) for j in range(3) for k in range(3) if (i+j+k)%2==1])
        g = nx.Graph()
        num_detectors = (T+2)*num_synd
        for k in range(num_detectors):
            g.add_node(k)
        for k in range(2):
            g.add_node(num_detectors+k, is_boundary=True)
        g.add_node(num_detectors+2, is_boundary=True) # getaway node for the other type of anyons

        """Adding the weighted edges"""
        for time in range(T+2):
            for pl_pos in relevant_plaquette_pos:
                if prob_Z_1Q!=0 and (self.gate_error_1q!=0 or time!=0): # at time = 0 there were no measurements yet, and therefore no idling errors assumed
                    # Z-error on the qubit upwards
                    if det_plaq_ind(time=time,a_pos=pl_pos+2j)!=[]:
                        g.add_edge(det_plaq_ind(time=time,a_pos=pl_pos),det_plaq_ind(time=time,a_pos=pl_pos+2j), weight=-math.log(prob_Z_1Q/(1-prob_Z_1Q)), qubit_id=[])
                    if det_plaq_ind(time=time+1,a_pos=pl_pos+2j)!=[] and self.gate_error_1q!=0: # if there are G1 errors, the schedule perimts this syndrome in every round
                        g.add_edge(det_plaq_ind(time=time,a_pos=pl_pos),det_plaq_ind(time=time+1,a_pos=pl_pos+2j), weight=-math.log(prob_Z_1Q/(1-prob_Z_1Q)), qubit_id=[])
                    if det_plaq_ind(time=time,a_pos=pl_pos+2j)==[] and pl_pos in relevant_plaquette_pos_A and pl_pos not in bulk_semiplaquette_pos:
                        g.add_edge(det_plaq_ind(time=time,a_pos=pl_pos),num_detectors, weight=-math.log(prob_Z_1Q/(1-prob_Z_1Q)), qubit_id=[0])
                    if det_plaq_ind(time=time,a_pos=pl_pos+2j)==[] and pl_pos not in relevant_plaquette_pos_A and pl_pos not in bulk_semiplaquette_pos:
                        g.add_edge(det_plaq_ind(time=time,a_pos=pl_pos),num_detectors+1, weight=-math.log(prob_Z_1Q/(1-prob_Z_1Q)), qubit_id=[])
                    # Z-error on the qubit downwards
                    if det_plaq_ind(time=time,a_pos=pl_pos-2j)!=[]:
                        g.add_edge(det_plaq_ind(time=time,a_pos=pl_pos),det_plaq_ind(time=time,a_pos=pl_pos-2j), weight=-math.log(prob_Z_1Q/(1-prob_Z_1Q)), qubit_id=[])
                    if det_plaq_ind(time=time,a_pos=pl_pos-2j)==[] and pl_pos in relevant_plaquette_pos_A and pl_pos not in bulk_semiplaquette_pos:
                        g.add_edge(det_plaq_ind(time=time,a_pos=pl_pos),num_detectors, weight=-math.log(prob_Z_1Q/(1-prob_Z_1Q)), qubit_id=[0])
                    if det_plaq_ind(time=time,a_pos=pl_pos-2j)==[] and pl_pos not in relevant_plaquette_pos_A and pl_pos not in bulk_semiplaquette_pos:
                        g.add_edge(det_plaq_ind(time=time,a_pos=pl_pos),num_detectors+1, weight=-math.log(prob_Z_1Q/(1-prob_Z_1Q)), qubit_id=[])

            for q_pos in link_pos:
                if prob_XY_1Q!=0:
                    if q_pos in link_pos_typea:
                        # XY-error on the qubit upwards, equal time
                        if det_plaq_ind(time=time,a_pos=q_pos+1j)!=[] and (self.gate_error_1q>0 or time!=0):
                            g.add_edge(det_link_ind(time=time,q_pos=q_pos),det_plaq_ind(time=time,a_pos=q_pos+1j), weight=-math.log(prob_XY_1Q/(1-prob_XY_1Q)), qubit_id=[])
                        # XY-error on the qubit upwards, shifted time
                        if det_plaq_ind(time=time+1,a_pos=q_pos+1j)!=[]:
                            g.add_edge(det_link_ind(time=time,q_pos=q_pos),det_plaq_ind(time=time+1,a_pos=q_pos+1j), weight=-math.log(prob_XY_1Q/(1-prob_XY_1Q)), qubit_id=[])
                        if det_plaq_ind(time=time,a_pos=q_pos+1j)==[] and q_pos in linkA_pos:
                            g.add_edge(det_link_ind(time=time,q_pos=q_pos),num_detectors, weight=-math.log(prob_XY_1Q/(1-prob_XY_1Q)), qubit_id=[0])
                        if det_plaq_ind(time=time,a_pos=q_pos+1j)==[] and q_pos not in linkA_pos:
                            g.add_edge(det_link_ind(time=time,q_pos=q_pos),num_detectors+1, weight=-math.log(prob_XY_1Q/(1-prob_XY_1Q)), qubit_id=[])
                        # XY-error on the qubit downwards, equal time
                        if det_plaq_ind(time=time,a_pos=q_pos-1j)!=[] and (self.gate_error_1q>0 or time!=0):
                            g.add_edge(det_link_ind(time=time,q_pos=q_pos),det_plaq_ind(time=time,a_pos=q_pos-1j), weight=-math.log(prob_XY_1Q/(1-prob_XY_1Q)), qubit_id=[])
                        # XY-error on the qubit downwards, shifted time
                        if det_plaq_ind(time=time+1,a_pos=q_pos-1j)!=[]:
                            g.add_edge(det_link_ind(time=time,q_pos=q_pos),det_plaq_ind(time=time+1,a_pos=q_pos-1j), weight=-math.log(prob_XY_1Q/(1-prob_XY_1Q)), qubit_id=[])
                        if det_plaq_ind(time=time,a_pos=q_pos-1j)==[] and q_pos in linkA_pos:
                            g.add_edge(det_link_ind(time=time,q_pos=q_pos),num_detectors, weight=-math.log(prob_XY_1Q/(1-prob_XY_1Q)), qubit_id=[0])
                        if det_plaq_ind(time=time,a_pos=q_pos-1j)==[] and q_pos not in linkA_pos:
                            g.add_edge(det_link_ind(time=time,q_pos=q_pos),num_detectors+1, weight=-math.log(prob_XY_1Q/(1-prob_XY_1Q)), qubit_id=[])
                    else:
                        # XY-error on the qubit to the right, equal time
                        if det_plaq_ind(time=time,a_pos=q_pos+2)!=[] and (self.gate_error_1q>0 or time!=0):
                            g.add_edge(det_link_ind(time=time,q_pos=q_pos),det_plaq_ind(time=time,a_pos=q_pos+2), weight=-math.log(prob_XY_1Q/(1-prob_XY_1Q)), qubit_id=[])
                        # XY-error on the qubit to the right, shifted time
                        if det_plaq_ind(time=time+1,a_pos=q_pos+2)!=[]:
                            g.add_edge(det_link_ind(time=time,q_pos=q_pos),det_plaq_ind(time=time+1,a_pos=q_pos+2), weight=-math.log(prob_XY_1Q/(1-prob_XY_1Q)), qubit_id=[])
                        if det_plaq_ind(time=time,a_pos=q_pos+2)==[] and q_pos in linkA_pos:
                            g.add_edge(det_link_ind(time=time,q_pos=q_pos),num_detectors, weight=-math.log(prob_XY_1Q/(1-prob_XY_1Q)), qubit_id=[0])
                        if det_plaq_ind(time=time,a_pos=q_pos+2)==[] and q_pos not in linkA_pos:
                            g.add_edge(det_link_ind(time=time,q_pos=q_pos),num_detectors+1, weight=-math.log(prob_XY_1Q/(1-prob_XY_1Q)), qubit_id=[])
                        # XY-error on the qubit to the left, equal time
                        if det_plaq_ind(time=time,a_pos=q_pos-2)!=[] and (self.gate_error_1q>0 or time!=0):
                            g.add_edge(det_link_ind(time=time,q_pos=q_pos),det_plaq_ind(time=time,a_pos=q_pos-2), weight=-math.log(prob_XY_1Q/(1-prob_XY_1Q)), qubit_id=[])
                        # XY-error on the qubit to the left, shifted time
                        if det_plaq_ind(time=time+1,a_pos=q_pos-2)!=[]:
                            g.add_edge(det_link_ind(time=time,q_pos=q_pos),det_plaq_ind(time=time+1,a_pos=q_pos-2), weight=-math.log(prob_XY_1Q/(1-prob_XY_1Q)), qubit_id=[])
                        if det_plaq_ind(time=time,a_pos=q_pos-2)==[] and q_pos in linkA_pos:
                            g.add_edge(det_link_ind(time=time,q_pos=q_pos),num_detectors, weight=-math.log(prob_XY_1Q/(1-prob_XY_1Q)), qubit_id=[0])
                        if det_plaq_ind(time=time,a_pos=q_pos-2)==[] and q_pos not in linkA_pos:
                            g.add_edge(det_link_ind(time=time,q_pos=q_pos),num_detectors+1, weight=-math.log(prob_XY_1Q/(1-prob_XY_1Q)), qubit_id=[])
                        ## Note what is missing right now: there are no single qubit gates for Z parity map -> different probablilities needed and too many edges are added now

            for pl_pos in relevant_plaquette_pos:
                if det_plaq_ind(time=time+1, a_pos=pl_pos)!=[] and prob_pl_meas_flicker>0:
                    g.add_edge(det_plaq_ind(time=time,a_pos=pl_pos),det_plaq_ind(time=time+1,a_pos=pl_pos), weight=-math.log(prob_pl_meas_flicker/(1-prob_pl_meas_flicker)), qubit_id=[])
                #also the semiplaquette errors (1Q) should be treated somewhere
                if det_plaq_ind(time=time-1, a_pos=pl_pos+4-logical_to_sign*2j)!=[] and prob_ancilla_YZ_3Q>0: #the syndrome to the left appears already in the same round as the ancillla error
                    g.add_edge(det_plaq_ind(time=time, a_pos=pl_pos),det_plaq_ind(time=time-1, a_pos=pl_pos+4-logical_to_sign*2j), weight=-math.log(prob_ancilla_YZ_3Q/(1-prob_ancilla_YZ_3Q)), qubit_id=[])
                if det_plaq_ind(time=time+1, a_pos=pl_pos-4+logical_to_sign*2j)!=[] and prob_ancilla_YZ_3Q>0: #the syndrome to the left appears already in the same round as the ancillla error
                    g.add_edge(det_plaq_ind(time=time, a_pos=pl_pos),det_plaq_ind(time=time+1, a_pos=pl_pos-4+logical_to_sign*2j), weight=-math.log(prob_ancilla_YZ_3Q/(1-prob_ancilla_YZ_3Q)), qubit_id=[])

            for q_pos in link_pos:
                if det_link_ind(time=time+1, q_pos=q_pos)!=[] and prob_pl_meas_flicker>0:
                    g.add_edge(det_link_ind(time=time,q_pos=q_pos),det_link_ind(time=time+1,q_pos=q_pos), weight=-math.log(prob_link_meas_flicker/(1-prob_link_meas_flicker)), qubit_id=[])
                # induced 2-qubit errors, they do not lead to logical failures, and here we do not connect them to trivial boundaries (it would be just a weight adjustment to an existing edge)
                if det_plaq_ind(time=time, a_pos=pl_pos+2-logical_to_sign*2j)!=[] and prob_ancilla_YZ_2Q>0:
                    g.add_edge(det_link_ind(time=time, a_pos=pl_pos),det_plaq_ind(time=time, a_pos=pl_pos+2-logical_to_sign*2j), weight=-math.log(prob_ancilla_YZ_2Q/(1-prob_ancilla_YZ_2Q)), qubit_id=[])
                if det_plaq_ind(time=time+1, a_pos=pl_pos-2+logical_to_sign*2j)!=[] and prob_ancilla_YZ_2Q>0:
                    g.add_edge(det_link_ind(time=time, a_pos=pl_pos),det_plaq_ind(time=time+1, a_pos=pl_pos-2+logical_to_sign*2j), weight=-math.log(prob_ancilla_YZ_2Q/(1-prob_ancilla_YZ_2Q)), qubit_id=[])
        
        for k in range(num_detectors):
            if g.degree[k] == 0:
                g.add_edge(k, num_detectors + 2, weight=9999999, qubit_id = [])      
        return g


    def matching(self):
        return pymatching.Matching(self.decoding_graph2())
    
    def PredictedObservableOutcome(self, sample: List[bool], m: pymatching.Matching):
        return m.decode(sample)[0]


    def draw_lattice(self, boundaries: bool = True, connectivity: bool = False, schedule: bool = False, links: bool = False, logicals: bool = False, logreadout: bool = False):
        if self.d <= 5:
            lattice = rx.PyGraph()
            q2i, a2i = self.qubit_indices()
            pos =[]
            node_color=[]
            color_palette = ['red','green','blue','purple','black','grey']

            for q in q2i:
                lattice.add_node(q)
                pos.append([q.real+(q.real//2)-(q.real//4)+(q.real%2)-(q.imag//2)-(q.imag%2),
                            q.imag+(q.imag//2)+((q.imag+1)%2)*(q.real//4) + (q.imag%2)*((q.real+2)//4)])
                node_color.append('lightgray')

            for a in a2i:
                lattice.add_node(a)
                pos.append([a.real+(a.real//2)-(a.real//4)-(a.imag//2)-(a.imag%2)+(a.real%4==0)+(a.real%4==1),
                            a.imag+((a.real+1)%2)+(a.imag//2)+((a.imag+1)%2)*(a.real//4) + (a.imag%2)*((a.real+2)//4)])
                node_color.append('gray')

            fig, ax = plt.subplots()

            if logreadout:
                sorted_5pos_typea = []
                for a,b,c,d,e in zip(self.q0_typea,self.q1_typea,self.q2_typea,self.a0_typea,self.a1_typea):
                    if c==-1:
                        q = pos[b][0]+1j*pos[b][1]+1-1j
                        lattice.add_node(q)
                        a2i[q]=lattice.num_nodes()-1
                        pos.append([q.real, q.imag])
                        node_color.append('gray')
                    sorted_5pos_typea.append(sorted([pos[a],pos[b],pos[c],pos[d],pos[e]]))
                
                sorted_5pos_typeb = []
                for a,b,c,d,e in zip(self.q0_typeb,self.q1_typeb,self.q2_typeb,self.a0_typeb,self.a1_typeb):
                    if c==-1:
                        q = pos[b][0]+1j*pos[b][1]+1-1j
                        lattice.add_node(q)
                        a2i[q]=lattice.num_nodes()-1
                        pos.append([q.real, q.imag])
                        node_color.append('gray')
                    sorted_5pos_typeb.append(sorted([pos[a],pos[b],pos[c],pos[d],pos[e]]))

                sorted_5pos_typea_typeb = [sorted_5pos_typea, sorted_5pos_typeb]
            
                for color,sorted_5pos in zip(['deepskyblue','red'],sorted_5pos_typea_typeb):
                    for poslist in sorted_5pos:
                        minx_poslist=[poslist[i] for i in range(len(poslist)) if poslist[i][0] == poslist[0][0]]
                        if len(minx_poslist) == 2:
                            minx_poslist.extend([poslist[i] for i in range(len(poslist)) if poslist[i][0] == poslist[-1][0]])
                            ax.fill_between([minx_poslist[0][0], minx_poslist[2][0]],
                                            [minx_poslist[0][1], minx_poslist[2][1]],
                                            [minx_poslist[1][1], minx_poslist[2][1]],color=color, alpha = 0.2)
                        else:
                            minx_poslist.extend([poslist[i] for i in range(len(poslist)) if poslist[i][0] == poslist[-1][0]])
                            ax.fill_between([minx_poslist[0][0], minx_poslist[1][0]-1, minx_poslist[1][0]],
                                            [minx_poslist[0][1], minx_poslist[1][1]-1, minx_poslist[1][1]],
                                            [minx_poslist[0][1], minx_poslist[0][1], minx_poslist[1][1]],color=color, alpha = 0.2)

            e = 0
            edge_cols = []
            edge_widths = [] 
            e_connectivity = 0               
            if connectivity:
                if schedule:
                    round = 0
                    for pair_color, pair_targets in zip(color_palette,self.type1_pair_targets_all):
                        pind = 0
                        for pairs in [[pair_targets[j], pair_targets[j+1]] for j in range(0,len(pair_targets),2)]:
                            lattice.add_edge(pairs[0],pairs[1],e)
                            e += 1
                            pind += 1
                        edge_cols.extend([pair_color]*pind)
                        edge_widths.extend([1.0]*pind)
                        print(pair_color,': round', round)   
                        round += 1  
                    round = 0
                    for pair_color, pair_targets in zip(color_palette,self.type2_pair_targets_all):
                        pind = 0
                        for pairs in [[pair_targets[j], pair_targets[j+1]] for j in range(0,len(pair_targets),2)]:
                            lattice.add_edge(pairs[0],pairs[1],e)
                            e += 1
                            pind += 1
                        edge_cols.extend([pair_color]*pind)
                        edge_widths.extend([1.0]*pind)
                        round += 1           

                else:
                    pair_targets_all = []
                    pair_targets_all.extend(self.type1_pair_targets_all)
                    pair_targets_all.extend(self.type2_pair_targets_all)
                    for ind,ai in enumerate(self.even_ancilla_index_list):
                        if ai not in self.pl_ancilla_index_list:
                           pair_targets_all.extend([[ai+self.d,self.links[ind][0]],[ai+self.d,self.links[ind][1]]])
                    for pairs in [[pair_targets_all[i][j], pair_targets_all[i][j+1]] for i in range(len(pair_targets_all)) for j in range(0,len(pair_targets_all[i]),2)]:
                        lattice.add_edge(pairs[0],pairs[1],e)
                        e += 1
                    e_connectivity = e
                    edge_cols.extend(['gray']*e_connectivity)
                    edge_widths.extend([1.0]*e_connectivity)
                e_connectivity = e
                for i in range(0,len(self.ancilla_pairs),2):
                    lattice.add_edge(self.ancilla_pairs[i],self.ancilla_pairs[i+1],e)
                    e += 1
                edge_cols.extend(['gray']*(e-e_connectivity))
                edge_widths.extend([3.0]*(e-e_connectivity))
                e_connectivity = e
                for i in range(0,len(self.q2_to_q1_typea),2):
                    lattice.add_edge(self.q2_to_q1_typea[i],self.q2_to_q1_typea[i+1],e)
                    e += 1
                for i in range(0,len(self.q2_to_q1_typeb),2):
                    lattice.add_edge(self.q2_to_q1_typeb[i],self.q2_to_q1_typeb[i+1],e)
                    e += 1
                if logreadout:
                    for a in [a for a in a2i.keys()][-self.d:]:
                        lattice.add_edge(a2i[a], [i for i in q2i.values()][pos.index([a.real-1,a.imag+1])],e)
                        e += 1                
                edge_cols.extend(['lightgray']*(e-e_connectivity))
                edge_widths.extend([3.0]*(e-e_connectivity))
                e_connectivity = e
                mpl_draw(lattice, ax = ax, pos = pos, with_labels=True, node_color = node_color, edge_color = edge_cols, style = 'solid', width = edge_widths)   
                if links:
                    for i in range(len(self.links)):
                        lattice.add_edge(self.links[i][0],self.links[i][1],e)
                        e += 1
                    edge_cols.extend(['green']*(e-e_connectivity))
                    edge_widths.extend([2.0]*(e-e_connectivity))
                e_connectivity = e

            mpl_draw(lattice, ax = ax, pos = pos, with_labels=True, node_color = node_color, edge_color = edge_cols, style = 'dashed', width = edge_widths)   
            
            if schedule:
                for i,color in enumerate(color_palette): 
                    round0, = ax.plot([],label = 'round-'+str(i),c=color)
                ax.legend(loc='upper right')

            if boundaries:
                for plind,pl in enumerate(self.type1_plaquettes):
                    if len(pl) == 6:
                        for pairs in [(pl[i][0],pl[(i+1)%len(pl)][0]) for i in range(6)]:
                            lattice.add_edge(pairs[0],pairs[1],e)
                            e += 1
                        ax.fill_between([pos[pl[3][0]][0],pos[pl[2][0]][0],pos[pl[5][0]][0],pos[pl[1][0]][0]],
                                        [pos[pl[3][0]][1],pos[pl[2][0]][1],pos[pl[2][0]][1],pos[pl[1][0]][1]],
                                        [pos[pl[4][0]][1],pos[pl[4][0]][1],pos[pl[4][0]][1],pos[pl[0][0]][1]],color='red', alpha = 0.2)
                    else:
                        qpos_minx,qpos_midx,qpos_maxx=sorted([pos[pl[0][0]],pos[pl[1][0]],pos[pl[2][0]]])
                        if qpos_minx[0] < 0:
                            ancpos = [qpos_midx[0],qpos_midx[1]-2]
                        else:
                            ancpos = [qpos_midx[0],qpos_midx[1]+2]
                        ax.fill_between([qpos_minx[0],qpos_midx[0],qpos_maxx[0]],
                                        [qpos_minx[1], min(qpos_midx[1],ancpos[1]),qpos_maxx[1]],
                                        [qpos_minx[1], max(qpos_midx[1],ancpos[1]),qpos_maxx[1]],color='red', alpha = 0.2)
                for plind,pl in enumerate(self.type2_plaquettes):
                    if len(pl) == 6:
                        for pairs in [(pl[i][0],pl[(i+1)%len(pl)][0]) for i in range(6)]:
                            lattice.add_edge(pairs[0],pairs[1],e)
                            e += 1
                        ax.fill_between([pos[pl[3][0]][0],pos[pl[2][0]][0],pos[pl[5][0]][0],pos[pl[1][0]][0]],
                                        [pos[pl[3][0]][1],pos[pl[2][0]][1],pos[pl[2][0]][1],pos[pl[1][0]][1]],
                                        [pos[pl[4][0]][1],pos[pl[4][0]][1],pos[pl[4][0]][1],pos[pl[0][0]][1]],color='deepskyblue', alpha = 0.2)
                    else:
                        qpos_minx,qpos_midx,qpos_maxx=sorted([pos[pl[0][0]],pos[pl[1][0]],pos[pl[2][0]]])
                        if qpos_minx[0] < 0:
                            ancpos = [qpos_minx[0],qpos_minx[1]+2]
                        else:
                            ancpos = [qpos_maxx[0],qpos_maxx[1]-2]
                        ancind = [i for i,a in enumerate(pos) if a == ancpos][0]
                        lattice.add_edge(pl[0][0],ancind,e)
                        lattice.add_edge(pl[1][0],ancind,e+1)
                        lattice.add_edge(pl[2][0],ancind,e+2)
                        ax.fill_between([qpos_minx[0],qpos_maxx[0]],
                                        [min(qpos_midx[1],ancpos[1]), min(qpos_midx[1],ancpos[1])],
                                        [max(qpos_midx[1],ancpos[1]), max(qpos_midx[1],ancpos[1])],color='deepskyblue', alpha = 0.2)
                        e += 3
                e_boundary = e - e_connectivity
                e_all = e
                edge_cols.extend(['gray']*e_boundary)
                edge_widths.extend([1.0]*e_boundary)

            # mpl_draw(lattice, ax = ax, pos = pos, node_list = [], edge_color = edge_cols, style = 'dotted', width = edge_widths)
            
            if logicals:
                for ind,(x,y) in enumerate(zip([pos[i][0] for i in self.edge1_qubits],[pos[i][1] for i in self.edge1_qubits])):
                    circle1 = plt.Circle((x,y), radius=0.4, edgecolor='blue', fill= False,linewidth=1.5)
                    ax.add_patch(circle1)
                    ax.text(x-0.2,y-1,['X','Y','Z'][(ind-1)%3], color='blue',fontsize=13)
                for ind,(x,y) in enumerate(zip([pos[i][0] for i in self.edge2_qubits],[pos[i][1] for i in self.edge2_qubits])):
                    circle1 = plt.Circle((x,y), radius=0.47, edgecolor='red', fill= False,linewidth=1.5)
                    ax.add_patch(circle1)
                    ax.text(x-0.2,y+0.6,['X','Y','Z'][(1-ind)%3], color='red',fontsize=13)
                ax.axis('equal')

            plt.show()

        else:
            print("Code distance is too large. Try d <= 5.")
