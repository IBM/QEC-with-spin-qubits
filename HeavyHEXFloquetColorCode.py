# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

from typing import List
import pymatching
import networkx as nx
import rustworkx as rx
from rustworkx.visualization import mpl_draw
import matplotlib.pyplot as plt
import stim
import numpy as np

class HeavyHexFloquetColorCode:
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
        self.stim_circuit = self.generate_circuit(distance = d, cycles = T,
                                                  gate_error_1q = gate_error_1q, gate_error_2q = gate_error_2q,
                                                  idle_error_T1 = idle_error_T1, idle_error_T2 = idle_error_T2,
                                                  init_error_rate = measurement_error_rate, measurement_error_rate = measurement_error_rate,
                                                  T2_bias_X_axis = (logical_observable != "Z"), noSWAP = noSWAP)

    def generate_circuit(self,
                        distance: int, cycles: int,
                        gate_error_1q: float = 0,
                        gate_error_2q: float = 0,
                        idle_error_T1: float = 0,
                        idle_error_T2: float = 0,
                        init_error_rate: float = -1,
                        measurement_error_rate: float = 0,
                        T2_bias_X_axis: bool = False,
                        noSWAP: bool = False
                        ) -> stim.Circuit:

        if init_error_rate < 0:
            init_error_rate = measurement_error_rate
        cycles
        L = distance

        Blue_Stabilizer_Measurements = self.Blue_Stabilizers(L=L)
        Green_Stabilizer_Measurements = self.Green_Stabilizers(L=L)
        Red_Stabilizer_Measurements = self.Red_Stabilizers(L=L)

        Initial_Stabilizer_Readout = stim.Circuit()
        Stabilizer_Readout = stim.Circuit()
        Final_Stabilizer_Readout = stim.Circuit()

        Initial_Stabilizer_Readout += self.Measure_Red_Pauli_X_Edges(L=L, error_G1=0, error_G2=0,init_error_rate=0, error_RO=0, noSWAP=noSWAP)
        Initial_Stabilizer_Readout += self.Measure_Green_Pauli_Z_Edges(L=L, error_G2=0,init_error_rate=0, error_RO=0, noSWAP=noSWAP)
        Initial_Stabilizer_Readout += self.Measure_Blue_Pauli_X_Edges(L=L, error_G1=0, error_G2=0,init_error_rate=0, error_RO=0, noSWAP=noSWAP)
        Initial_Stabilizer_Readout += self.Measure_Red_Pauli_Z_Edges(L=L, error_G2=0,init_error_rate=0, error_RO=0, noSWAP=noSWAP)
        Initial_Stabilizer_Readout += self.Measure_Green_Pauli_X_Edges(L=L, error_G1=0, error_G2=0,init_error_rate=0, error_RO=0, noSWAP=noSWAP)
        Initial_Stabilizer_Readout += self.Measure_Blue_Pauli_Z_Edges(L=L, error_G2=0,init_error_rate=0, error_RO=0, noSWAP=noSWAP)
        # Initial_Stabilizer_Readout += self.Idling_Errors(L=L, error_T1=idle_error_T1, error_T2=idle_error_T2, T2_bias_X_axis)
    
        Stabilizer_Readout += self.Measure_Red_Pauli_X_Edges(L=L, error_G1=gate_error_1q, error_G2=gate_error_2q,init_error_rate=init_error_rate,error_RO=measurement_error_rate, noSWAP=noSWAP)
        Stabilizer_Readout += self.Idling_Errors(L=L, error_T1=idle_error_T1, error_T2=idle_error_T2, T2_bias_X_axis = T2_bias_X_axis)
        Stabilizer_Readout += self.Measure_Green_Pauli_Z_Edges(L=L, error_G2=gate_error_2q,init_error_rate=init_error_rate,error_RO=measurement_error_rate, noSWAP=noSWAP)
        Stabilizer_Readout += self.Idling_Errors(L=L, error_T1=idle_error_T1, error_T2=idle_error_T2, T2_bias_X_axis = T2_bias_X_axis)
        Stabilizer_Readout += Blue_Stabilizer_Measurements
        Stabilizer_Readout += self.Measure_Blue_Pauli_X_Edges(L=L, error_G1=gate_error_1q, error_G2=gate_error_2q,init_error_rate=init_error_rate,error_RO=measurement_error_rate, noSWAP=noSWAP)
        Stabilizer_Readout += self.Idling_Errors(L=L, error_T1=idle_error_T1, error_T2=idle_error_T2, T2_bias_X_axis = T2_bias_X_axis)
        Stabilizer_Readout += self.Measure_Red_Pauli_Z_Edges(L=L, error_G2=gate_error_2q,init_error_rate=init_error_rate,error_RO=measurement_error_rate, noSWAP=noSWAP)
        Stabilizer_Readout += self.Idling_Errors(L=L, error_T1=idle_error_T1, error_T2=idle_error_T2, T2_bias_X_axis = T2_bias_X_axis)
        Stabilizer_Readout += Green_Stabilizer_Measurements
        Stabilizer_Readout += self.Measure_Green_Pauli_X_Edges(L=L, error_G1=gate_error_1q, error_G2=gate_error_2q,init_error_rate=init_error_rate,error_RO=measurement_error_rate, noSWAP=noSWAP)
        Stabilizer_Readout += self.Idling_Errors(L=L, error_T1=idle_error_T1, error_T2=idle_error_T2,  T2_bias_X_axis = T2_bias_X_axis)
        Stabilizer_Readout += self.Measure_Blue_Pauli_Z_Edges(L=L, error_G2=gate_error_2q,init_error_rate=init_error_rate,error_RO=measurement_error_rate, noSWAP=noSWAP)
        Stabilizer_Readout += self.Idling_Errors(L=L,error_T1=idle_error_T1, error_T2=idle_error_T2, T2_bias_X_axis = T2_bias_X_axis)
        Stabilizer_Readout += Red_Stabilizer_Measurements

        Final_Stabilizer_Readout += self.Measure_Red_Pauli_X_Edges(L=L, error_G1=0, error_G2=0,init_error_rate=0, error_RO=0, noSWAP=noSWAP)
        Final_Stabilizer_Readout += self.Measure_Green_Pauli_Z_Edges(L=L, error_G2=0,init_error_rate=0, error_RO=0, noSWAP=noSWAP)
        Final_Stabilizer_Readout += Blue_Stabilizer_Measurements
        Final_Stabilizer_Readout += self.Measure_Blue_Pauli_X_Edges(L=L, error_G1=0, error_G2=0,init_error_rate=0, error_RO=0, noSWAP=noSWAP)
        Final_Stabilizer_Readout += self.Measure_Red_Pauli_Z_Edges(L=L, error_G2=0,init_error_rate=0, error_RO=0, noSWAP=noSWAP)
        Final_Stabilizer_Readout += Green_Stabilizer_Measurements
        Final_Stabilizer_Readout += self.Measure_Green_Pauli_X_Edges(L=L, error_G1=0, error_G2=0,init_error_rate=0, error_RO=0, noSWAP=noSWAP)
        Final_Stabilizer_Readout += self.Measure_Blue_Pauli_Z_Edges(L=L, error_G2=0,init_error_rate=0, error_RO=0, noSWAP=noSWAP)
        Final_Stabilizer_Readout += Red_Stabilizer_Measurements

        Logical_RO = stim.Circuit()
        Logical_RO += self.Logical_Readout(Rounds=cycles, L=L)

        Error_Correction_Circuit = stim.Circuit()
        Error_Correction_Circuit += Initial_Stabilizer_Readout
        Error_Correction_Circuit += Stabilizer_Readout * cycles
        Error_Correction_Circuit += Final_Stabilizer_Readout
        Error_Correction_Circuit += Logical_RO

        return Error_Correction_Circuit


    # Define functions to get the red hexagons coordinates.
    # HexX gives the X coordinate of a red hexagon
    def HexX(self,HexNumber, L):
        return HexNumber % L

    # HexY gives the Y coordinate of a red hexagon
    def HexY(self,HexNumber, L):
        return HexNumber // L

    # Function to find the hexagon to the right of a given hexagon
    def HexLeftNeighbour(self,HexNumber, L):
        X = self.HexX(HexNumber, L)
        Y = self.HexY(HexNumber, L)
        X = (X - 1) % L
        return L * Y + X

    # Function to find the hexagon to the right of a given hexagon
    def HexRightNeighbour(self,HexNumber, L):
        X = self.HexX(HexNumber, L)
        Y = self.HexY(HexNumber, L)
        X = (X + 1) % L
        return L * Y + X

    # Function to find the hexagon above and to the right of a given hexagon
    def HexTopNeighbour(self,HexNumber, L):
        X = self.HexX(HexNumber, L)
        Y = self.HexY(HexNumber, L)
        Y = (Y - 1) % L
        return L * Y + X

    # Function to find the hexagon above and to the right of a given hexagon
    def HexTopRightNeighbour(self,HexNumber, L):
        X = self.HexX(HexNumber, L)
        Y = self.HexY(HexNumber, L)
        X = (X + 1) % L
        Y = (Y - 1) % L
        return L * Y + X

    # Function to find the hexagon underneath a given hexagon
    def HexBottomNeighbour(self,HexNumber, L):
        X = self.HexX(HexNumber, L)
        Y = self.HexY(HexNumber, L)
        Y = (Y + 1) % L
        return L * Y + X

    # Function to find a hexagon below and to the left
    def HexBottomLeftNeighbour(self,HexNumber, L):
        X = self.HexX(HexNumber, L)
        Y = self.HexY(HexNumber, L)
        X = (X - 1) % L
        Y = (Y + 1) % L
        return L * Y + X

    '''
    Qubits are indexed on red hexagons. There are 15 qubits per hexagon The qubits are indexed as follows

    0 -b- 6-b- 7-b- 1 
    g               g
    17              8
    g               g
    16              9
    g               g
    5               2-r-18-r-19-r-
    b               b
    15             10
    b               b
    14             11
    b               b
    4 -g-13-g-12-g- 3
    r               r
    22             20 
    r               r
    23             21  
    r               r

    '''

    def Red_Edge_CNOT_1_SWAP(self,L,error, noSWAP):
        RedEdgeCNOTpairs1 = []
        RedEdgeSWAPpairs = []
        for Hex in range(L*L):
            Hex1 = self.HexBottomNeighbour(Hex,L)
            Hex2 = Hex
            RedEdgeCNOTpairs1.append([24*Hex + 2, 24*Hex + 18])
            RedEdgeCNOTpairs1.append([24*Hex1 + 0, 24*Hex2 + 21])
            RedEdgeCNOTpairs1.append([24*Hex + 4, 24*Hex + 22])
            RedEdgeSWAPpairs.append([24*Hex + 18, 24*Hex + 19])
            RedEdgeSWAPpairs.append([24*Hex + 20, 24*Hex + 21])
            RedEdgeSWAPpairs.append([24*Hex + 22, 24*Hex + 23])
        Red_Edge_CNOT1 = stim.Circuit()    
        if noSWAP:
            ##XX-ZZ eigenstates
            Red_Edge_CNOT1.append_operation("H", [edges[0] for edges in RedEdgeSWAPpairs])
            Red_Edge_CNOT1.append_operation("CNOT", [q for edges in RedEdgeSWAPpairs for q in edges])
        Red_Edge_CNOT1.append_operation("CNOT", [q for edges in RedEdgeCNOTpairs1 for q in edges])
        if error>0:
            Red_Edge_CNOT1.append_operation("DEPOLARIZE2", [q for edges in RedEdgeCNOTpairs1 for q in edges],error)
        if not noSWAP:
            Red_Edge_CNOT1.append_operation("SWAP", [q for edges in RedEdgeSWAPpairs for q in edges])
            if error>0:
                Red_Edge_CNOT1.append_operation("DEPOLARIZE2", [q for edges in RedEdgeSWAPpairs for q in edges],error)
        return Red_Edge_CNOT1

    def Red_Edge_CNOT_2(self,L,error):
        RedEdgeCNOTpairs2 = []
        for Hex in range(L*L):
            Hex1 = self.HexRightNeighbour(Hex, L)
            Hex2 = Hex
            RedEdgeCNOTpairs2.append([24*Hex1 + 5, 24*Hex2 + 19])
            RedEdgeCNOTpairs2.append([24*Hex + 3, 24*Hex + 20])
            Hex1 = self.HexBottomLeftNeighbour(Hex, L)
            Hex2 = Hex
            RedEdgeCNOTpairs2.append([24*Hex1 + 1, 24*Hex2 + 23])
        Red_Edge_CNOT2 = stim.Circuit()
        Red_Edge_CNOT2.append_operation("CNOT", [q for edges in RedEdgeCNOTpairs2 for q in edges])
        if error>0:
            Red_Edge_CNOT2.append_operation("DEPOLARIZE2", [q for edges in RedEdgeCNOTpairs2 for q in edges],error)
        return Red_Edge_CNOT2

    def Red_Edge_Qubit_Measurement(self,L,init_error_rate,meas_error_rate):
        RedEdgeMeasureSites = []
        RedEdgeAncillapairs = []
        for Hex in range(L*L):
            RedEdgeMeasureSites.append(24*Hex + 19)
            RedEdgeMeasureSites.append(24*Hex + 21)
            RedEdgeMeasureSites.append(24*Hex + 23)
            RedEdgeAncillapairs.append([24*Hex + 18, 24*Hex + 19])
            RedEdgeAncillapairs.append([24*Hex + 20, 24*Hex + 21])
            RedEdgeAncillapairs.append([24*Hex + 22, 24*Hex + 23])
        Red_Edge_Qubit_Measurements = stim.Circuit()
        Red_Edge_Qubit_Measurements.append_operation("CNOT", [q for edges in RedEdgeAncillapairs for q in edges])
        if meas_error_rate>0:
            Red_Edge_Qubit_Measurements.append_operation("DEPOLARIZE2", [q for edges in RedEdgeAncillapairs for q in edges], meas_error_rate)
            Red_Edge_Qubit_Measurements.append_operation("X_ERROR", RedEdgeMeasureSites,meas_error_rate)
        Red_Edge_Qubit_Measurements.append_operation("M", RedEdgeMeasureSites)
        Red_Edge_Qubit_Measurements.append_operation("R", [q for edges in RedEdgeAncillapairs for q in edges])
        if init_error_rate>0:
            Red_Edge_Qubit_Measurements.append_operation("DEPOLARIZE2", [q for edges in RedEdgeAncillapairs for q in edges], init_error_rate)
        return Red_Edge_Qubit_Measurements


    def Green_Edge_CNOT1_SWAP(self,L,error,noSWAP):
        GreenEdgeCNOTpairs1 = []
        GreenEdgeSWAPpairs = []
        for Hex in range(L*L):
            GreenEdgeCNOTpairs1.append([24*Hex + 2, 24*Hex + 9])
            GreenEdgeCNOTpairs1.append([24*Hex + 4, 24*Hex + 13])
            GreenEdgeCNOTpairs1.append([24*Hex + 0, 24*Hex + 17])
            GreenEdgeSWAPpairs.append([24*Hex + 8, 24*Hex + 9])
            GreenEdgeSWAPpairs.append([24*Hex + 12, 24*Hex + 13])
            GreenEdgeSWAPpairs.append([24*Hex + 16, 24*Hex + 17])
        Green_Edge_CNOT1 = stim.Circuit()
        if noSWAP:
            ##XX-ZZ eigenstates    
            Green_Edge_CNOT1.append_operation("H", [edges[0] for edges in GreenEdgeSWAPpairs])
            Green_Edge_CNOT1.append_operation("CNOT", [q for edges in GreenEdgeSWAPpairs for q in edges])
        Green_Edge_CNOT1.append_operation("CNOT", [q for edges in GreenEdgeCNOTpairs1 for q in edges])
        if error>0:
            Green_Edge_CNOT1.append_operation("DEPOLARIZE2", [q for edges in GreenEdgeCNOTpairs1 for q in edges],error)
        if not noSWAP:
            Green_Edge_CNOT1.append_operation("SWAP", [q for edges in GreenEdgeSWAPpairs for q in edges])
            if error>0:
                Green_Edge_CNOT1.append_operation("DEPOLARIZE2", [q for edges in GreenEdgeSWAPpairs for q in edges],error)
        return Green_Edge_CNOT1

    def Green_Edge_CNOT2(self,L,error):
        GreenEdgeCNOTpairs2 = []
        for Hex in range(L*L):
            GreenEdgeCNOTpairs2.append([24*Hex + 1, 24*Hex + 8])
            GreenEdgeCNOTpairs2.append([24*Hex + 3, 24*Hex + 12])
            GreenEdgeCNOTpairs2.append([24*Hex + 5, 24*Hex + 16])
        Green_Edge_CNOT2 = stim.Circuit()
        Green_Edge_CNOT2.append_operation("CNOT", [q for edges in GreenEdgeCNOTpairs2 for q in edges])
        if error>0:
            Green_Edge_CNOT2.append_operation("DEPOLARIZE2", [q for edges in GreenEdgeCNOTpairs2 for q in edges],error)
        return Green_Edge_CNOT2

    def Green_Edge_Qubit_Measurement(self,L,init_error_rate,meas_error_rate):
        GreenEdgeMeasureSites = []
        GreenEdgeAncillapairs = []
        for Hex in range(L*L):
            GreenEdgeMeasureSites.append(24*Hex + 9)
            GreenEdgeMeasureSites.append(24*Hex + 13)
            GreenEdgeMeasureSites.append(24*Hex + 17)
            GreenEdgeAncillapairs.append([24*Hex + 8, 24*Hex + 9])
            GreenEdgeAncillapairs.append([24*Hex + 12, 24*Hex + 13])
            GreenEdgeAncillapairs.append([24*Hex + 16, 24*Hex + 17])
        Green_Edge_Qubit_Measurements = stim.Circuit()
        Green_Edge_Qubit_Measurements.append_operation("CNOT", [q for edges in GreenEdgeAncillapairs for q in edges])
        if meas_error_rate>0:
            Green_Edge_Qubit_Measurements.append_operation("DEPOLARIZE2", [q for edges in GreenEdgeAncillapairs for q in edges], meas_error_rate)
            Green_Edge_Qubit_Measurements.append_operation("X_ERROR", GreenEdgeMeasureSites,meas_error_rate)
        Green_Edge_Qubit_Measurements.append_operation("M", GreenEdgeMeasureSites)
        Green_Edge_Qubit_Measurements.append_operation("R", [q for edges in GreenEdgeAncillapairs for q in edges])
        if init_error_rate>0:
            Green_Edge_Qubit_Measurements.append_operation("DEPOLARIZE2", [q for edges in GreenEdgeAncillapairs for q in edges], init_error_rate)
        return Green_Edge_Qubit_Measurements


    def Blue_Edge_CNOT1_SWAP(self,L,error,noSWAP):
        BlueEdgeCNOTpairs1 = []
        BlueEdgeSWAPpairs = []
        for Hex in range(L*L):
            BlueEdgeCNOTpairs1.append([24*Hex + 0, 24*Hex + 6])
            BlueEdgeCNOTpairs1.append([24*Hex + 2, 24*Hex + 10])
            BlueEdgeCNOTpairs1.append([24*Hex + 4, 24*Hex + 14])
            BlueEdgeSWAPpairs.append([24*Hex + 6, 24*Hex + 7])
            BlueEdgeSWAPpairs.append([24*Hex + 10, 24*Hex + 11])
            BlueEdgeSWAPpairs.append([24*Hex + 14, 24*Hex + 15])
        Blue_Edge_CNOT1 = stim.Circuit()
        if noSWAP:
            ##XX-ZZ eigenstates
            Blue_Edge_CNOT1.append_operation("H", [edges[0] for edges in BlueEdgeSWAPpairs])
            Blue_Edge_CNOT1.append_operation("CNOT", [q for edges in BlueEdgeSWAPpairs for q in edges])
        Blue_Edge_CNOT1.append_operation("CNOT", [q for edges in BlueEdgeCNOTpairs1 for q in edges])
        if error>0:
            Blue_Edge_CNOT1.append_operation("DEPOLARIZE2", [q for edges in BlueEdgeCNOTpairs1 for q in edges],error)
        if not noSWAP:
            Blue_Edge_CNOT1.append_operation("SWAP", [q for edges in BlueEdgeSWAPpairs for q in edges])
            if error>0:
                Blue_Edge_CNOT1.append_operation("DEPOLARIZE2", [q for edges in BlueEdgeSWAPpairs for q in edges],error)
        return Blue_Edge_CNOT1

    def Blue_Edge_CNOT2(self,L,error):
        BlueEdgeCNOTpairs2 = []
        for Hex in range(L*L):
            BlueEdgeCNOTpairs2.append([24*Hex + 1, 24*Hex + 7])
            BlueEdgeCNOTpairs2.append([24*Hex + 3, 24*Hex + 11])
            BlueEdgeCNOTpairs2.append([24*Hex + 5, 24*Hex + 15])
        Blue_Edge_CNOT2 = stim.Circuit()
        Blue_Edge_CNOT2.append_operation("CNOT", [q for edges in BlueEdgeCNOTpairs2 for q in edges])
        if error>0:
            Blue_Edge_CNOT2.append_operation("DEPOLARIZE2", [q for edges in BlueEdgeCNOTpairs2 for q in edges],error)
        return Blue_Edge_CNOT2

    def Blue_Edge_Qubit_Measurement(self,L,init_error_rate,meas_error_rate):
        BlueEdgeMeasureSites = []
        BlueEdgeAncillapairs = []
        for Hex in range(L*L):
            BlueEdgeMeasureSites.append(24*Hex + 7)
            BlueEdgeMeasureSites.append(24*Hex + 11)
            BlueEdgeMeasureSites.append(24*Hex + 15)
            BlueEdgeAncillapairs.append([24*Hex + 6, 24*Hex + 7])
            BlueEdgeAncillapairs.append([24*Hex + 10, 24*Hex + 11])
            BlueEdgeAncillapairs.append([24*Hex + 14, 24*Hex + 15])
        Blue_Edge_Qubit_Measurements = stim.Circuit()
        Blue_Edge_Qubit_Measurements.append_operation("CNOT", [q for edges in BlueEdgeAncillapairs for q in edges])
        if meas_error_rate>0:
            Blue_Edge_Qubit_Measurements.append_operation("DEPOLARIZE2", [q for edges in BlueEdgeAncillapairs for q in edges], meas_error_rate)
            Blue_Edge_Qubit_Measurements.append_operation("X_ERROR", BlueEdgeMeasureSites,meas_error_rate)
        Blue_Edge_Qubit_Measurements.append_operation("M", BlueEdgeMeasureSites)
        Blue_Edge_Qubit_Measurements.append_operation("R", [q for edges in BlueEdgeAncillapairs for q in edges])
        if init_error_rate>0:
            Blue_Edge_Qubit_Measurements.append_operation("DEPOLARIZE2", [q for edges in BlueEdgeAncillapairs for q in edges], init_error_rate)
        return Blue_Edge_Qubit_Measurements

    def Idling_Errors(self,L, error_T1, error_T2, T2_bias_X_axis):
        Idling_Errors = stim.Circuit()
        DataQubits = []
        for k in range(L*L):
            DataQubits.extend(range(24*k,24*k + 6))
        if T2_bias_X_axis:
            if error_T1>0:
                Idling_Errors.append_operation("PAULI_CHANNEL_1", DataQubits, [0,error_T1/2,error_T1/2])
            if error_T2>0:
                Idling_Errors.append_operation("X_ERROR", DataQubits, error_T2)
        else: # if T2_bias_X_axis is false or not specified...
            if error_T1>0:
                Idling_Errors.append_operation("PAULI_CHANNEL_1", DataQubits, [error_T1/2,error_T1/2,0])
            if error_T2>0:
                Idling_Errors.append_operation("Z_ERROR", DataQubits, error_T2)

        return Idling_Errors


    def Measure_Red_Pauli_Z_Edges(self,L, error_G2,init_error_rate,error_RO,noSWAP):
        Red_Z_Edge_Measurements = stim.Circuit()
        Red_Z_Edge_Measurements += self.Red_Edge_CNOT_1_SWAP(L, error_G2,noSWAP)
        Red_Z_Edge_Measurements += self.Red_Edge_CNOT_2(L, error_G2)
        Red_Z_Edge_Measurements += self.Red_Edge_Qubit_Measurement(L,init_error_rate,error_RO)
        return Red_Z_Edge_Measurements

    def Measure_Green_Pauli_Z_Edges(self,L, error_G2,init_error_rate,error_RO,noSWAP):
        Green_Z_Edge_Measurements = stim.Circuit()
        Green_Z_Edge_Measurements += self.Green_Edge_CNOT1_SWAP(L, error_G2,noSWAP)
        Green_Z_Edge_Measurements += self.Green_Edge_CNOT2(L, error_G2)
        Green_Z_Edge_Measurements += self.Green_Edge_Qubit_Measurement(L,init_error_rate,error_RO)
        return Green_Z_Edge_Measurements

    def Measure_Blue_Pauli_Z_Edges(self,L, error_G2,init_error_rate,error_RO,noSWAP):
        Blue_Z_Edge_Measurements = stim.Circuit()
        Blue_Z_Edge_Measurements += self.Blue_Edge_CNOT1_SWAP(L, error_G2,noSWAP)
        Blue_Z_Edge_Measurements += self.Blue_Edge_CNOT2(L, error_G2)
        Blue_Z_Edge_Measurements += self.Blue_Edge_Qubit_Measurement(L,init_error_rate,error_RO)
        return Blue_Z_Edge_Measurements

    def Global_Hadamard(self,L,error):
        DataQubits = []
        for k in range(L*L):
            DataQubits.extend(range(24*k, 24*k + 6))
        Global_Hadamard = stim.Circuit()
        Global_Hadamard.append_operation("H", DataQubits)
        if error>0:
            Global_Hadamard.append_operation("DEPOLARIZE1", DataQubits, error)
        return Global_Hadamard

    def Measure_Red_Pauli_X_Edges(self,L, error_G1, error_G2,init_error_rate, error_RO,noSWAP):
        Red_X_Edge_Measurements = stim.Circuit()
        Red_X_Edge_Measurements += self.Global_Hadamard(L, error_G1)
        Red_X_Edge_Measurements += self.Measure_Red_Pauli_Z_Edges(L, error_G2,init_error_rate, error_RO,noSWAP)
        Red_X_Edge_Measurements += self.Global_Hadamard(L, error_G1)
        return Red_X_Edge_Measurements

    def Measure_Green_Pauli_X_Edges(self,L, error_G1, error_G2,init_error_rate, error_RO,noSWAP):
        Green_X_Edge_Measurements = stim.Circuit()
        Green_X_Edge_Measurements += self.Global_Hadamard(L, error_G1)
        Green_X_Edge_Measurements += self.Measure_Green_Pauli_Z_Edges(L, error_G2,init_error_rate, error_RO,noSWAP)
        Green_X_Edge_Measurements += self.Global_Hadamard(L, error_G1)
        return Green_X_Edge_Measurements

    def Measure_Blue_Pauli_X_Edges(self,L, error_G1, error_G2,init_error_rate, error_RO,noSWAP):
        Blue_X_Edge_Measurements = stim.Circuit()
        Blue_X_Edge_Measurements += self.Global_Hadamard(L, error_G1)
        Blue_X_Edge_Measurements += self.Measure_Blue_Pauli_Z_Edges(L, error_G2,init_error_rate, error_RO,noSWAP)
        Blue_X_Edge_Measurements += self.Global_Hadamard(L, error_G1)
        return Blue_X_Edge_Measurements



    def Blue_Stabilizers(self,L):
        Blue_Stabilizer_Detector = stim.Circuit()
        for block in range(L*L):
            Red_Edge_1 = -15*L*L + 3*block
            Red_Edge_2 = -15*L*L + 3*self.HexTopRightNeighbour(block, L) + 1
            Red_Edge_3 = -15*L*L + 3*self.HexTopRightNeighbour(block, L) + 2
            Green_Edge_1 = -3*L*L + 3*block
            Green_Edge_2 = -3*L*L + 3*self.HexRightNeighbour(block, L) + 2
            Green_Edge_3 = -3*L*L + 3*self.HexTopRightNeighbour(block, L) + 1
            Blue_Stabilizer_Detector.append_operation("DETECTOR", [stim.target_rec(Red_Edge_1), stim.target_rec(Red_Edge_2),
                                                                stim.target_rec(Red_Edge_3), stim.target_rec(Green_Edge_1),
                                                                stim.target_rec(Green_Edge_2),
                                                                stim.target_rec(Green_Edge_3)])
        return Blue_Stabilizer_Detector

    def Green_Stabilizers(self,L):
        Green_Stabilizer_Detector = stim.Circuit()
        for block in range(L*L):
            Blue_Edge_1 = -15*L*L + 3*block + 1
            Blue_Edge_2 = -15*L*L + 3*self.HexRightNeighbour(block, L) + 2
            Blue_Edge_3 = -15*L*L + 3*self.HexBottomNeighbour(block, L)
            Red_Edge_1 = -3*L*L + 3*block
            Red_Edge_2 = -3*L*L + 3*block + 1
            Red_Edge_3 = -3*L*L + 3*self.HexRightNeighbour(block, L) + 2
            Green_Stabilizer_Detector.append_operation("DETECTOR",
                                                    [stim.target_rec(Blue_Edge_1), stim.target_rec(Blue_Edge_2),
                                                        stim.target_rec(Blue_Edge_3), stim.target_rec(Red_Edge_1),
                                                        stim.target_rec(Red_Edge_2), stim.target_rec(Red_Edge_3)])
        return Green_Stabilizer_Detector

    def Red_Stabilizers(self,L):
        Red_Stabilizer_Detector = stim.Circuit()
        for block in range(L*L):
            Green_Edge_1 = -15*L*L + 3*block
            Green_Edge_2 = -15*L*L + 3*block + 1
            Green_Edge_3 = -15*L*L + 3*block + 2
            Blue_Edge_1 = -3*L*L + 3*block
            Blue_Edge_2 = -3*L*L + 3*block + 1
            Blue_Edge_3 = -3*L*L + 3*block + 2
            Red_Stabilizer_Detector.append_operation("DETECTOR", [stim.target_rec(Green_Edge_1), stim.target_rec(Green_Edge_2),
                                                                stim.target_rec(Green_Edge_3), stim.target_rec(Blue_Edge_1),
                                                                stim.target_rec(Blue_Edge_2), stim.target_rec(Blue_Edge_3)])
        return Red_Stabilizer_Detector

    def Logical_Readout(self,Rounds, L):
        Readout_Circuit = stim.Circuit()
        Readout_Qubits = []
        for qubits in range(L):
            Readout_Qubits.append(24*qubits + 5)
            Readout_Qubits.append(24*qubits + 2)
        Readout_Circuit.append_operation("M", Readout_Qubits)
        Detector_Sites = range(-2*L, 0)
        Record_Targets = []
        for a in Detector_Sites:
            Record_Targets.append(stim.target_rec(a))
        Number_Of_Rows = Rounds + 2
        for row in range(Number_Of_Rows):

            Edge_Locations = []
            Periodic_Row = row % L
            Start_Of_Measurement_Round = -(3 + 18*row)*L*L + 3*L*Periodic_Row - 2*L
            for edge in range(L):
                Edge_Locations.append(Start_Of_Measurement_Round + 3*edge + 2)
                Edge_Locations.append(Start_Of_Measurement_Round + 3*edge + 1)
            for a in Edge_Locations:
                Record_Targets.append(stim.target_rec(a))

            Edge_Locations = []
            Start_Of_Measurement_Round = -(9 + 18*row)*L*L + 3*L*Periodic_Row - 2*L
            for edge in range(L):
                Edge_Locations.append(Start_Of_Measurement_Round + 3 * edge + 2)
                Edge_Locations.append(Start_Of_Measurement_Round + 3 * edge + 1)
            for a in Edge_Locations:
                Record_Targets.append(stim.target_rec(a))

            Edge_Locations = []
            Periodic_Row = (row + 1) % L
            Start_Of_Measurement_Round = -(15 + 18*row)*L*L + 3*L*Periodic_Row - 2*L
            for edge in range(L):
                Edge_Locations.append(Start_Of_Measurement_Round + 3 * edge + 2)
                Edge_Locations.append(Start_Of_Measurement_Round + 3 * edge + 0)
            for a in Edge_Locations:
                Record_Targets.append(stim.target_rec(a))

        Readout_Circuit.append_operation("DETECTOR", Record_Targets)
        return Readout_Circuit

    def Matching_Graph(self, p):
        g = nx.Graph()
        L = self.d
        rounds = L+1
        for height in range(3*L + 1):
            layer = height * L * L
            layer_above = (height + 1) * L * L
            layer_two_above = (height + 2) * L * L
            for hex in range(L * L):
                vertex = layer + hex
                qubit_edge = 6*layer + 3*hex
                
                PHYS = 36 * p / 15
                w = np.log((PHYS * PHYS + (1 - PHYS) * (1 - PHYS)) / (2 * PHYS * (1 - PHYS)))
                if height % 3 == 0:
                    g.add_edge(vertex, layer_above + hex, qubit_id = qubit_edge, weight=w)
                    g.add_edge(vertex, layer_above + self.HexTopNeighbour(hex, L), qubit_id = qubit_edge + 1, weight=w)
                    g.add_edge(vertex, layer_above + self.HexTopRightNeighbour(hex, L), qubit_id = qubit_edge + 2, weight=w)
                elif ((height + 2) % 3 == 0):
                    g.add_edge(vertex, layer_above + self.HexBottomNeighbour(hex, L), qubit_id = qubit_edge, weight=w)
                    g.add_edge(vertex, layer_above + self.HexRightNeighbour(hex, L), qubit_id = qubit_edge + 1, weight=w)
                    g.add_edge(vertex, layer_above + hex, qubit_id = qubit_edge + 2, weight=w)
                elif ((height + 1) % 3 == 0):
                    if height != 3 * rounds + 2:
                        g.add_edge(vertex, layer_above + self.HexBottomLeftNeighbour(hex, L), qubit_id= qubit_edge, weight=w)
                        g.add_edge(vertex, layer_above + self.HexLeftNeighbour(hex, L), qubit_id= qubit_edge + 1, weight=w)
                        g.add_edge(vertex, layer_above + hex, qubit_id= qubit_edge + 2, weight=w)
                qubit_edge = 6*layer + 3*L*L + 3*hex
                
                q = 38*p / 15
                w = np.log((1 - q) / q)
                if height % 3 == 0:
                    g.add_edge(vertex, layer_two_above + hex, qubit_id= qubit_edge, weight=w)
                    g.add_edge(vertex, layer_two_above + self.HexRightNeighbour(hex, L), qubit_id= qubit_edge + 1, weight=w)
                    g.add_edge(vertex, layer_two_above + self.HexTopRightNeighbour(hex, L), qubit_id= qubit_edge + 2, weight=w)
                elif ((height + 2) % 3 == 0):
                    if height != (3 * rounds + 1):
                        g.add_edge(vertex, layer_two_above + self.HexBottomNeighbour(hex, L), qubit_id= qubit_edge, weight=w)
                        g.add_edge(vertex, layer_two_above + self.HexBottomLeftNeighbour(hex, L), qubit_id= qubit_edge + 1, weight=w)
                        g.add_edge(vertex, layer_two_above + hex, qubit_id= qubit_edge + 2, weight=w)
                elif ((height + 1) % 3 == 0):
                    if height != 3 * rounds + 2:
                        g.add_edge(vertex, layer_two_above + hex, qubit_id= qubit_edge, weight=w)
                        g.add_edge(vertex, layer_two_above + self.HexLeftNeighbour(hex, L), qubit_id= qubit_edge + 1, weight=w)
                        g.add_edge(vertex, layer_two_above + self.HexTopNeighbour(hex, L), qubit_id= qubit_edge + 2, weight=w)
                else:
                    print("ERROR: layer does not exist")
        return g

    def matching(self):
        return pymatching.Matching(self.Matching_Graph(p = 0.01)) # p = 0.01 only for simplicity, because the decoder is not accounting for noise biases

    def PredictedObservableOutcome(self, sample: List[bool], m: pymatching.Matching):
        L = self.d
        height = L + 1
        No_Of_Flips_On_Logical = 0
        Logical_Support = []
        s = np.delete(sample, -1)
        c = m.decode(s)
        for layer in range(height):
            periodic_row = (L + 1 - layer) % L
            for hex in range(L):
                Logical_Support.append(18*L*L*layer + 3*L*periodic_row + 3*hex)
            for hex in range(L):
                Logical_Support.append(18*L*L*layer + 3*L*L + 3*L*periodic_row + 3*hex)
                Logical_Support.append(18*L*L*layer + 3*L*L + 3*L*periodic_row + 3*hex + 1)
            periodic_row = (L - layer) % L
            if layer != L:
                for hex in range(L):
                    Logical_Support.append(18*L*L*layer + 6*L*L + 3*L*periodic_row + 3*hex)
                for hex in range(L):
                    Logical_Support.append(18*L*L*layer + 9*L*L + 3*L*periodic_row + 3*hex)
                    Logical_Support.append(18*L*L*layer + 9*L*L + 3*L*periodic_row + 3*hex + 1)
                for hex in range(L):
                    Logical_Support.append(18*L*L*layer + 12*L*L + 3*L*periodic_row + 3*hex)
                for hex in range(L):
                    Logical_Support.append(18*L*L*layer + 15*L*L + 3*L*periodic_row + 3*hex)
                    Logical_Support.append(18*L*L*layer + 15*L*L + 3*L*periodic_row + 3*hex + 1)
        for site in Logical_Support:
            No_Of_Flips_On_Logical += c[site]
        No_Of_Flips_On_Logical = No_Of_Flips_On_Logical % 2

        return No_Of_Flips_On_Logical

    def draw_lattice(self, d: int=2):
        if d != 2:
            print('only d = 2 is available')

        lattice = rx.PyGraph()
        pos =[]
        node_color=[]

        q2i =  {(2+0j): 0,(2+3j): 1,(2+6j): 2,(6+0j): 3,(6+3j): 4,(6+6j): 5}
        ancA2i = {(2+1j): 6,(2+4j): 7,(2+7j): 8,(3+0j): 9,(3+6j): 10,(6+1j): 11,(6+4j): 12,(6+7j): 13,(8+3j): 14}
        ancB2i = {(2+2j): 15,(2+5j): 16,(2+8j): 17,(5+0j): 18,(5+6j): 19,(6+2j): 20,(6+5j): 21,(6+8j): 22,(7+3j): 23}

        for q in q2i:
            lattice.add_node(q)
            pos.append([q.imag,-q.real])
            node_color.append('lightgray')
        for a in ancA2i:
            lattice.add_node(a)
            pos.append([a.imag,-a.real])
            node_color.append('gray')
        for a in ancB2i:
            lattice.add_node(a)
            pos.append([a.imag,-a.real])
            node_color.append('gray')

        e = 0
        edge_cols = []
        edge_widths = []

        for pairs in zip(ancA2i.values(),ancB2i.values()):
            lattice.add_edge(pairs[0],pairs[1],e)
            e+=1
        edge_cols.extend(['gray']*e)
        edge_widths.extend([4.0]*e)

        for pairs in [[2,8],[4,23],[5,13]]:
            lattice.add_edge(pairs[0],pairs[1],e)
        edge_cols.extend(['red']*3)
        edge_widths.extend([2.0]*3)
        for pairs in [[2,10],[5,19],[3,11],[4,20],[0,6],[1,15]]:
            lattice.add_edge(pairs[0],pairs[1],e)
        edge_cols.extend(['green']*6)
        edge_widths.extend([2.0]*6)
        for pairs in [[1,7],[2,16],[0,9],[3,18],[4,12],[5,21]]:
            lattice.add_edge(pairs[0],pairs[1],e)
        edge_cols.extend(['blue']*6)
        edge_widths.extend([2.0]*6)

        fig, ax = plt.subplots()
        ax.fill_between([pos[0][0],pos[2][0]],[pos[3][1],pos[5][1]],[pos[0][1],pos[2][1]],color='red', alpha = 0.2)
        ax.fill_between([pos[4][0],pos[22][0]+1],[pos[14][1]-1,pos[14][1]-1],[pos[4][1],pos[4][1]],color='green', alpha = 0.2)
        ax.fill_between([pos[2][0],pos[17][0]+1],[pos[5][1],pos[5][1]],[pos[2][1],pos[2][1]],color='blue', alpha = 0.2)
        ax.fill_between([pos[3][0],pos[4][0]],[pos[14][1]-1,pos[14][1]-1],[pos[3][1],pos[3][1]],color='blue', alpha = 0.2)

        mpl_draw(lattice, ax = ax, pos = pos, with_labels=True, node_color = node_color, edge_color = edge_cols, style = 'solid', width = edge_widths)   

        plt.show()
