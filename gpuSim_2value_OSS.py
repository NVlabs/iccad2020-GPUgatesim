# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# sys.argv[1] = block name
# sys.argv[2] = testname
# sys.argv[3] = cycles
# sys.argv[4] = clock period
# python code that uses pytorch and DGL.ai packages to perform per cycle, zero delay mode 2 value oblivious simulation with parallelism across cycles and gates.
# suggest run on GPU, need following packages
# command line variables described above.
# script takes as input a lil_matrix graph object, traces of input port and register outputs in array format, the clock period, and the number of cycles to be simulated
# outputs an array/tensor with the simulated values for the combinational logic

import torch as th
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn
from dgl import DGLGraph
import pickle
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
import argparse
import glob, os
import re
from openpyxl import load_workbook
import openpyxl
import pandas as pd
from pandas import DataFrame
from sklearn import preprocessing
from datetime import datetime

BLOCK=sys.argv[1]
TESTNAME=sys.argv[2]
CYCLES=int(sys.argv[3])
#BLOCK = "mul3"
#TESTNAME='random10cycles'
#CYCLES=10

def is_update_edge(edges): return (edges.data['edge_type'] == 2)

def is_prop_edge(edges): return (edges.data['edge_type'] != 2)

#data loading, builds the DGL graph from lil_matrix graph, and loads the input traces into array format
def build_graph(pkl, traces_features, total_cycles):
  data = np.load(pkl, allow_pickle=1) #assume the lil_matrix graph, and the corresponding graph feature fields, are stored in a pkl file
  Adj=data['adjacency_matrix'] #pkl data has adjacency_matrix lil_matrix field, which stores the graph structure
  PinOrder=data['adjacency_pin_ids'] #another lil_matrix stores the pin order of each net connection/graph edge
  EdgeTypes=data['adjacency_edge_types'] #another lil_matrix stores the net connection/edge types. This is used for cutting the graph, and setting register outputs as source nodes, register inputs as sink nodes
  print("pkl loaded")
  #create the graph
  g=dgl.DGLGraph()
  ll=Adj.toarray() ; ee=PinOrder.toarray() ; tt=EdgeTypes.toarray() ; 
  g.add_nodes(ll.shape[0])
  edges=np.argwhere(ll==True) ; src, dst = tuple(zip(*edges)) ; g.add_edges(src, dst) ;
  print("graph created")
  #"global variables"
  edge_orders = np.argwhere(ee > 0)
  g.edges[edge_orders[:,0],edge_orders[:,1]].data['edge_type'] = th.ByteTensor(tt[edge_orders[:,0],edge_orders[:,1]])
  cell_names=data['cell_index'] # this is a dictionary that keeps track of the waveform tensor indexes and their corresponding cell name "row 0 is instance I1234/Z", for example
  cell_types=data['cell_ids'] ; cell_types=th.LongTensor(cell_types) ; g.ndata['cell_type'] = cell_types # create the cell_type node feature
  #start prop graph
  prop=dgl.DGLGraph() ; prop.add_nodes(ll.shape[0]) ; edges=g.filter_edges(is_prop_edge) ; #this cuts the graph into pipe stages
  srcs=g.edges()[0][edges] ; dsts=g.edges()[1][edges] ; prop.add_edges(srcs, dsts) ;
  #add edge features
  prop.edges[srcs,dsts].data['x'] = th.Tensor(ee[srcs,dsts])
  prop.edata['x'] =prop.edata['x'].type(th.ByteTensor)
  #add the node features
  prop.ndata['cell_type'] = cell_types
  print("cut graph done")
  ##graphs done, now load the input features
  cell_names.update( { v:k for k,v in cell_names.items() } ) # create a bi-directional dictionary, we can look up cell instance by row number index, or look up row number index by cell name
  features_array = np.empty([len(g.nodes()), total_cycles], dtype=int) ;
  features_array = np.full((len(g.nodes()), total_cycles), 9) # just initialize everything to 9, which means 'x'
  features_array[len(g.nodes())-1] = np.full((1,total_cycles), 1) # we can make one node always 1 (VDD), for tie hi's in our netlist. Here we designate the last node to be VDD
  features_array[len(g.nodes())-2] = np.full((1,total_cycles), 0) # we can make one node always 0 (GND), for tie lo's in our netlist. Here we designate the 2nd to last node to be GND
  with open(traces_features, 'r') as f: # open the known trace waveform file. Here we stored it as an array. Each line in the file is of format "<cell instance name> <0/1 value at init> <0/1 value at cycle 1
    for line in f:
      signals = line.split(' ', 1)
      reg_name = signals[0]
      features_array[cell_names[reg_name]] = np.fromstring(signals[1].rstrip(), dtype=int, sep=' ')[0:total_cycles] #lookup which row the line belongs to using the bi-directional dictionary, store the per cycle 0/1 values
  features_array=th.ByteTensor(features_array)
  return prop, features_array, cell_names


prop_block, features_block, cell_names_block= build_graph(BLOCK + "_graph_object", "traces_ios_" + BLOCK + "_" + TESTNAME + ".tbl", CYCLES)


# +
#after this step you can query the DGL.ai graph object (prop_block) to see what information is stored in the graph
#you can query features_block, along with cell_names_block to see how the waveforms will be set up
# -

#load the truth tables for the simulation
def dec_to_bin(x,bit_width):
  return bin(x)[2:].zfill(bit_width)

logic_truth_tables = {}

#list all the cell types in your standard cell library. Example sample shown. Ordering of list should match cell_type node feature
#ordering of pin names should match pin order 'x' edge feature
cells_list=[ ("AND2", ['A1', 'A2'], "int(bits[0] and bits[1])"), \
("AND3", ['A1', 'A2', 'A3'], "int(bits[0] and bits[1] and bits[2])"), \
("AO211", ['A1', 'A2', 'B', 'C'], "int((bits[0] and bits[1]) or bits[2] or bits[3])"), \
("AO21", ['A1', 'A2', 'B'], "int((bits[0] and bits[1]) or bits[2])"), \
("AO221", ['A1', 'A2', 'B1', 'B2', 'C'], "int((bits[0] and bits[1]) or (bits[2] and bits[3]) or bits[4])"), \
("AO31", ['A1', 'A2', 'A3', 'B'], "int((bits[0] and bits[1] and bits[2]) or bits[3])"), \
("AO32", ['A1', 'A2', 'A3', 'B1', 'B2'], "int((bits[0] and bits[1] and bits[2]) or (bits[3] and bits[4]))"), \
("AOI21", ['A1', 'A2', 'B'], "int(not((bits[0] and bits[1]) or bits[2]))"), \
("AOI31", ['A1', 'A2', 'A3', 'B'], "int(not((bits[0] and bits[1] and bits[2]) or bits[3]))"), \
("BUF", ['I'], "int(bits[0])"), \
("MUX2", ['I0', 'I1', 'S'], "int((not(bits[2]) and bits[0]) or (bits[2] and bits[1]))"), \
("MUX3", ['I0', 'I1', 'I2', 'S0', 'S1'], "int((not(bits[3]) and not(bits[4]) and bits[0]) or (bits[3] and (not(bits[4])) and bits[1]) or (not(bits[3]) and bits[4] and bits[2]))"), \
("INV", ['I'], "int(not(bits[0]))"), \
("NAND2", ['A1', 'A2'], "int(not(bits[0] and bits[1]))"), \
("NAND3", ['A1', 'A2', 'A3'], "int(not(bits[0] and bits[1] and bits[2]))"), \
("NOR2", ['A1', 'A2'], "int(not(bits[0] or bits[1]))"), \
("NOR3", ['A1', 'A2', 'A3'], "int(not(bits[0] or bits[1] or bits[2]))"), \
("OR2", ['A1', 'A2'], "int(bits[0] or bits[1])"), \
("FA_SUM", ['A', 'B', 'C'], "int(bits[0] ^ bits[1] ^ bits[2])"), \
("FA_CO", ['A', 'B', 'C'], "int((bits[0] and bits[1]) or (bits[2] and (bits[0] ^ bits[1])))"), \
("HA_SUM", ['A', 'B'], "int(bits[0] ^ bits[1])"), \
("HA_CO", ['A', 'B'], "int(bits[0] and bits[1])"), \
("OA21", ['A1', 'A2', 'B'], "int((bits[0] or bits[1]) and bits[2])"), \
("OA31", ['A1', 'A2', 'A3', 'B'], "int((bits[0] or bits[1] or bits[2]) and bits[3])"), \
("OAI21", ['A1', 'A2', 'B'], "int(not((bits[0] or bits[1]) and bits[2]))"), \
("OAI31", ['A1', 'A2', 'A3', 'B'], "int(not((bits[0] or bits[1] or bits[2]) and bits[3]))"), \
("OR3", ['A1', 'A2', 'A3'], "int(bits[0] or bits[1] or bits[2])"), \
("XOR2", ['A1', 'A2'], "int(bits[0] ^ bits[1])"), \
("XOR3", ['A1', 'A2', 'A3'], "int(bits[0] ^ bits[1] ^ bits[2])") ]


#all the following does is translate the 'string' described logic above into array format results of logic evaluation
cell_counter = 0
for cell_info in cells_list:
  cell_name=cell_info[0]
  cell_pins=cell_info[1]
  cell_func=cell_info[2]
  logic_truth_tables[cell_name]={}
  logic_truth_tables[cell_name]['pins']=cell_pins
  logic_truth_tables[cell_name]['cell_id']=cell_counter ; cell_counter+=1 ;
  truth_table = np.zeros(shape=(2**len(logic_truth_tables[cell_name]['pins']),len(logic_truth_tables[cell_name]['pins'])+1))
  for i in range(2**len(logic_truth_tables[cell_name]['pins'])):
    bits=dec_to_bin(i,len(logic_truth_tables[cell_name]['pins']))
    bits=[int(b) for b in str(bits)]
    output = eval(cell_func)
    bits.append(output)
    truth_table[i] = bits
  logic_truth_tables[cell_name]['truth_table']=truth_table

#make the truth table a 2d 'dictionary-like' tensor
out_tables=th.zeros([len(logic_truth_tables.keys()), 32], dtype=th.uint8)
for cell_type in logic_truth_tables.keys():
  out_tables[logic_truth_tables[cell_type]['cell_id'],0:len(logic_truth_tables[cell_type]['truth_table'][:,-1])]=th.ByteTensor(logic_truth_tables[cell_type]['truth_table'][:,-1])


print(out_tables.element_size())
#out_tables houses ALL the logic in the netlist.
#each row correponds to a different gate. for example, AND2 is row 0. 
#each node in the graph has a 'cell_type' node feature. This matches the row number in out_tables.
#For example, an AND2 gate node will have 'cell_type'=0. This way, the simulator knows the graph node is an AND2 gate,
#and will reference row 0 in out_tables to get the output pin value
#each column in out_tables corresponds to an entry in each gate type's logic truth table. 
#For example, row 0, column 0, correponds to when AND2 gate inputs are [0,0].
#row 0, column 1, corresponds to when AND2 gate inputs are [0,1]
#row 0, column 2, corresponds to when AND2 gate inputs are [1,0]
#row 0, column 3, corresponds to when AND2 gate inputs are [1,1]
#which input edge corresponds to which pin order in the logic truth table is noted by edge feature 'x' in the graph
#in this way, if we know the inputs to the gate, then we simply to lookup table lookup to get the output value


#set up the simulator
#calculate which column we should use when doing the LUT lookup
#'h' is just a node feature created during actually running the simulation that stores the combinational logic waveforms
def gcn_msg(edges):
  return {'m' : (th.mul( (2 ** (edges.data['x']-1)), edges.src['h'].permute(1,0) ) ).permute(1,0)} #.size = [edges, cycles]

def gcn_reduce(nodes):
  #'m'.size= [nodes, edges, cycles]
  return { 'h' : th.sum(nodes.mailbox['m'], dim=1)} #.size = [nodes, cycles]

#"activation function is actually just reading the truth tables"
class NodeApplyModule(nn.Module):
  def __init__(self, activation):
    super(NodeApplyModule, self).__init__()
    self.activation = activation
  def forward(self, node):
    h = (self.activation[node.data['cell_type'], node.data['h'].type(th.LongTensor).permute(1,0)]).permute(1,0) #.size = [nodes, cycles]
    return {'h' : h}

class GPUSim(nn.Module):
  def __init__(self, activation):
    super(GPUSim, self).__init__()
    self.apply_mod = NodeApplyModule(activation)
  def forward(self, g, feature):
    g.ndata['h'] = feature
    #this statement levelizes the netlist
    g.prop_nodes(dgl.traversal.topological_nodes_generator(g)[1:], message_func=gcn_msg, reduce_func=gcn_reduce, apply_node_func=self.apply_mod)
    return g.ndata.pop('h')

class EntireNet(nn.Module):
  def __init__(self, activation):
    super(EntireNet, self).__init__()
    self.layers1 = nn.ModuleList([GPUSim(activation)])
  def forward(self, g,feature):
    h = feature #.size = [nodes, cycles]
    for GCNconv1 in self.layers1:
       h = GCNconv1(g, h) #.size = [nodes, cycles]
    return h

#small function that sends the DGL graph to GPU device
def send_graph_to_device(g, device):
  # nodes
  labels = g.node_attr_schemes()
  for l in labels.keys():
    g.ndata[l] = g.ndata.pop(l).to(device, non_blocking=True) 
  # edges
  labels = g.edge_attr_schemes()
  for l in labels.keys():
    g.edata[l] = g.edata.pop(l).to(device, non_blocking=True)
  return g 

#define the GPU we want to use and send the relevant data to the GPU
#FIXME: possibly need to add a function to figure out how much GPU memory is available and stream the traces fed into 
#the GPU. Also possibly need multi GPU integration
device = th.device("cuda:0" if th.cuda.is_available() else "cpu")
out_tables=out_tables.to(device)
netgpu = EntireNet(out_tables)
netgpu=netgpu.to(device)
print(netgpu)
th.cuda.empty_cache()
prop_block=send_graph_to_device(prop_block, device)
features_block=features_block.to(device)


#this is the execution of the actual simulation
#'simulation_results' variable houses the simulation results. .size = [nodes, cycles] --it's just a 2d array.
netgpu.eval()
num_cycles=CYCLES
now=datetime.now()
simulation_results = netgpu(prop_block,features_block)
later=datetime.now()
delta=(later-now).total_seconds()
print("the whole simulation took " + str(delta) + " seconds on the GPU")

# +
#for the example, querying the cell_names_block tells us the output PRODUCT will be row numbers 113 to 118
#so if we query simulation_results[113:119], we can see the results of the integer mul3 is correct
#we should have
#tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
#        [0, 0, 1, 0, 0, 1, 1, 0, 0, 0],
#        [0, 0, 1, 1, 0, 1, 0, 0, 1, 0],
#        [1, 0, 0, 0, 1, 1, 1, 1, 0, 0],
#        [0, 1, 0, 1, 0, 1, 0, 1, 1, 1],
#        [0, 1, 1, 0, 1, 0, 0, 0, 0, 1]], device='cuda:0', dtype=torch.uint8)
