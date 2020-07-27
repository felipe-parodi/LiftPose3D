import networkx as nx

'Nose-Head-Neck-RShoulder-RHand-LShoulder-LHand- Hip-RKnee-RFoot-LKnee-LFoot'Hip'

'''
Joints
------
0:  NOSE,    
1:  Head,
2:  Neck,
3:  RShoulder,
4:  RHand,

5:  LShoulder,
6:  LHand,
7:  Hip,
8:  RKnee,
9:  RFoot, 
    
10: LKnee,
11: LFoot, 
12: Hip,

'''

def skeleton():
    edges = [[(0,1),(1,2),(2,3),(3,4),(2,5),(5,6),(2,7),(7,8),(8,9),(7,10),(10,11),(7,12)]

    #0: LF, 1: LM, 2: LH, 3: RF, 4: RM, 5: RH, 
    limb_id = [i for i in range(6) for j in range(2)]
    nodes = [i for i in range(13)]
    
    colors = [[186,30,49], [201,86,79], [213,133,121], #RF, RM, RH
              [15,115, 153], [26,141, 175], [117,190,203] #LF, LM, LH
              ]
    
    edge_colors = [[x / 255.0 for x in colors[i]]  for i in limb_id]
    
    #build graph
    G=nx.Graph()
    G.add_edges_from(edges)
    G.add_nodes_from(nodes)
    
    return G, edge_colors