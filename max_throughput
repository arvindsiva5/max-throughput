import math

class ResidualNetworkEdge:
    """
    A class that represent the edges in the residual network represented by ResidualNetwork class
    """

    def __init__(self, parent: int, child: int, cap: int) -> None:
        """
        Function description: Initializes an edge for the ResidualNetwork. It represents 2 edges
                              (parent, child) with weight of cap and (child, parent) with weight
                              of flow

        :Input:
            parent: An integer representing the parent of the edge, ex and edge (3,4), 3 is the parent
            child: An integer representing the child of the edge, ex and edge (3,4), 4 is the child
            cap: An integer representing the capacity of the edge
        :Output:
            No output
            A ResidualNetworkEdge is initialized with parent and child vertices, capacity and flow

        :Time complexity: O(1)
        :Aux space complexity: O(1)
        """
        self.parent = parent  # vertex parent
        self.child = child  # vertex child
        self.cap = cap  # capacity of edge (parent, child)
        self.flow = 0  # flow along edge (parent, child)


class ResidualNetwork:
    """
    A class that represents a residual network using adjacency lists
    """

    def __init__(self, n: int) -> None:
        """
        Function description: Initialiazes a residual network with n vertices

        :Input:
            n: An integer representing the number of vertices in the residual network
        :Output:
            No output
            A residual network is created

        :Time complexity: O(N), where N is the number of vertices n
        :Aux space complexity: O(N), where N is the number of vertices n
        """
        # initialize empty lists for each vertex
        self.vertices = [[] for _ in range(n)]

    def insert_edge(self, parent: int, child: int, cap: int) -> None:
        """
        Function description: Inserts two edges in the residual network which are (parent, child)
                              with weight cap and (child, parent) with weight flow = 0

        :Input:
            parent: An integer representing the parent of the edge, ex and edge (3,4), 3 is the parent
            child: An integer representing the child of the edge, ex and edge (3,4), 4 is the child
            cap: An integer representing the capacity of the edge

        :Output:
            No output
            An edge (parent, child) with capacity cap and an edge (child, parent) with flow 0 is inserted

        :Time complexity: O(1)
        :Aux space complexity: O(1)
        """
        # create edge (parent, child) with weight == cap (capacity)
        # create edge (child, parenr) with weight == flow == 0
        edge = ResidualNetworkEdge(parent, child, cap)

        # add edge (parent, child)
        self.vertices[parent].append(edge)
        # add edge (child, parent)
        self.vertices[child].append(edge)

    def get_edges(self, v: int) -> list[ResidualNetworkEdge]:
        """
        Function description: Returns a list of outgoing edges (ResidualNetworkEdge) from
                              vertex v

        :Input:
            v: An integer representing the vertex where the edges are required
        :Output:
            A list of ResidualNetworkEdge (The list of outgoing edges from vertex v)

        :Time complexity: O(1)
        :Aux space complexity: O(N), where N is the number of outgoing edges from vertex v
        """
        return self.vertices[v]  # outgoing edges from vertex v

    def __len__(self) -> int:
        """
        Function description: Built in Python len function. Used to return the number of
                              vertices in the residual network

        :Input: No input
        :Output:
            An integer representing the number of vertices in the residual network

        :Time complexity: O(1)
        :Aux space complexity: O(1)
        """
        return len(self.vertices)  # the number of vertices


class Queue:
    """
    Queue class represents a Circular Queue that is implemented using a list
    """

    def __init__(self, n: int) -> None:
        """
        Function description: Initializes a queue of size n that can store a maximum of n
                              elements at any given time

        :Input:
            n: An integer representing the max number of elements to be stored in the queue
        :Output:
            No output
            A circular queue that can store up to n elements at a time is created

        :Time complexity: O(N), where N is the max number of elements to be stored in the queue
        :Aux space complexity: O(N), where N is the max number of elements to be stored in the queue
        """
        self.list = [None for _ in range(n)]  # list to store the elements
        self.front = 0  # the index of the earliest pushed data
        self.rear = 0  # the index of the last pushed data
        self.length = 0  # number of elements in queue

    def push(self, data: int) -> None:
        """
        Function description: Push data (an integer) to the queue

        :Input:
            data: An integer that is to be pushed to the queue
        :Output:
            No output
            The data is pushed to the queue

        :Time complexity: O(1)
        :Aux space complexity: O(1)
        """
        # do not push if queue is full
        if self.is_full():
            raise Exception("Queue is full")

        # push data to queue
        self.list[self.rear] = data
        # increase number of elements in queue
        self.length += 1
        # index for next pushed data
        self.rear = (self.rear + 1) % len(self.list)

    def serve(self) -> int:
        """
        Function description: Serve data (an integer) from the queue. The data that was pushed the\
                              earliest into the queue. (First In First Out)
                              
        :Input: No input
        :Output:
            An integer that is the data to be served. The data is the earliest data added to queue
            and it is served (based on Last In Last Out (LILO))

        :Time complexity: O(1)
        :Aux space complexity: O(1)
        """
        # do not serve if queue is empty
        if self.is_empty():
            raise Exception("Queue is empty")

        # decrease number of elements in queue
        self.length -= 1
        # retrive data to serve
        item = self.list[self.front]
        # set index to the next element that can be served
        self.front = (self.front + 1) % len(self.list)
        return item  # serve data from queue

    def __len__(self) -> int:
        """
        Function description: Built in python len function. Used to return the number of
                              elements stored in the queue

        :Input: No input
        :Output:
            An integer representing the number of elements currently stored in the queue

        :Time complexity: O(1)
        :Aux space complexity: O(1)
        """
        return self.length  # number of elements stored in queue

    def is_empty(self) -> bool:
        """
        Function description: Used to check if the queue is empty (self.list is empty), returns True
                              if the queue is empty else False

        :Input: No input
        :Output:
            A boolean that represents wheter the queue is empty or not
            True is if the queue is empty else False

        :Time complexity: O(1)
        :Aux space complexity: O(1)
        """
        # True if no elements stored else False
        return len(self) == 0

    def is_full(self) -> bool:
        """
        Function description: Used to check if the queue is full (self.list is full), returns True
                              if the queue is full else False

        :Input: No input
        :Output:
            A boolean that represents wheter the queue is full or not
            True is if the queue is full else False

        :Time complexity: O(1)
        :Aux space complexity: O(1)
        """
        # True if number of elements stored == self.list length else False
        return len(self) == len(self.list)


def maxThroughput(
    connections: list[tuple[int]],
    maxIn: list[int],
    maxOut: list[int],
    origin: int,
    targets: list[int],
) -> int:
    """
    Function description: Returns the maximum possible data throughput from the data centre origin
                          to the data centres specified in targets

    Approach description:
    The problem comprises of data centres with connections among then with directions. Moreover, the
    connections have their maximum throughputs and each data centres have their own maximum incoming
    and outgoing data. This problem requires sending data from one data centre (origin) to another 1 or
    more data centres (targets) along the available connections and data centres. We need to find the
    maximum throughput of sending data from origin to the targets. This represents a network and can
    be modeled as a network flow and the maximum flow (the maximum throughput) can be computed using
    the Ford-Fulkerson method. A residual network can be built. There are n data centres.
    For every data centre a vertex is created from 0 to n-1. To represent the maxIn for every data centre i,
    an edge from vertex (n+i) to i is created with weight which is the maxIn value and edge in opposite
    direction with weight 0 representing flow of 0 is inserted. To represent the maxOut for every data centre i,
    an edge from i to vertex (i + n*2) is created with weight which is the maxOut value and edge in opposite
    direction with weight 0 representing flow of 0 is inserted. To model the connections, for every (u, v, c) in
    connections, an edge from vertex (u + n*2) to vertex (v + n) is created with weight c (represent capacity)
    and edges in opposite direction with weight 0 is added to represent flow of 0. This ensures the connections
    take into account the maxIn and maxOut of each data centre ensuring the correct amount of data enters and
    leaves a data centre. Then for source vertex, we create vertex n*3 and add edge from it to the origin vertex
    with weight of maxOut of origin and opposite edge of weight 0 representing flow 0. This ensures only the
    maxOut origin amount of data can leave origin. Then for sink vertex we create vertex n*3+1. For every i in
    targets we add an edge from i to n*3+1 with weight of maxIn of i representing capacity and opposited edge
    of weight 0 representing flow 0. Doing so ensure teh targets only receive maxIn amount of data. With the
    residual network model, Ford-Fulkerson method is performed by finding augmenting paths from source to sink
    vertex and augment them with as much flow as possible till there is no more augmenting path. The total
    flow augmented is the maximum throughput of data from origin to targets.

    D is the number of data centres (vertices)
    C is the number of connections (edges)

    To create the residual network using adjacency list the time complexity is O(D + C). Ford-Fulkerson
    method is performed on the residual network to obtain the maximum throughput from origin to targets.
    The Ford-Fulkerson was performed using BFS which is a Edmond Karp variation. Morevover, the residual
    network is connected because for every data centre there is at least one channel arriving or departing
    it and for every pair of data centre there will be at most one direct communication channel in each 
    direction between them. The BFS will perform in O(D + C) as the residual network use adjacency list and is
    connected. When we have a BFS performing in O(D + C), Ford-Fulkerson method will compute in O(D * C ^ 2).

    The creation of residual network uses space of O(D + C). BFS and getting the path from BFS uses O(D) space.
    Here O(D + C) dominates O(D) so the aux complexity is O(D + C).

    :Input:
        connections: A list of tuples (a, b, t) where
            a is the ID of the data centre from which the communication channel departs,
            b is the ID of the data centre to which the communication channel arrives,
            t is a positive integer representing the maximum throughput of that channel
        maxIn: maxIn is a list of integers in which maxIn[i] specifies the maximum amount of incoming data
               that data centre i can process per second
        maxOut: maxOut is a list of integers in which maxOut[i] specifies the maximum amount of outgoing data
                that data centre i can process per second
        origin: The integer ID origin of the data centre where the data to be backed up is located
        targets: A list of integers such that each integer i in it indicates that backing up data to server
                 with integer ID i is fine
    :Output:
        An integer representing the maximum possible data throughput from the data centre origin to
        the data centres specified in targets

    :Time complexity: O(D * C ^ 2), where D is the number of data centres
                                  , where C is the number of connections
    :Aux space complexity: O(D + C), where D is the number of data centres
                                   , where C is the number of connections
    """

    s = len(maxIn) * 3  # source vertex
    t = s + 1  # sink vertex

    # initialize residual network represeting connections between data centres
    res_net = create_residual_network(connections, maxIn, maxOut, origin, targets)

    # perform Ford-fulkerson method and return max flow (max data throughput)
    return ford_fulkerson(res_net, s, t)


def create_residual_network(
    connections: list[tuple[int]],
    maxIn: list[int],
    maxOut: list[int],
    origin: int,
    targets: list[int],
) -> ResidualNetwork:
    """
    Function description: Create and return a ResidualNetwork that that models the connections
                          between the data centres considering their maxIn and maxOut and the transfer
                          of data from origin to the targets as a maximum network flow problem

    :Input:
        connections: A list of tuples (a, b, t) where
            a is the ID of the data centre from which the communication channel departs,
            b is the ID of the data centre to which the communication channel arrives,
            t is a positive integer representing the maximum throughput of that channel
        maxIn: maxIn is a list of integers in which maxIn[i] specifies the maximum amount of incoming data
               that data centre i can process per second
        maxOut: maxOut is a list of integers in which maxOut[i] specifies the maximum amount of outgoing data
                that data centre i can process per second
        origin: The integer ID origin of the data centre where the data to be backed up is located
        targets: A list of integers such that each integer i in it indicates that backing up data to server
                 with integer ID i is fine
    :Output:
        A residual network that models the connections between the data centres considering their maxIn and maxOut
        and the transfer of data from origin to the targets as a maximum network flow problem

    :Time complexity: O(D + C), where D is the number of data centres
                              , where C is the number of connections
    :Aux space complexity: O(D + C), where D is the number of data centres
                                   , where C is the number of connections
    """

    n = len(maxIn)  # the number of data centres
    # initialise residual network representing the connections of data centres
    r = ResidualNetwork(n * 3 + 2)

    # add ingoing edges with capacity maxIn for each data centre
    for i in range(n):
        v = i + n
        u = i
        c = maxIn[i]
        r.insert_edge(v, u, c)

    # add outgoing edges with capacity maxOut for each data centre
    for i in range(n):
        v = i
        u = i + n * 2
        c = maxOut[i]
        r.insert_edge(v, u, c)

    # add edges that represent connections between data centres with their capacities
    for i in range(len(connections)):
        v = connections[i][0] + n * 2
        u = connections[i][1] + n
        c = connections[i][2]
        r.insert_edge(v, u, c)

    # insert edge from source vertex to origin
    r.insert_edge(n * 3, origin, maxOut[origin])

    # insert edges from targets to sink vertex
    for i in targets:
        r.insert_edge(i, n * 3 + 1, maxIn[i])

    # Residual network representing the problem is returned
    return r


def BFS(r: ResidualNetwork, s: int) -> list[ResidualNetworkEdge]:
    """
    Function description: Performs a breadth first search (BFS) on the ResidualNetwork r by using vertex s
                          s the source vertex. The BFS will compute the predescessor for every vertex in r
                          to traverse from to reach back to the source vertex s in shortest distance. By doing
                          so, a list of the list of ResidualNetworkEdge, pred which represents the edges
                          connecting pred[i] for every i from 0 to len(pred) - 1 to their predecessors which
                          form the shortest path from source vertex s to vertex i

    :Input:
        r: The ResidualNetwork to perform a BFS to find paths from vertex s to other vertices
        s: An integer representing the source vertex for BFS
    :Output:
        A list of ResidualNetworkEdge, pred which represents the edges connecting pred[i] for every i
        from 0 to len(pred) - 1 to their predecessors which form the shortest path from source vertex s
        to vertex i

    :Time complexity: O(V+E), where V is the number of vertices in r
                            , where E is the number of edges in r
    :Aux space complexity: O(V), where V is the number of vertices in r
    """
    n = len(r)  # the number of vertices

    visited = [False for _ in range(n)]  # visited vertices
    pred = [None for _ in range(n)]  # predecessor of each vertex
    visited[s] = True  # source vertex is visited

    queue = Queue(n)  # initialize queue
    queue.push(s)  # push source vertex

    # visits all vertices
    while not queue.is_empty():
        parent = queue.serve()  # serve vertex from queue
        edges = r.get_edges(parent)  # get outgoing edges form parent

        # visit all the outgoing edges of parent vertex
        for e in edges:
            # executed if the edge represents capacity with value > 0
            # and if e.child is not visited
            if parent == e.parent and e.cap > 0 and not visited[e.child]:
                visited[e.child] = True  # e.child vertex is visited
                pred[e.child] = e  # setting edge e as the predescessor edge
                queue.push(e.child)  # push e.child vertex to queue

            # executed if the edge represents flow with value > 0
            # and if e.parent is not visited
            elif parent == e.child and e.flow > 0 and not visited[e.parent]:
                visited[e.parent] = True  # e.parent vertex is visited
                pred[e.parent] = e  # setting edge e as the predescessor edge
                queue.push(e.parent)  # push e.parent vertex to queue

    # return edges that connect the vertices to their predescessors
    return pred


def get_BFS_path(
    pred: list[ResidualNetworkEdge], s: int, t: int
) -> list[tuple[int, ResidualNetworkEdge]]:
    """
    Function description: Computes the path (the list of edges to traverse) to take to traverse
                          from the source vertex s to vertex t

    :Input:
        pred: A list of ResidualNetworkEdge generated by BFS which represents a the edge connecting to its
              predescessor of a vertex i at pred[i]
        s: An integer representing the source vertex
        t: An integer representing the sink vertex (the ending vertx of the path to be computed)
    :Output:
        A list of tuples (pred, e) where e is an edge that has to be traversed to reach vertex t from vertex s
        Since e represent edges (parent, child) and (child, parent), pred will be child or parent to determine which
        edge to select when traversing the path

    :Time complexity: O(V), where V is the length of pred (number of vertices in residual network)
    :Aux space complexity: O(V), where V is the length of pred (number of vertices in residual network)
    """
    # stores the path of edges from s to t
    path = []

    # executed till source vertex is reached
    while t != s:
        # occurs if no path between s and t
        if pred[t] == None:
            # path set to None and break to ensure no path returned
            path = None
            break

        # get edge connecting to predecessor for vertex t
        e = pred[t]

        # executed if the edge represents capacity
        if e.child == t:
            path.append((e.parent, e))  # append the predescessor and edge
            t = pred[t].parent  # set t to predescessor

        # executed if the edge represents flow
        else:
            path.append((e.child, e))  # append the predescessor and edge
            t = pred[t].child  # set t to predescessor

    # executed if there is a path
    if path:
        # path is reversed to get correct order of edged
        path.reverse()

    # return path from s to t
    return path


def get_min_weight(path: list[tuple[int, ResidualNetworkEdge]]) -> int:
    """
    Function description: Gets the minimum weight of the edges (flow or cap [see :Output:]) of the
                          edges in the path

    :Input:
        path: A list of tuples (pred, e) where e is an edge that has to be traversed to reach target vertex from
              source vertex. Since e represent edges (parent, child) and (child, parent), pred will be child or
              parent to determine which edge to select when traversing the path

    :Output:
        An integer representing the minimum weight of the edges in the path
        (Edge weights are represented by flow or cap of ResidualNetworkEdge e, cap if pred == parent else flow)

    :Time complexity: O(N), where N is the number of edges (tuple of (int, ResidualNetworkEdge))
                            in path (length of path)
    :Aux space complexity: O(1)
    """
    # store the minimum weight of all edges along path
    min_weight = math.inf

    # loop every edge in path
    for pred, e in path:
        # executed if the edge represents capacity
        if e.parent == pred:
            # update min_weight if cap is < min_weight
            if e.cap < min_weight:
                min_weight = e.cap
        # executed if the edge represents flow
        else:
            # update min_weight if flow < min_weight
            if e.flow < min_weight:
                min_weight = e.flow

    # return the minimum weight
    return min_weight


def augment_path(path: list[tuple[int, ResidualNetworkEdge]]) -> int:
    """
    Function description: Updates the weight (represented by flow or cap) of the edges along path with the
                          minimum weight value along the path. If the edge represents cap (capacity) in the
                          residual network, the flow is added and cap is subtracted with minimum weight along
                          the path whereas if the edge represents the flow, the cap is added and flow is
                          subtracted with minimum weight along the path

    :Input:
        path: A list of tuples (pred, e), where pred is the predecessor of the edge e that
              represent the edges to traverse to reach from source vertex to target vertex
    :Output:
        An integer representing the amount of flow that has been added to the path in the
        residual network

    :Time complexity: O(N), where N is the number of edges in path (length of path)
    :Aux space complexity: O(1)
    """
    # the minimum weight of edges along path
    flow_to_add = get_min_weight(path)

    # update weight for every edge in path
    for pred, e in path:
        # executed if the edge represents capacity
        if pred == e.parent:
            e.cap -= flow_to_add  # reduce capacity by flow_to_add
            e.flow += flow_to_add  # increase flow by flow_to_add
        # executed if the edge represents flow
        else:
            e.cap += flow_to_add  # increase capacity by flow_to_add
            e.flow -= flow_to_add  # reduce flow by flow_to_add

    # the amount of flow added to the residual network
    return flow_to_add


def ford_fulkerson(r: ResidualNetworkEdge, s: int, t: int) -> int:
    """
    Function description: Performs the Ford-Fulkerson method on the residual network r by augmenting
                          flow along the augmenting paths from source vertex s to sink vertex t until
                          no such augmenting paths exists. Returns an integer representing the maximum
                          possible flow along the residual network from s to t. BFS is used to find tht
                          augmenting paths.

    :Input:
        r: The residual network to perform Ford Fulkerson method to find the maximum flow
        s: An integer representing the source vertex in r
        t: An integer representing the sink vertex in r
    :Output:
        An integer representing the maximum flow possible in residual network r

    :Time complexity: O(D * C ^ 2), where D is the number of data centres
                                , where C is the number of connections
    :Aux space complexity: O(D), where D is the number of data centres
    """
    # perform BFS and get path froms to t
    path = get_BFS_path(BFS(r, s), s, t)
    # maximum flow possible in residual network r
    max_flow = 0

    # executed while augmenting path exists from s to t
    while path != None:
        # path is augmented and the flow augmented is added to max_flow
        max_flow += augment_path(path)
        path = get_BFS_path(BFS(r, s), s, t)  # perform BFS and get path froms to t

    # return the max flow in residual network r
    return max_flow
