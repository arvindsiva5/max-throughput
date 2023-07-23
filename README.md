Calculates maximum throughput from one data centre to the target data centres through a network of data centres by modeling the problem as a max-flow network problem solved using the ford-fulkerson method\

The function maxThroughput(connections, maxIn, maxOut, origin, targets) that returns the maximum possible data throughput from the data centre origin to the data centres specified in targets\
Example:\
connections = [(0, 1, 3000), (1, 2, 2000), (1, 3, 1000), (0, 3, 2000), (3, 4, 2000), (3, 2, 1000)]\
maxIn = [5000, 3000, 3000, 3000, 2000]\
maxOut = [5000, 3000, 3000, 2500, 1500]\
origin = 0\
targets = [4, 2]

\# function should return the maximum possible data throughput from the\
\# data centre origin to the data centres specified in targets.\
maxThroughput(connections, maxIn, maxOut, origin, targets)\
\>> 4500
