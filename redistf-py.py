import redis
import numpy as np

class RedisTF(object):
    def __init__(self, conn):
        self.__conn = conn

    # Loads a graph
    def SetGraph(self, graph, path):
        with open(path, 'rb') as f:
            payload = f.read()
        return self.__conn.execute_command('DL.GRAPH', graph, payload)

    # Sets a tensor
    def SetTensor(self, tensor, dtype, shape, values):
        args = [tensor, dtype]
        args += [str(x) for x in shape]
        args.append('VALUES')
        args += [str(x) for x in values]
        return self.__conn.execute_command('DL.TENSOR', *args)

    # Runs a graph with a list of input tensor-name tuples, storing the result
    def Run(self, graph, inputs, output):
        args = [graph, len(inputs)]
        for i in inputs:
            args.append(i[0])
            args.append(i[1])
        args += [str(x) for x in output]
        return self.__conn.execute_command('DL.RUN', *args)

    # Gets the value of a tensor
    def Values(self, tensor):
        return self.__conn.execute_command('DL.VALUES', tensor)

if __name__ == '__main__':
    conn = redis.StrictRedis()
    rtf = RedisTF(conn)
    cap = 50
    print ('Setting the graph: {}'.format(rtf.SetGraph('graph', 'graph2.pb')))
    for i in range(100):
        print ('----------------------')
        x = np.random.randint(cap)
        print ('Setting tensor t1: {}'.format(rtf.SetTensor('t1', 'FLOAT', [2, 1, 1], [x])))
        print ('Setting tensor t2: {}'.format(rtf.SetTensor('t2', 'FLOAT', [2, 1, 1], [np.sqrt(x)])))
        print ('Running the thing: {}'.format(rtf.Run('graph', [('t1', 'X'), ('t2', 'Y')], ('t3', 'vy'))))
        print ('Resulting values: sqrt of {} -> {}'.format(x, rtf.Values('t3')))
