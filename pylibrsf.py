#!/usr/bin/python3

'''
extern int gst_esmt (
int         nterms,
double *        terms,
double *        length,
int *           nsps,
double *        sps,
int *           nedges,
int *           edges,
int *           soln_status,
gst_param_ptr       params
);
'''

from pygeosteiner import PyGeosteiner


if __name__ == "__main__":

    #pysteiner = PySteiner()
    clients = [(143, 123), (666, 123), (143, 490), (666, 490)]

    steiner = PyGeosteiner()

    steiner.findEsmt(clients)
    print(steiner.florest_[0].terminals)
    print(steiner.florest_[0].steiners)
    print(steiner.florest_[0].edges)
    for idx in range(steiner.getTree(0).getNumEdges()):
        print(steiner.getTree(0).getEdge(idx))

