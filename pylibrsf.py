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

from pylibrsf import PyLibRSF


if __name__ == "__main__":

    #pysteiner = PySteiner()
    clients = [(143, 123), (666, 123), (143, 490), (666, 490)]

    steiner = PyLibRSF("gauss")

    print(steiner.teste([4,5]))


