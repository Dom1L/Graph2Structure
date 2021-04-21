import numpy as np
import networkx as nx
from numpy import sqrt, zeros, sum
from numpy.linalg import svd, norm, lstsq
from scipy.optimize import minimize
from numba import jit

from g2s.constants import vdw_radii


class edge_t:
    def __init__(self, i, j, l, u):
        self.i = i
        self.j = j
        self.l = l
        self.u = u


class get_graph:
    def __init__(self, dmat, nuclear_charges):
        self.edgeEqu = []  # dij == lij == uij (exact distance information)
        self.edgeBnd = []  # lij <= dij <= uij (bound distance information)

        n, m = np.triu_indices(dmat.shape[1], k=1)
        for i, j in zip(n, m):
            u = dmat[i, j]
            l = dmat[j, i]
            if dmat[i, j] == 0.0:
                u = 20.
                l = 1. if nuclear_charges is None else vdw_radii[nuclear_charges[i]] + vdw_radii[
                    nuclear_charges[j]]
            edge = edge_t(i, j, l, u)
            if l == u:
                self.edgeEqu.append(edge)
            else:
                self.edgeBnd.append(edge)
        self.G = self.init_graph()
        self.numNodes = len(self.G)
        self.numEdgeEqu = len(self.edgeEqu)
        self.numEdgeBnd = len(self.edgeBnd)

    def init_graph(self):
        G = nx.Graph()
        for edge in self.edgeEqu + self.edgeBnd:
            G.add_edge(edge.i, edge.j, l=edge.l, u=edge.u)
        return G


def calcNodeMaxRelRes(G, lstB, setB, x, i):
    maxRelRes = 0.0
    if i in lstB:
        xi = x[i]
        for j in G.neighbors(i):
            if not setB[j]:  # only nodes whose coords are fixed
                continue
            xj = x[j]
            dij = norm(xi - xj)
            gij = G[i][j]
            lij, uij = gij['l'], gij['u']
            mij = (lij + uij)
            relResL, relResU = (lij - dij) / mij, (dij - uij) / mij
            # ijMaxRelErr = max(relErrL, relErrU)
            relRes = relResU if relResU > relResL else relResL
            if relRes > 0 and maxRelRes < relRes:  # maxRelErr can be negative
                maxRelRes = relRes
    else:
        for j in G.neighbors(i):
            if not setB[j]:
                continue
            relRes = calcNodeMaxRelRes(G, lstB, setB, x, j)
            if relRes > maxRelRes:
                maxRelRes = relRes
    return maxRelRes


def calcBaseMaxRelRes(G, lstB, setB, x):
    n = len(lstB)
    maxRelRes = 0.0
    for k in range(n):
        i = lstB[k]
        xi = x[i]
        for j in G.neighbors(i):
            if j > i or not setB[j]:
                continue
            xj = x[j]
            dij = norm(xi - xj)
            gij = G[i][j]
            lij, uij = gij['l'], gij['u']
            mij = (lij + uij)
            relResL, relResU = (lij - dij) / mij, (dij - uij) / mij
            # ijMaxRelErr = max(relErrL, relErrU)
            relRes = relResU if relResU > relResL else relResL
            if relRes > 0 and maxRelRes < relRes:  # maxRelErr can be negative
                maxRelRes = relRes
    return maxRelRes


@jit(nopython=True, fastmath=True, cache=True)
def smoothMaxZero(z, tau, lam):
    lam_z = lam * z
    fz = lam_z + sqrt(lam_z ** 2 + tau)
    gz = lam * fz / (fz - lam_z)
    return fz, gz


@jit(nopython=True, fastmath=True, cache=True)
def smoothDistance(yi, yj, tau):
    fy = sqrt(sum(yi ** 2) + sum(yj ** 2) - 2 * (yi @ yj) + tau)
    gz = (yi - yj) / fy
    return fy, gz


@jit(nopython=True, fastmath=True, cache=True)
def funNLP(y, tau, lam, L, U):
    f = 0.0
    g = np.zeros(len(y))
    for l, u in zip(L, U):
        # for i in L:
        i, j, l_val = l
        u_val = u[2]

        yi = y[(3 * i):(3 * (i + 1))]
        # for j in L[i]:
        yj = y[(3 * j):(3 * (j + 1))]
        fdij, gdij = smoothDistance(yi, yj, tau)
        zlij = l_val - fdij
        zuij = fdij - u_val
        fmlij, gmlij = smoothMaxZero(zlij, tau, lam)
        fmuij, gmuij = smoothMaxZero(zuij, tau, lam)
        aij = gmuij - gmlij
        f += fmlij + fmuij
        g[(3 * i):(3 * (i + 1))] += aij * gdij
        g[(3 * j):(3 * (j + 1))] -= aij * gdij
    return (f, g)


def checkDiff(f, y, args, rad=1, ntests=50, tol=1e-3):
    # check derivatives of f around y using numerical approximation as probe.
    ynrm = norm(y)
    dy = ynrm / 1000
    maxRelErr = 0
    for _ in range(ntests):
        x = y + np.random.uniform(-1, 1, size=y.shape) * ynrm
        _, g = f(x, args)
        gnum = np.zeros(y.shape, dtype=float)
        xfw = x.copy()
        xbk = x.copy()
        for i in range(len(y)):
            xfw[i] = x[i] + dy
            xbk[i] = x[i] - dy
            ffw, _ = f(xfw, args)
            fbk, _ = f(xbk, args)
            gnum[i] = (ffw - fbk) / (2 * dy)
            # reset xfw and xbw
            xfw[i] = x[i]
            xbk[i] = x[i]
        relErr = norm(g - gnum) / norm(g)
        if relErr > maxRelErr:
            maxRelErr = relErr
        if relErr > tol:
            raise Exception('Numerical diff and g are different.')


def saveResult(nmr, summary, ans):
    fn_ans = nmr.file.replace('.nmr', '.ans')
    print('Writing ' + fn_ans)
    with open(fn_ans, 'w') as fid:
        fid.write(summary + '\n')
        fid.write('\nCoords [ x y z ] =====\n')
        x = ans['x']
        for i in range(nmr.numNodes):
            fid.write('% 12.8g % 12.8g % 12.8g\n' %
                      (x[i, 0], x[i, 1], x[i, 2]))


def lsbuild(distance_matrix, nuclear_charges, lstB=None, setB=None, t=0.5):
    nmr = get_graph(distance_matrix, nuclear_charges)
    tolMaxRelRes = 1e-4  # tolerance of distance constraints
    G = nmr.G
    if lstB is None and setB is None:
        lstB, setB = initB(G)
    x = initX(G, lstB, t=t)
    maxRelRes = calcBaseMaxRelRes(G, lstB, setB, x)
    print('Base (maxRelRes: %.3e)' % maxRelRes)
    if maxRelRes > tolMaxRelRes:  # refining solution?
        solveNLP(G, setB, lstB, x, False, None)
        maxRelRes = calcBaseMaxRelRes(G, lstB, setB, x)
        print('   NLP   (maxRelRes: %.3e)' % maxRelRes)
    # rank of unsolved (not fixed) atoms
    nodeRnk, nodeIdx = initNodeRnk(G, lstB, setB, x)
    for k in range(len(lstB), nmr.numNodes):
        i = nodeRnk[k]['node']
        if nodeRnk[k]['deg'] < 4:
            # There is not enough info to set i-th node
            print('Warning: Not all nodes were solved.')
            break
        # add i to the solved set of vertices
        setB[i] = True
        lstB.append(i)
        nodeRnk[k]['deg'] = np.inf
        # calc coords
        solveLSTSQ(G, setB, x, i, t=t)
        maxRelRes = calcNodeMaxRelRes(G, lstB, setB, x, i)
        if maxRelRes > tolMaxRelRes:
            solveNLP(G, setB, lstB, x, False, None)
            maxRelRes = calcBaseMaxRelRes(G, lstB, setB, x)
        # TODO This step is O(n * max(deg)) and it could be approximately done
        updtNodeRnk(G, lstB, setB, x, i, nodeRnk, nodeIdx)
    notSolvedNodes = [i for i in range(nmr.numNodes) if not setB[i]]
    for k in notSolvedNodes:
        i = nodeRnk[k]['node']
        if nodeRnk[k]['deg'] < 4:
            # There is not enough info to set i-th node
            print('Warning: Not all nodes were solved.')
            break
        # add i to the solved set of vertices
        setB[i] = True
        lstB.append(i)
        nodeRnk[k]['deg'] = np.inf
        # calc coords
        solveLSTSQ(G, setB, x, i, t=t)
        maxRelRes = calcNodeMaxRelRes(G, lstB, setB, x, i)
        if maxRelRes > tolMaxRelRes:
            solveNLP(G, setB, lstB, x, False, None)
            maxRelRes = calcBaseMaxRelRes(G, lstB, setB, x)
        # TODO This step is O(n * max(deg)) and it could be approximately done
        updtNodeRnk(G, lstB, setB, x, i, nodeRnk, nodeIdx)
    ans = {
        'x': x,
        'maxRelRes': maxRelRes,
        'numSolvedNodes': len(lstB),
        'notSolvedNodes': notSolvedNodes,
    }
    return ans


def updtNodeRnk(G, lstB, setB, x, i, nodeRnk, nodeIdx):
    # update maxRelRes of nodes on B
    for j in lstB:
        jidx = nodeIdx[j]
        nodeRnk[jidx]['maxRelRes'] = calcNodeMaxRelRes(G, lstB, setB, x, j)

    # update naxRelRes of nodes out of B
    V = np.zeros(len(nodeRnk), dtype=bool)  # set of updated vertices
    V[lstB] = True
    for k in lstB:
        for j in G.neighbors(k):
            if V[j]:
                continue
            V[j] = True
            jidx = nodeIdx[j]
            nodeRnk[jidx]['maxRelRes'] = calcNodeMaxRelRes(G, lstB, setB, x, j)

    # update degree of nodes out of B
    for j in G.neighbors(i):
        if j in setB:
            continue
        jidx = nodeIdx[j]
        nodeRnk[jidx]['deg'] += 1

    # sort nodeRnk
    sortNodeRnk(nodeRnk)
    for k, u in enumerate(nodeRnk):
        nodeIdx[u['node']] = k


def initX(G, lstB, t=0.5):
    # x[lstB[0]] is set to [0,0,0] (anchor)
    x = zeros((len(G.nodes), 3), dtype=float)
    M = zeros((3, 3), dtype=float)
    # TODO: It could be better to select a node with the best data (tight boundaries)
    # create the approximated squadred distance matrix
    D = zeros((4, 4), dtype=float)
    n = len(lstB)
    for iIdx in range(n):
        i = lstB[iIdx]
        for jIdx in range(iIdx + 1, n):
            j = lstB[jIdx]
            lij, uij = G[i][j]['l'], G[i][j]['u']
            dij = ((1 - t) * lij + t * uij) ** 2
            D[iIdx, jIdx], D[jIdx, iIdx] = dij, dij
    # set M
    for i in range(3):
        for j in range(i, 3):
            mij = (D[0, i + 1] + D[0, j + 1] - D[i + 1, j + 1]) / 2.0
            M[i, j], M[j, i] = mij, mij
    U, S, _ = svd(M)
    # scaling cols by the largest singular values
    for i in range(3):
        U[:, i] *= sqrt(S[i])
    # set values
    for i in range(3):
        x[lstB[i + 1]] = U[i]
    return x


def sortNodeRnk(nodeRnk):
    # TODO Some specialized algorithm could be used here
    nodeRnk.sort(key=lambda u: (-u['deg'], u['maxRelRes']))


def initNodeRnk(G, lstB, setB, x):
    # TODO Consider to store maxRelRes of each node on setB to reduce calcs
    nodeRnk = []
    for i in G:  # this loop is O(nnodes * max(deg(G)))
        # ensure that fixed nodes (on B) will be the first ones after sorting
        iNeighsB = [j for j in G.neighbors(i) if setB[j]]
        deg = np.inf if setB[i] else len(iNeighsB)
        maxRelRes = calcNodeMaxRelRes(G, lstB, setB, x, i)
        nodeRnk.append({'node': i, 'deg': deg, 'maxRelRes': maxRelRes, })
    sortNodeRnk(nodeRnk)
    nodeIdx = np.zeros(len(nodeRnk), dtype=int)
    for k in range(len(nodeRnk)):
        nodeIdx[nodeRnk[k]['node']] = k
    return nodeRnk, nodeIdx


def initB(G):
    # TODO We may consider all cliques and get the "best" of them (resp. to some stability criterion) or just a good enough clique for the sake of time efficiency.
    # Reference:
    # Östergård, Patric RJ. "A fast algorithm for the maximum clique problem." Discrete Applied Mathematics 120.1-3 (2002): 197-207.

    # look for a base that allows to fix the maximum number of nodes
    numBases = 0  # number of tested bases
    numNodes = len(G.nodes)
    maxB = []
    maxLenB = 0  # largest base len
    for v1 in G.nodes:
        N1 = set(G.neighbors(v1))
        if len(N1) < 3:
            continue
        for v2 in N1:
            if v2 < v1:  # removing symmetry
                continue
            N2 = set(G.neighbors(v2)) & N1
            if len(N2) < 2:
                continue
            for v3 in N2:
                if v3 < v1 or v3 < v2:  # removing symmetry
                    continue
                N3 = set(G.neighbors(v3)) & N2
                if len(N1) < 1:
                    continue
                for v4 in N3:
                    if v4 < v1 or v4 < v2 or v4 < v3:  # removing symmetry
                        continue
                    numBases += 1
                    # numFixNeighs[i]: num of neighs of vertice i already fixed
                    numFixNeighs = zeros(numNodes, dtype=int)
                    # the base is fixed by construction
                    numFixNeighs[v1], numFixNeighs[v2], numFixNeighs[v3], numFixNeighs[v4] = 4, 4, 4, 4
                    F = {v1, v2, v3, v4}  # nodes to be fixed
                    B = set()  # set of fixed nodes
                    lenB = len(B)
                    while len(F) > 0:
                        v = F.pop()
                        B.add(v)
                        lenB += 1
                        for u in G.neighbors(v):
                            # ensure that u will be added once
                            # u has one more fixed neigh (which is v)
                            numFixNeighs[u] += 1
                            if numFixNeighs[u] == 4:
                                F.add(u)
                    if lenB > maxLenB:
                        maxLenB = lenB
                        maxB = B.copy()
                        lstB = [v1, v2, v3, v4]
                        print('Updating Bmax (Blen = %d)' % lenB)
    print('Optimal base %s found after %d tests.' % (str(lstB), numBases))
    if len(maxB) < numNodes:
        F = set(G.nodes) - maxB
        print('Warning:')
        print('   Only %d/%d nodes can be solved (fixed).' %
                     (maxLenB, numNodes))
        print(
            '   Nodes %s will not fixed (not enough constraints).' % (str([i + 1 for i in F])))
    setB = np.zeros(numNodes, dtype=bool)
    setB[lstB] = True
    return lstB, setB


def solveLSTSQ(G, setB, x, i, t=0.5):
    iNeighB = [j for j in G.neighbors(i) if setB[j]]
    n = len(iNeighB)
    A = np.zeros((n - 1, 3), dtype=float)
    b = np.zeros(n - 1, dtype=float)
    for k, j in enumerate(iNeighB):
        xj = x[j]
        gij = G[i][j]
        lij, uij = gij['l'], gij['u']
        dij = (1.0 - t) * lij + t * uij
        aij = dij ** 2 - norm(xj) ** 2
        if k == 0:
            x0, ai0 = x[j], aij
        else:
            A[k - 1, :] = x0 - xj
            b[k - 1] = (aij - ai0) / 2.0
    x[i] = lstsq(A, b, rcond=None)[0]


def solveNLP(G, setB, lstB, x, check, tm, tau=1e-5, lam=0.5):
    # TODO A C/C++ call would have a dramatically impact here (see https://realpython.com/python-bindings-overview/).
    # marshalling data
    L, U = [], []
    locIdxB = {lstB[k]: k for k in range(len(lstB))}
    for i in lstB:
        iloc = locIdxB[i]
        for j in G.neighbors(i):
            if j < i and setB[j]:
                jloc = locIdxB[j]
                gij = G[i][j]
                L.append([iloc, jloc, gij['l']])
                U.append([iloc, jloc, gij['u']])
    L, U = np.array(L), np.array(U)
    # convert list of 3D points to a single array
    y = x[lstB].reshape(3 * len(lstB))
    if check:
        checkDiff(funNLP, y, args=(tau, lam, L, U))
    ans = minimize(funNLP, y, args=(tau, lam, L, U), method='BFGS', jac=True)
    # retrieve solution
    for k, i in enumerate(lstB):
        x[i] = ans.x[(3 * k):(3 * (k + 1))]
