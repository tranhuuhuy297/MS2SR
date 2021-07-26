import os
import sys
import random
import time

from .state import *
from .neighborhood import *
from .core import *
from .demand import DemandsData
from .constraint import MaxLoad


class LoadOptimizer:
    def __init__(self, sp, capacity, nNodes, nEdges):
        labels = []
        srcs = []
        dest = []
        bws = []
        for i in range(nNodes):
            for j in range(nNodes):
                if i != j:
                    srcs.append(i)
                    dest.append(j)
                    bws.append(0)

        decisionDemands = DemandsData(labels, srcs, dest, bws)

        self.sp = sp
        self.nDemands = decisionDemands.nDemands
        self.capacity = capacity
        self.nNodes = nNodes
        self.nEdges = nEdges
        self.decisionDemands = decisionDemands
        self.edgeDemandState = EdgeDemandStateTree(self.nDemands, self.nEdges, self.capacity)
        self.pathState = PathState(decisionDemands)
        self.flowState = FlowStateRecomputeDAG(nNodes, nEdges, sp, self.pathState, decisionDemands)
        self.flowStateOnCommit = FlowStateRecomputeDAGOnCommit(nNodes, nEdges, sp, self.pathState, decisionDemands,
                                                               self.edgeDemandState)
        self.maxLoad = MaxLoad(nNodes, nEdges, capacity, self.flowState, sp)
        self.bestPaths = SavedState(self.pathState)

        self.pathState.addTrial(self.flowState)
        self.pathState.addTrial(self.flowStateOnCommit)
        self.pathState.addTrial(self.maxLoad)
        self.pathState.addTrial(self.bestPaths)

        self.neighborhoods = [Reset(self.pathState), Remove(self.pathState), Insert(self.nNodes, self.pathState),
                              Replace(self.nNodes, self.pathState)]
        self.kickNeighborhoods = [Reset(self.pathState), Remove(self.pathState)]

    def selectDemand(self):
        edge = self.maxLoad.selectRandomMaxEdge()
        return self.edgeDemandState.selectRandomDemand(edge)

    def visitNeighborhood(self, neighborhood, setter):
        nBestMoves = 0
        bestNeighborhoodLoad = 999999999.0
        improvementFound = False
        neighborhood.setNeighborhood(setter)
        while neighborhood.hasNext():
            neighborhood.next()
            neighborhood.apply()

            if (self.pathState.nChanged > 0 & self.pathState.check()):
                score = self.maxLoad.score()

                if (score == bestNeighborhoodLoad):
                    nBestMoves += 1
                    if (random.randint(0, nBestMoves) == 0):
                        neighborhood.saveBest()
                else:
                    if (score < bestNeighborhoodLoad):
                        nBestMoves = 1
                        improvementFound = True
                        neighborhood.saveBest()
                        bestNeighborhoodLoad = self.maxLoad.score()
                self.pathState.revert()

        return improvementFound

    def kick(self, demand):
        choice = random.randint(0, self.kickNeighborhoods)
        neighborhood = self.kickNeighborhoods[choice]
        if self.visitNeighborhood(neighborhood, demand):
            neighborhood.applyBest()
            self.pathState.update()
            self.pathState.commit()

    def startMoving(self, timeLimit):
        startTime = time.time() * 1000000000
        stopTime = startTime + timeLimit * 1000000
        bestLoad = self.maxLoad.score()
        nIterations = 0
        bestIteration = 0
        while (time.time() * 1000000000 < stopTime):

            nIterations += 1
            if (self.maxLoad.score() > bestLoad) & (nIterations > (bestIteration + 1000)):
                self.bestPaths.restorePath()
                self.pathState.update()
                self.pathState.commit()
                bestIteration = nIterations - 1

            demand = self.selectDemand()

            if (self.maxLoad.score == bestLoad) & (nIterations > (bestIteration + 3)):
                bestIteration = nIterations

                self.kick(demand)

            improvementFound = False
            pNeighborhood = 0

            while (improvementFound == False) & (pNeighborhood < len(self.neighborhoods)):
                neighborhood = self.neighborhoods[pNeighborhood]
                improvementFound = self.visitNeighborhood(neighborhood, demand)

                if improvementFound:

                    neighborhood.applyBest()
                    self.pathState.update()
                    self.pathState.commit()

                    if self.maxLoad.score() < bestLoad:
                        self.bestPaths.savePath()
                        bestLoad = self.maxLoad.score()
                        bestIteration = nIterations

                pNeighborhood += 1

    def solve(self, timeLimit):

        self.startMoving(timeLimit)

        self.bestPaths.restorePath()
        self.pathState.update()
        self.pathState.commit()
        return self.pathState

    def modifierDemands(self, newDemand):
        diffs = []
        for i in range(len(self.decisionDemands.demandTraffics)):
            diffs.append(newDemand.demandTraffics[i] - self.decisionDemands.demandTraffics[i])
            self.decisionDemands.demandTraffics[i] += diffs[i]
        for demand in range(self.nDemands):
            path = self.pathState.path(demand)
            pDetour = self.pathState.size(demand) - 1

            while pDetour > 0:
                pDetour -= 1
                src = path[pDetour]
                dest = path[pDetour + 1]
                self.flowState.modify(src, dest, diffs[demand])
                self.flowStateOnCommit.modify(demand, src, dest, diffs[demand])
        self.flowState.update()
        self.flowStateOnCommit.update()
        self.flowStateOnCommit.commit()
        self.flowState.commit()
        self.maxLoad.initialize()
        self.maxLoad.commit()

    def modifierTrafficMatrix(self, tf):
        labels = []
        srcs = []
        dest = []
        bws = []

        for i in range(len(tf)):
            for j in range(len(tf)):
                if i != j:
                    srcs.append(i)
                    dest.append(j)
                    bws.append(tf[i][j])
        demandData = DemandsData(labels, srcs, dest, bws)
        self.modifierDemands(demandData)

    def extractRoutingPath(self):
        paths = []
        for i in range(self.nNodes):
            A = []
            for j in range(self.nNodes):
                A.append([])
            paths.append(A)

        for i in range(self.pathState.nDemands):
            paths[self.pathState.source(i)][self.pathState.destination(i)] = self.pathState.paths[i].currentPath
        return paths

    def evaluate(self, srPaths, TM):
        mlu = 0
        values = [0] * self.nEdges
        for i in range(self.nNodes):
            for j in range(self.nNodes):
                if i != j:
                    for k in range(len(srPaths[i][j]) - 1):
                        n = srPaths[i][j][k]
                        m = srPaths[i][j][k + 1]
                        paths = self.sp.pathEdges[n][m]
                        nPath = self.sp.nPaths[n][m]
                        increment = TM[i][j] / nPath
                        for path in paths:
                            for edge in path:
                                values[edge] += increment / self.capacity.capacity[edge]
                                if values[edge] > mlu:
                                    mlu = values[edge]
        return mlu
