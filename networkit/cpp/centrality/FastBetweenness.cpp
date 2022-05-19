// no-networkit-format
/*
 * FastBetweenness.cpp
 *
 *  Created on: 29.07.2014
 *      Author: cls, ebergamini
 */

#include <memory>
#include <omp.h>
#include <algorithm>    // std::random_shuffle

#include <networkit/centrality/FastBetweenness.hpp>
#include <networkit/auxiliary/Log.hpp>
#include <networkit/auxiliary/SignalHandling.hpp>
#include <networkit/distance/Dijkstra.hpp>
#include <networkit/distance/BFS.hpp>

namespace NetworKit {

  FastBetweenness::FastBetweenness(const Graph& G, bool normalized, bool computeEdgeCentrality, double sampling_rate) :
    Centrality(G, normalized, computeEdgeCentrality), sampling_rate(sampling_rate) {}

void FastBetweenness::run() {
    Aux::SignalHandler handler;
    const count z = G.upperNodeIdBound();
    scoreData.clear();
    scoreData.resize(z);
    if (computeEdgeCentrality) {
        count z2 = G.upperEdgeIdBound();
        edgeScoreData.clear();
        edgeScoreData.resize(z2);
    }

    std::vector<std::vector<double>> dependencies(omp_get_max_threads(), std::vector<double>(z));
    std::vector<std::unique_ptr<SSSP>> sssps;
    sssps.resize(omp_get_max_threads());
#pragma omp parallel
    {
        omp_index i = omp_get_thread_num();
        if (G.isWeighted())
            sssps[i] = std::unique_ptr<SSSP>(new Dijkstra(G, 0, true, true));
        else
            sssps[i] = std::unique_ptr<SSSP>(new BFS(G, 0, true, true));
    }

    auto computeDependencies = [&](node s) -> void {

        std::vector<double> &dependency = dependencies[omp_get_thread_num()];
        std::fill(dependency.begin(), dependency.end(), 0);

        // run SSSP algorithm and keep track of everything
        auto &sssp = *sssps[omp_get_thread_num()];
        sssp.setSource(s);
        if (!handler.isRunning()) return;
        sssp.run();
        if (!handler.isRunning()) return;
        // compute dependencies for nodes in order of decreasing distance from s
        std::vector<node> stack = sssp.getNodesSortedByDistance();
        while (!stack.empty()) {
            node t = stack.back();
            stack.pop_back();
            for (node p : sssp.getPredecessors(t)) {
                // workaround for integer overflow in large graphs
                bigfloat tmp = sssp.numberOfPaths(p) / sssp.numberOfPaths(t);
                double weight;
                tmp.ToDouble(weight);
                double c= weight * (1 + dependency[t]);
                dependency[p] += c;

                if (computeEdgeCentrality) {
                    const edgeid edgeId = G.edgeId(p, t);
#pragma omp atomic
                    edgeScoreData[edgeId] += c;
                }
            }

            if (t != s)
#pragma omp atomic
                scoreData[t] += dependency[t];
        }
    };
    handler.assureRunning();
    // G.balancedParallelForNodes(computeDependencies);
    /////////////////
    std::vector<int> n_ids(z);
    std::iota(std::begin(n_ids), std::end(n_ids), 0); // Fill with 0, 1, ..., z-1
    std::random_shuffle(n_ids.begin(), n_ids.end());
    int num_samples = z*sampling_rate;

#pragma omp parallel for schedule(guided)
    for (omp_index idx = 0; idx < static_cast<omp_index>(num_samples); ++idx) {
      int v = n_ids[idx];
      if (G.hasNode(v)) {
            computeDependencies(v);
        }
    }    
    /////////////////
    handler.assureRunning();

    if (normalized) {
        // divide by the number of possible pairs
        const double n = static_cast<double>(G.numberOfNodes());
        const double pairs = (n-2.) * (n-1.);
        const double edges =  n    * (n-1.);
        G.parallelForNodes([&](node u){
            scoreData[u] /= sampling_rate * pairs;
        });

        if (computeEdgeCentrality) {
#pragma omp parallel for
            for (omp_index i = 0; i < static_cast<omp_index>(edgeScoreData.size()); ++i) {
  	        edgeScoreData[i] =  edgeScoreData[i] / (edges * sampling_rate);
            }
        }
    }

    hasRun = true;
}

double FastBetweenness::maximum(){
    if (normalized) {
        return 1;
    }

    const double n = static_cast<double>(G.numberOfNodes());
    double score = (n-1)*(n-2);
    if (!G.isDirected())
        score /= 2.;
    return score;
}

} /* namespace NetworKit */
