/*
 * GraphToolsGTest.cpp
 *
 *  Created on: 22.11.14
 *      Author: Maximilian Vogel
 */

#include <gtest/gtest.h>

#include <networkit/auxiliary/Random.hpp>
#include <networkit/generators/ErdosRenyiGenerator.hpp>
#include <networkit/graph/Graph.hpp>
#include <networkit/graph/GraphTools.hpp>

namespace NetworKit {

class GraphToolsGTest : public testing::TestWithParam<std::pair<bool, bool>> {
protected:
    Graph generateRandomWeights(const Graph &G) const;
    bool weighted() const noexcept;
    bool directed() const noexcept;
};

INSTANTIATE_TEST_CASE_P(InstantiationName, GraphToolsGTest,
                        testing::Values(std::make_pair(false, false),
                                        std::make_pair(true, false),
                                        std::make_pair(false, true),
                                        std::make_pair(true, true)), ); // comma required for variadic macro

Graph GraphToolsGTest::generateRandomWeights(const Graph &G) const {
    Graph Gw(G, true, G.isDirected());
    Gw.forEdges([&](node u, node v) { Gw.setWeight(u, v, Aux::Random::probability()); });
    return Gw;
}

bool GraphToolsGTest::weighted() const noexcept { return GetParam().first; }

bool GraphToolsGTest::directed() const noexcept { return GetParam().second; }

TEST_P(GraphToolsGTest, testGetContinuousOnContinuous) {
    Graph G(10, weighted(), directed());
    auto nodeIds = GraphTools::getContinuousNodeIds(G);
    std::unordered_map<node,node> reference = {{0,0},{1,1},{2,2},{3,3},{4,4},{5,5},{6,6},{7,7},{8,8},{9,9}};
    EXPECT_EQ(reference,nodeIds);
}

TEST_P(GraphToolsGTest, testGetContinuousOnDeletedNodes1) {
    Graph G(10, weighted(), directed());
    G.removeNode(0);
    G.removeNode(1);
    G.removeNode(2);
    G.removeNode(3);
    G.removeNode(4);
    auto nodeIds = GraphTools::getContinuousNodeIds(G);
    std::unordered_map<node,node> reference = {{5,0},{6,1},{7,2},{8,3},{9,4}};
    EXPECT_EQ(reference,nodeIds);
}

TEST_P(GraphToolsGTest, testGetContinuousOnDeletedNodes2) {
    Graph G(10, weighted(), directed());
    G.removeNode(0);
    G.removeNode(2);
    G.removeNode(4);
    G.removeNode(6);
    G.removeNode(8);
    auto nodeIds = GraphTools::getContinuousNodeIds(G);
    std::unordered_map<node,node> reference = {{1,0},{3,1},{5,2},{7,3},{9,4}};
    EXPECT_EQ(reference,nodeIds);
}

TEST_F(GraphToolsGTest, testGetCompactedGraphUndirectedUnweighted1) {
    Graph G(10,false,false);
    G.addEdge(0,1);
    G.addEdge(2,1);
    G.addEdge(0,3);
    G.addEdge(2,4);
    G.addEdge(3,6);
    G.addEdge(4,8);
    G.addEdge(5,9);
    G.addEdge(3,7);
    G.addEdge(5,7);

    auto nodeMap = GraphTools::getContinuousNodeIds(G);
    auto Gcompact = GraphTools::getCompactedGraph(G,nodeMap);

    EXPECT_EQ(G.numberOfNodes(),Gcompact.numberOfNodes());
    EXPECT_EQ(G.numberOfEdges(),Gcompact.numberOfEdges());
    EXPECT_EQ(G.isDirected(),Gcompact.isDirected());
    EXPECT_EQ(G.isWeighted(),Gcompact.isWeighted());
    // TODOish: find a deeper test to check if the structure of the graphs are the same,
    // probably compare results of some algorithms or compare each edge with a reference node id map.
}

TEST_F(GraphToolsGTest, testGetCompactedGraphUndirectedUnweighted2) {
    Graph G(10,false,false);
    G.removeNode(0);
    G.removeNode(2);
    G.removeNode(4);
    G.removeNode(6);
    G.removeNode(8);
    G.addEdge(1,3);
    G.addEdge(5,3);
    G.addEdge(7,5);
    G.addEdge(7,9);
    G.addEdge(1,9);

    auto nodeMap = GraphTools::getContinuousNodeIds(G);
    auto Gcompact = GraphTools::getCompactedGraph(G,nodeMap);

    EXPECT_NE(G.upperNodeIdBound(),Gcompact.upperNodeIdBound());
    EXPECT_EQ(G.numberOfNodes(),Gcompact.numberOfNodes());
    EXPECT_EQ(G.numberOfEdges(),Gcompact.numberOfEdges());
    EXPECT_EQ(G.isDirected(),Gcompact.isDirected());
    EXPECT_EQ(G.isWeighted(),Gcompact.isWeighted());
    // TODOish: find a deeper test to check if the structure of the graphs are the same,
    // probably compare results of some algorithms or compare each edge with a reference node id map.
}

TEST_F(GraphToolsGTest, testGetCompactedGraphUndirectedWeighted1) {
    Graph G(10,true,false);
    G.removeNode(0);
    G.removeNode(2);
    G.removeNode(4);
    G.removeNode(6);
    G.removeNode(8);
    G.addEdge(1,3,0.2);
    G.addEdge(5,3,2132.351);
    G.addEdge(7,5,3.14);
    G.addEdge(7,9,2.7);
    G.addEdge(1,9,0.12345);

    auto nodeMap = GraphTools::getContinuousNodeIds(G);
    auto Gcompact = GraphTools::getCompactedGraph(G,nodeMap);

    EXPECT_EQ(G.totalEdgeWeight(),Gcompact.totalEdgeWeight());
    EXPECT_NE(G.upperNodeIdBound(),Gcompact.upperNodeIdBound());
    EXPECT_EQ(G.numberOfNodes(),Gcompact.numberOfNodes());
    EXPECT_EQ(G.numberOfEdges(),Gcompact.numberOfEdges());
    EXPECT_EQ(G.isDirected(),Gcompact.isDirected());
    EXPECT_EQ(G.isWeighted(),Gcompact.isWeighted());
    // TODOish: find a deeper test to check if the structure of the graphs are the same,
    // probably compare results of some algorithms or compare each edge with a reference node id map.
}

TEST_F(GraphToolsGTest, testGetCompactedGraphDirectedWeighted1) {
    Graph G(10,true,true);
    G.removeNode(0);
    G.removeNode(2);
    G.removeNode(4);
    G.removeNode(6);
    G.removeNode(8);
    G.addEdge(1,3,0.2);
    G.addEdge(5,3,2132.351);
    G.addEdge(7,5,3.14);
    G.addEdge(7,9,2.7);
    G.addEdge(1,9,0.12345);

    auto nodeMap = GraphTools::getContinuousNodeIds(G);
    auto Gcompact = GraphTools::getCompactedGraph(G,nodeMap);

    EXPECT_EQ(G.totalEdgeWeight(),Gcompact.totalEdgeWeight());
    EXPECT_NE(G.upperNodeIdBound(),Gcompact.upperNodeIdBound());
    EXPECT_EQ(G.numberOfNodes(),Gcompact.numberOfNodes());
    EXPECT_EQ(G.numberOfEdges(),Gcompact.numberOfEdges());
    EXPECT_EQ(G.isDirected(),Gcompact.isDirected());
    EXPECT_EQ(G.isWeighted(),Gcompact.isWeighted());
    // TODOish: find a deeper test to check if the structure of the graphs are the same,
    // probably compare results of some algorithms or compare each edge with a reference node id map.
}

TEST_F(GraphToolsGTest, testGetCompactedGraphDirectedUnweighted1) {
    Graph G(10,false,true);
    G.removeNode(0);
    G.removeNode(2);
    G.removeNode(4);
    G.removeNode(6);
    G.removeNode(8);
    G.addEdge(1,3);
    G.addEdge(5,3);
    G.addEdge(7,5);
    G.addEdge(7,9);
    G.addEdge(1,9);
    auto nodeMap = GraphTools::getContinuousNodeIds(G);
    auto Gcompact = GraphTools::getCompactedGraph(G,nodeMap);

    EXPECT_EQ(G.totalEdgeWeight(),Gcompact.totalEdgeWeight());
    EXPECT_NE(G.upperNodeIdBound(),Gcompact.upperNodeIdBound());
    EXPECT_EQ(G.numberOfNodes(),Gcompact.numberOfNodes());
    EXPECT_EQ(G.numberOfEdges(),Gcompact.numberOfEdges());
    EXPECT_EQ(G.isDirected(),Gcompact.isDirected());
    EXPECT_EQ(G.isWeighted(),Gcompact.isWeighted());
    // TODOish: find a deeper test to check if the structure of the graphs are the same,
    // probably compare results of some algorithms or compare each edge with a reference node id map.
}

TEST_P(GraphToolsGTest, testInvertedMapping) {
    Graph G(10, weighted(), directed());
    G.removeNode(0);
    G.removeNode(2);
    G.removeNode(4);
    G.removeNode(6);
    G.removeNode(8);
    G.addEdge(1,3);
    G.addEdge(5,3);
    G.addEdge(7,5);
    G.addEdge(7,9);
    G.addEdge(1,9);
    auto nodeMap = GraphTools::getContinuousNodeIds(G);
    auto invertedNodeMap = GraphTools::invertContinuousNodeIds(nodeMap,G);

    EXPECT_EQ(6,invertedNodeMap.size());

    std::vector<node> reference = {1,3,5,7,9,10};
    EXPECT_EQ(reference,invertedNodeMap);
}

TEST_F(GraphToolsGTest, testRestoreGraph) {
    Graph G(10,false,true);
    G.removeNode(0);
    G.removeNode(2);
    G.removeNode(4);
    G.removeNode(6);
    G.removeNode(8);
    G.addEdge(1,3);
    G.addEdge(5,3);
    G.addEdge(7,5);
    G.addEdge(7,9);
    G.addEdge(1,9);
    auto nodeMap = GraphTools::getContinuousNodeIds(G);
    auto invertedNodeMap = GraphTools::invertContinuousNodeIds(nodeMap,G);
    std::vector<node> reference = {1,3,5,7,9,10};


    EXPECT_EQ(6,invertedNodeMap.size());
    EXPECT_EQ(reference,invertedNodeMap);

    auto Gcompact = GraphTools::getCompactedGraph(G,nodeMap);
    Graph Goriginal = GraphTools::restoreGraph(invertedNodeMap,Gcompact);

    EXPECT_EQ(Goriginal.totalEdgeWeight(),Gcompact.totalEdgeWeight());
    EXPECT_NE(Goriginal.upperNodeIdBound(),Gcompact.upperNodeIdBound());
    EXPECT_EQ(Goriginal.numberOfNodes(),Gcompact.numberOfNodes());
    EXPECT_EQ(Goriginal.numberOfEdges(),Gcompact.numberOfEdges());
    EXPECT_EQ(Goriginal.isDirected(),Gcompact.isDirected());
    EXPECT_EQ(Goriginal.isWeighted(),Gcompact.isWeighted());
}

TEST_P(GraphToolsGTest, testGetRemappedGraph) {
    const auto n = 4;
    Graph G(n, weighted(), directed());
    for (auto i : {0, 1, 2})
        G.addEdge(i, i + 1, i);

    if (directed())
        G.addEdge(1, 1, 12);

    std::vector<node> perm(n);
    for (int i = 0; i < n; ++i) perm[i] = i;

    std::mt19937_64 gen;
    for (int iter = 0; iter < 10; iter++) {
        std::shuffle(perm.begin(), perm.end(), gen);
        auto G1 = GraphTools::getRemappedGraph(G, n, [&](node i) { return perm[i]; });
        ASSERT_EQ(G1.numberOfNodes(), n);
        ASSERT_EQ(G1.numberOfEdges(), G.numberOfEdges());
        ASSERT_EQ(G1.numberOfSelfLoops(), G.numberOfSelfLoops());

        for (int i = 0; i < n; ++i) {
            for (int j = 0; i < n; ++i) {
                ASSERT_EQ(G.hasEdge(i, j), G1.hasEdge(perm[i], perm[j]));
                ASSERT_EQ(G.weight(i, j), G1.weight(perm[i], perm[j]));
            }
        }
    }
}

TEST_P(GraphToolsGTest, testGetRemappedGraphWithDelete) {
    const auto n = 4;
    Graph G(n, weighted(), directed());
    for (auto i : {0, 1, 2})
        G.addEdge(i, i + 1, i);

    if (directed())
        G.addEdge(1, 1, 12);

    std::vector<node> perm(n);
    for (int i = 0; i < n; ++i) perm[i] = i;

    std::mt19937_64 gen;
    std::uniform_int_distribution<node> distr(0, n-1);
    for (int iter = 0; iter < 10; iter++) {
        std::shuffle(perm.begin(), perm.end(), gen);

        const auto del = distr(gen);

        auto G1 = GraphTools::getRemappedGraph(G, n,
            [&](node i) { return perm[i]; },
            [&](node i) { return i == del; }
        );

        auto expected_num_edges = G.numberOfEdges();
        expected_num_edges -= G.degree(del);
        if (directed())
            expected_num_edges -= G.degreeIn(del);
        //do double count self-loops
        expected_num_edges += G.hasEdge(del, del);

        ASSERT_EQ(G1.numberOfNodes(), n);
        ASSERT_EQ(G1.numberOfEdges(), expected_num_edges) << " del=" << del;
        ASSERT_EQ(G1.numberOfSelfLoops(), G.numberOfSelfLoops() - G.hasEdge(del, del)) << " del=" << del;

        for (int i = 0; i < n; ++i) {
            for (int j = 0; i < n; ++i) {
                if (i == static_cast<int>(del) || j == static_cast<int>(del)) {
                    ASSERT_FALSE(G1.hasEdge(perm[i], perm[j])) << "i=" << i << " j=" << j << " del=" << del;
                } else {
                    ASSERT_EQ(G.hasEdge(i, j), G1.hasEdge(perm[i], perm[j]));
                    ASSERT_EQ(G.weight(i, j), G1.weight(perm[i], perm[j]));
                }
            }
        }
    }
}

TEST_P(GraphToolsGTest, testCopyNodes) {
    constexpr count n = 200;
    constexpr double p = 0.01;
    constexpr count nodesToDelete = 50;

    auto checkNodes = [&](const Graph &G, const Graph &GCopy) {
        EXPECT_EQ(G.isDirected(), GCopy.isDirected());
        EXPECT_EQ(G.isWeighted(), GCopy.isWeighted());
        EXPECT_EQ(G.numberOfNodes(), GCopy.numberOfNodes());
        EXPECT_EQ(GCopy.numberOfEdges(), 0);
        for (node u = 0; u < G.upperNodeIdBound(); ++u) {
            EXPECT_EQ(G.hasNode(u), GCopy.hasNode(u));
        }
    };

    for (int seed : {1, 2, 3}) {
        Aux::Random::setSeed(seed, false);
        auto G = ErdosRenyiGenerator(n, p, directed()).generate();

        auto GCopy = GraphTools::copyNodes(G);
        checkNodes(G, GCopy);
        for (count i = 0; i < nodesToDelete; ++i) {
            G.removeNode(G.randomNode());
            GCopy = GraphTools::copyNodes(G);
            checkNodes(G, GCopy);
        }
    }
}

TEST_P(GraphToolsGTest, testSubgraphFromNodesUndirected) {
    auto G = Graph(4, weighted(), false);

    /**
     *      1
     *   /  |  \
     * 0    |    3
     *   \  |  /
     *      2
     */

    G.addEdge(0, 1, 1.0);
    G.addEdge(0, 2, 2.0);
    G.addEdge(3, 1, 4.0);
    G.addEdge(3, 2, 5.0);
    G.addEdge(1, 2, 3.0);

    {
        std::unordered_set<node> nodes = {0};
        auto res = GraphTools::subgraphFromNodes(G, nodes);
        EXPECT_EQ(weighted(), res.isWeighted());
        EXPECT_FALSE(res.isDirected());
        EXPECT_EQ(res.numberOfNodes(), 1);
        EXPECT_EQ(res.numberOfEdges(), 0);
    }

    {
        std::unordered_set<node> nodes = {0};
        auto res = GraphTools::subgraphFromNodes(G, nodes, true);

        EXPECT_EQ(res.numberOfNodes(), 3);
        EXPECT_EQ(res.numberOfEdges(), 2); // 0-1, 0-2, NOT 1-2

        EXPECT_DOUBLE_EQ(G.weight(0, 1), weighted() ? 1.0 : defaultEdgeWeight);
        EXPECT_DOUBLE_EQ(G.weight(0, 2), weighted() ? 2.0 : defaultEdgeWeight);
    }

    {
        std::unordered_set<node> nodes = {0, 1};
        auto res = GraphTools::subgraphFromNodes(G, nodes);
        EXPECT_EQ(res.numberOfNodes(), 2);
        EXPECT_EQ(res.numberOfEdges(), 1); // 0 - 1
    }

    {
        std::unordered_set<node> nodes = {0, 1};
        auto res = GraphTools::subgraphFromNodes(G, nodes, true);
        EXPECT_EQ(res.numberOfNodes(), 4);
        EXPECT_EQ(res.numberOfEdges(), 4); // 0-1, 0-2, 1-2, 1-3
    }
}

TEST_P(GraphToolsGTest, testSubgraphFromNodesDirected) {
    auto G = Graph(4, weighted(), true);

    /**
     *      1
     *   /  |  \
     * 0    |    3
     *   \  |  /
     *      2
     */

    G.addEdge(0, 1, 1.0);
    G.addEdge(0, 2, 2.0);
    G.addEdge(3, 1, 4.0);
    G.addEdge(3, 2, 5.0);
    G.addEdge(1, 2, 3.0);

    {
        std::unordered_set<node> nodes = {0};
        auto res = GraphTools::subgraphFromNodes(G, nodes);

        EXPECT_EQ(weighted(), res.isWeighted());
        EXPECT_TRUE(res.isDirected());

        EXPECT_EQ(res.numberOfNodes(), 1);
        EXPECT_EQ(res.numberOfEdges(), 0);
    }

    {
        std::unordered_set<node> nodes = {0};
        auto res = GraphTools::subgraphFromNodes(G, nodes, true);
        EXPECT_EQ(res.numberOfNodes(), 3);
        EXPECT_EQ(res.numberOfEdges(), 2); // 0->1, 0->2, NOT 1->2
    }

    {
        std::unordered_set<node> nodes = {0, 1};
        auto res = GraphTools::subgraphFromNodes(G, nodes);
        EXPECT_EQ(res.numberOfNodes(), 2);
        EXPECT_EQ(res.numberOfEdges(), 1); // 0 -> 1
    }

    {
        std::unordered_set<node> nodes = {0, 1};
        auto res = GraphTools::subgraphFromNodes(G, nodes, true);
        EXPECT_EQ(res.numberOfNodes(), 3);
        EXPECT_EQ(res.numberOfEdges(), 3); // 0->1, 0->2, 1->2
    }

    {
        std::unordered_set<node> nodes = {0, 1};
        auto res = GraphTools::subgraphFromNodes(G, nodes, true, true);
        EXPECT_EQ(res.numberOfNodes(), 4);
        EXPECT_EQ(res.numberOfEdges(), 4); // 0->1, 0->2, 1->2, 3->1
    }

}

TEST_P(GraphToolsGTest, testTranspose) {
    auto G = Graph(4, weighted(), true);

    /**
     *      1
     *   /  |  \
     * 0    |    3
     *   \  |  /
     *      2
     */

    G.addNode(); // node 4
    G.addNode(); // node 5
    G.addNode(); // node 6
    G.removeNode(5);

    G.addEdge(0, 0, 3.14);
    G.addEdge(0, 4, 3.14);
    G.removeEdge(0, 4);
    G.addEdge(0, 6, 3.14);

    // expect throw error when G is undirected
    if (!G.isDirected()) {
        EXPECT_ANY_THROW(GraphTools::transpose(G));
    } else {
        Graph Gtrans = GraphTools::transpose(G);
        // check summation statistics
        EXPECT_EQ(G.numberOfNodes(), Gtrans.numberOfNodes());
        EXPECT_EQ(G.upperNodeIdBound(), Gtrans.upperNodeIdBound());
        EXPECT_EQ(G.numberOfEdges(), Gtrans.numberOfEdges());
        EXPECT_EQ(G.upperEdgeIdBound(), Gtrans.upperEdgeIdBound());
        EXPECT_EQ(G.totalEdgeWeight(), Gtrans.totalEdgeWeight());
        EXPECT_EQ(G.numberOfSelfLoops(), Gtrans.numberOfSelfLoops());

        // test for regular edges
        EXPECT_TRUE(G.hasEdge(0, 6));
        EXPECT_FALSE(G.hasEdge(6, 0));
        EXPECT_TRUE(Gtrans.hasEdge(6, 0));
        EXPECT_FALSE(Gtrans.hasEdge(0, 6));
        // .. and for selfloops
        EXPECT_TRUE(G.hasEdge(0, 0));
        EXPECT_TRUE(Gtrans.hasEdge(0, 0));

        // check for edge weights
        EXPECT_EQ(G.weight(0, 6), weighted() ? 3.14 : defaultEdgeWeight);
        EXPECT_EQ(Gtrans.weight(6, 0), weighted() ? 3.14 : defaultEdgeWeight);
        EXPECT_EQ(G.weight(0, 0), Gtrans.weight(0, 0));
    }
}

TEST_P(GraphToolsGTest, testToUndirected) {
    constexpr count n = 200;
    constexpr double p = 0.2;

    auto testGraphs = [&](const Graph &G, const Graph &G1) {
        EXPECT_EQ(G.numberOfNodes(), G1.numberOfNodes());
        EXPECT_EQ(G.upperNodeIdBound(), G1.upperNodeIdBound());
        EXPECT_EQ(G.numberOfEdges(), G1.numberOfEdges());
        EXPECT_EQ(G.upperEdgeIdBound(), G1.upperEdgeIdBound());
        EXPECT_EQ(G.isWeighted(), G1.isWeighted());
        EXPECT_NE(G.isDirected(), G1.isDirected());
        EXPECT_EQ(G.hasEdgeIds(), G1.hasEdgeIds());

        G.forEdges([&](node u, node v, edgeweight w) {
            EXPECT_TRUE(G1.hasEdge(u, v));
            EXPECT_DOUBLE_EQ(G1.weight(u, v), w);
        });
    };

    for (int seed : {1, 2, 3}) {
        Aux::Random::setSeed(seed, false);
        auto G = ErdosRenyiGenerator(n, p, true).generate();
        if (weighted()) {
            G = generateRandomWeights(G);
        }
        auto G1 = GraphTools::toUndirected(G);
        testGraphs(G, G1);
    }
}

TEST_P(GraphToolsGTest, testToUnWeighted) {
    constexpr count n = 200;
    constexpr double p = 0.2;

    auto testGraphs = [&](const Graph &G, const Graph &G1) {
        EXPECT_EQ(G.numberOfNodes(), G1.numberOfNodes());
        EXPECT_EQ(G.upperNodeIdBound(), G1.upperNodeIdBound());
        EXPECT_EQ(G.numberOfEdges(), G1.numberOfEdges());
        EXPECT_NE(G.isWeighted(), G1.isWeighted());
        EXPECT_EQ(G.isDirected(), G1.isDirected());
        EXPECT_EQ(G.hasEdgeIds(), G1.hasEdgeIds());

        G.forEdges([&](node u, node v) {
            EXPECT_TRUE(G1.hasEdge(u, v));
            if (G1.isWeighted()) {
                EXPECT_EQ(G1.weight(u, v), defaultEdgeWeight);
            }
        });
    };

    for (int seed : {1, 2, 3}) {
        Aux::Random::setSeed(seed, false);
        auto G = ErdosRenyiGenerator(n, p, directed()).generate();
        auto G1 = GraphTools::toWeighted(G);
        testGraphs(G, G1);

        G = generateRandomWeights(G);
        G1 = GraphTools::toUnweighted(G);
        testGraphs(G, G1);
    }
}

} // namespace NetworKit
