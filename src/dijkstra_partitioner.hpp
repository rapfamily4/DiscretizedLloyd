#ifndef DIJKSTRA_PARTITIONER_H
#define DIJKSTRA_PARTITIONER_H

#include <cmath>
#include <cassert>
#include <cstdlib>
#include <iostream>
#include <algorithm>
#include <vector>
#include <utility>
#include "stopwatch.hpp"
#include "performance_stats.hpp"
#include "consts.hpp"

class DijkstraPartitioner {
private:
	enum NodeStatus {
		Unvisited,
		Visited,
		Processed,
	};

	struct Arch {
		int target;
		float cost;

		Arch(int target, float cost) :
			target{ target },
			cost{ cost } {}
	};

	struct Node {
		int regionId;
		int parentId;
		int frontierIndex;
		float weight;
		float distFromSeed;
		float scoreAsSeed;
		bool inSeedConfiguration; // Is this node in the current seed configuration?

		std::vector<Arch> arches;
		NodeStatus status;

		// Stats about the subtree rooted on this node.
		float subtreeWeight;
		float subtreeCost;


		Node(float weight, std::vector<int> neighbors, std::vector<float> edgeWeights) :
			regionId( -1 ),
			parentId( -1 ),
			frontierIndex( -1 ),
			weight( weight ),
			distFromSeed( +INF ),
			scoreAsSeed( -1 ),
			inSeedConfiguration( false ),
			status( Unvisited ), 
			subtreeWeight( 0.0f ),
			subtreeCost ( 0.0f ) {
			assert(neighbors.size() == edgeWeights.size());
			for (int i = 0; i < neighbors.size(); i++)
				arches.push_back(Arch(neighbors[i], edgeWeights[i]));
		}

		bool wasSeed() { return scoreAsSeed >= 0; }
	};

	// When regions are locked, the partitioner won't be able to assign new regions to nodes,
	// and parenting can be changed only among nodes of the same region.
	bool lockRegions = false;
	bool inGreedyPhase = true;
	bool greedyFirstReset = true;
	bool relaxationOver = false;
	std::vector<Node> nodes;
	std::vector<int> seeds;
	std::vector<int> seedsAtStart; // The configuration of seeds at the moment it was generated (or manually set).
	std::vector<int> prevSeeds; // The configuration of seeds at the previous iteration of the relaxation loop.
	std::vector<int> prevMacrostepSeeds; // The configuration of seeds at the end of the previous precise macrostep.
	std::vector<int> frontier; // Implemented as a heap.
	std::vector<int> sorted; // Indices of nodes sorted by their distance from seeds.
	std::vector<float> macrostepScores; // The scores at the beginning of the current precise macrostep.
	std::vector<float> prevMacrostepScores; // The scores at the beginning of the previous precise macrostep.
	PerformanceStatistics dijkstraPerformance;
	PerformanceStatistics greedyPerformance;
	PerformanceStatistics precisePerformance;
	Stopwatch swatch;


	void resetNodes() {
		for (Node& n : nodes) {
			n.inSeedConfiguration = false;

			if (optimizePreciseMicrostep && !inGreedyPhase && lockRegions && !seedChanged(n.regionId)) continue;

			n.distFromSeed = +INF;
			n.parentId = -1;
			n.frontierIndex = -1;
			n.status = Unvisited;
			if (inGreedyPhase) {
				n.regionId = -1;
				if (greedyFirstReset) {
					greedyFirstReset = false;
					n.scoreAsSeed = -1;
				}
			} else if (!lockRegions) {
				n.regionId = -1;
				n.scoreAsSeed = -1;
			}
		}
	}

	void initSeedNodes() {
		for (int i = 0; i < seeds.size(); i++) {
			Node& seed = nodes[seeds[i]];
			seed.inSeedConfiguration = true;

			if (optimizePreciseMicrostep && !inGreedyPhase && lockRegions && !seedChanged(i)) continue;

			seed.regionId = i;
			seed.distFromSeed = 0;
			seed.status = Visited;
			pushFrontier(seeds[i]);
		}
	}

	void initSortedVector() {
		sorted.clear();
		for (int i = 0; i < nodes.size(); i++)
			sorted.push_back(i);
	}

	void swapFrontier(int i, int j) {
		std::swap(frontier[i], frontier[j]);
		std::swap(nodes[frontier[i]].frontierIndex, nodes[frontier[j]].frontierIndex);
	}

	void floatFrontier(int i) {
		int parent = (i - 1) / 2;
		if (parent >= 0 && nodes[frontier[i]].distFromSeed < nodes[frontier[parent]].distFromSeed) {
			swapFrontier(parent, i);
			floatFrontier(parent);
		}
	}

	void pushFrontier(int nodeId) {
		frontier.push_back(nodeId);
		int newFrontierIndex = frontier.size() - 1;
		nodes[nodeId].frontierIndex = newFrontierIndex;
		assert(nodes[nodeId].frontierIndex == frontier.size() - 1);
		floatFrontier(newFrontierIndex);
		assert(std::find(frontier.begin(), frontier.end(), nodeId) - frontier.begin() == nodes[nodeId].frontierIndex);
	}

	void sinkFrontier(int i) {
		int leftChild = 2 * i + 1;
		int rightChild = 2 * i + 2;
		int best = i;
		if (leftChild < frontier.size() && nodes[frontier[best]].distFromSeed > nodes[frontier[leftChild]].distFromSeed)
			best = leftChild;
		if (rightChild < frontier.size() && nodes[frontier[best]].distFromSeed > nodes[frontier[rightChild]].distFromSeed)
			best = rightChild;
		if (best != i) {
			swapFrontier(best, i);
			sinkFrontier(best);
		}
	}

	int popFrontier() {
		int result = frontier.front();
		swapFrontier(0, frontier.size() - 1);
		frontier.pop_back();
		sinkFrontier(0);
		return result;
	}

	void processNode(int i) {
		Node& n = nodes[i];
		n.status = Processed;

		for (Arch e : n.arches) {
			Node& m = nodes[e.target];
			if (lockRegions && m.regionId != n.regionId) continue;
			float elapsedDistance = n.distFromSeed + e.cost;
			if (m.status == Unvisited) {
				m.status = Visited;
				m.distFromSeed = elapsedDistance;
				m.parentId = i;
				m.regionId = n.regionId;
				pushFrontier(e.target);
			} else if (m.status == Visited && m.distFromSeed > elapsedDistance) {
				m.distFromSeed = elapsedDistance;
				m.parentId = i;
				m.regionId = n.regionId;
				floatFrontier(m.frontierIndex);
			}
		}
	}

	void sortNodeIdsByDistance() {
		std::sort(sorted.begin(), sorted.end(), [this] (int i, int j) {
				return nodes[i].distFromSeed < nodes[j].distFromSeed;
			});
	}

	void updateSubtreeInfo() {
		for (Node& n : nodes) {
			n.subtreeWeight = n.weight;
			n.subtreeCost = n.distFromSeed;
		}

		for (int i = sorted.size() - 1; i >= 0; i--) {
			int j = sorted[i];
			int parent = nodes[j].parentId;
			if (parent == -1) continue;
			nodes[parent].subtreeWeight += nodes[j].subtreeWeight;
			nodes[parent].subtreeCost += nodes[j].subtreeCost;
		}
	}

	void scoreAllSeedsGreedy() {
		assert(inGreedyPhase);
		for (int s : seeds)
			nodes[s].scoreAsSeed = 1.0;
	}

	void scoreAllSeedsPrecise() {
		assert(!inGreedyPhase);

		if (!lockRegions)
			prevMacrostepScores = macrostepScores;

		std::vector<float> scores = std::vector<float>(seeds.size(), 0.0f);
		for (Node n : nodes)
			scores[n.regionId] += n.distFromSeed * n.distFromSeed * n.weight;
		for (int s : seeds)
			//if (!nodes[s].wasSeed())
				nodes[s].scoreAsSeed = scores[nodes[s].regionId];

		if (!lockRegions)
			// Keep in mind that when regions are unlocked, every node hasn't been a seed before
			// (nodes[s].wasSeed() always returns false).
			macrostepScores = scores;
	}

	bool seedChanged(int regionId) {
		assert(regionId < seeds.size());
		return regionId >= prevSeeds.size() || prevSeeds[regionId] != seeds[regionId];
	}

	bool macrostepScoreChanged(int regionId) {
		assert(regionId < seeds.size());
		return regionId >= prevMacrostepScores.size() || prevMacrostepScores[regionId] != macrostepScores[regionId];
	}

	float computeHeuristic(int nodeId, int seedId) {
		float heuristic = nodes[nodeId].subtreeCost * nodes[nodeId].subtreeWeight;
		if (greedyRelaxationType == GreedyOption::EXTENDED) {
			for (Arch n : nodes[nodeId].arches) {
				if (nodes[n.target].parentId != seedId) continue;
				heuristic += nodes[n.target].subtreeCost * nodes[n.target].subtreeWeight;
			}
		}
		return heuristic;
	}

	bool moveSeedsGreedy() {
		assert(!lockRegions && inGreedyPhase);
		bool movedSeed = false;
		for (int& s : seeds) {
			float maxHeuristic = -INF;
			int candidate = s;
			for (Arch e : nodes[s].arches) {
				if (nodes[e.target].inSeedConfiguration) continue;
				if (nodes[s].regionId != nodes[e.target].regionId) continue;

				float heuristic = computeHeuristic(e.target, s);
				if (heuristic > maxHeuristic) {
					maxHeuristic = heuristic;
					candidate = e.target;
				}
			}
			if (candidate != s && (!nodes[candidate].wasSeed())) {
				nodes[s].inSeedConfiguration = false;
				s = candidate;
				nodes[s].inSeedConfiguration = true;
				movedSeed = true;
			}
		}
		return movedSeed;
	}

	bool moveSeedsPrecise() {
		assert(!inGreedyPhase && lockRegions);
		bool movedSeed = false;
		for (int& s : seeds) {
			int seedId = static_cast<int>(&s - seeds.data());
			if (optimizePreciseMacrostep && !macrostepScoreChanged(seedId)) continue;

			int candidate = s;
			float minScore = nodes[s].scoreAsSeed;
			float maxHeuristic = -INF;
			for (Arch e : nodes[s].arches) {
				if (nodes[e.target].inSeedConfiguration) continue;
				if (nodes[s].regionId != nodes[e.target].regionId) continue;

				if (nodes[e.target].wasSeed() && nodes[e.target].scoreAsSeed < minScore) {
					minScore = nodes[e.target].scoreAsSeed;
					candidate = e.target;
				} else if (minScore == nodes[s].scoreAsSeed) { // If the fallback mechanism has not been enabled yet...
					float heuristic = computeHeuristic(e.target, s);
					if (!nodes[e.target].wasSeed() && heuristic > maxHeuristic) {
						maxHeuristic = heuristic;
						candidate = e.target;
					}
				}
			}

			if (minScore != nodes[s].scoreAsSeed || maxHeuristic != -INF) {
				nodes[s].inSeedConfiguration = false;
				s = candidate;
				nodes[s].inSeedConfiguration = true;
				movedSeed = true;
			}
		}
		return movedSeed;
	}

	/*
	It makes a vector of seeds of the specifies size.
	Random seeds will be generated from the specified inclusive index.
	No seeds will be randomly generated if such index is out of the new vector's bounds.
	By not generating newly added seed indices, these will be set to -1 (invalid index).
	It returns the size of the seeds' vector actually made.
	*/
	int makeSeedsVector(int size, int generateRandomFrom = 0) {
		assert(size > 0);
		int numberOfNodes = nodes.size();
		if (size > numberOfNodes / 2)
			size = numberOfNodes / 2;

		seeds.resize(size, -1);
		if (generateRandomFrom >= 0 && generateRandomFrom < size) {
			// Set as used the already chosen seeds.
			std::vector<bool> used(numberOfNodes, false);
			for (int i = 0; i < generateRandomFrom; i++)
				used[seeds[i]] = true;

			for (int i = generateRandomFrom; i < size; i++) {
				int s;
				do s = std::rand() % numberOfNodes;
				while (used[s]);
				used[s] = true;
				seeds[i] = s;
			}
		}
		seedsAtStart = seeds;

		return size;
	}

public:
	enum GreedyOption {
		DISABLED,
		ENABLED,
		EXTENDED,
	};

	GreedyOption greedyRelaxationType = GreedyOption::EXTENDED;
	bool optimizePreciseMacrostep = true; // It optimizes the precise macrostep by preventing converged seeds from executing movement routines.
	bool optimizePreciseMicrostep = true; // It optimizes the precise microstep by preventing converged seeds from executing parallel Dijkstra.


	DijkstraPartitioner() {}

	DijkstraPartitioner(std::vector<float> nodeWeights, std::vector<std::vector<int>> neighbors, std::vector<std::vector<float>> edgeWeights, int seedsCount) {
		assert(nodeWeights.size() == neighbors.size() && neighbors.size() == edgeWeights.size() && edgeWeights.size() == nodeWeights.size());
		resetState();
		generateNodes(nodeWeights, neighbors, edgeWeights);
		makeSeedsVector(seedsCount);
	}

	void resetState() {
		lockRegions = false;
		inGreedyPhase = greedyRelaxationType == GreedyOption::DISABLED ? false : true;
		greedyFirstReset = true;
		relaxationOver = false;
		prevSeeds.clear();
		prevMacrostepSeeds.clear();
		macrostepScores.clear();
		prevMacrostepScores.clear();
		dijkstraPerformance.reset();
		greedyPerformance.reset();
		precisePerformance.reset();
	}

	void generateNodes(std::vector<float> nodeWeights, std::vector<std::vector<int>> neighbors, std::vector<std::vector<float>> edgeWeights) {
		assert(nodeWeights.size() == neighbors.size() && neighbors.size() == edgeWeights.size() && edgeWeights.size() == nodeWeights.size());
		nodes.clear();
		for (int i = 0; i < neighbors.size(); i++)
			nodes.push_back(Node(nodeWeights[i], neighbors[i], edgeWeights[i]));
		initSortedVector();
	}

	int generateRandomSeeds() {
		return makeSeedsVector(seeds.size());
	}

	void partitionNodes() {
		swatch.begin();
		frontier.clear();
		resetNodes();
		initSeedNodes();

		while (!frontier.empty()) {
			int i = popFrontier();
			processNode(i);
		}

		assert(frontier.empty());
		dijkstraPerformance.recordIteration(swatch.end());
	}

	void relaxSeedsOnce() {
		if (relaxationOver) return;
		prevSeeds = seeds;

		sortNodeIdsByDistance();
		updateSubtreeInfo();
		if (inGreedyPhase) {
			swatch.begin();
			scoreAllSeedsGreedy();
			inGreedyPhase = moveSeedsGreedy();
			greedyPerformance.recordIteration(swatch.end());
		} else {
			if (!lockRegions) swatch.begin();
			scoreAllSeedsPrecise();
			lockRegions = true; // ALWAYS lock regions before moving seeds
			lockRegions = moveSeedsPrecise();
			if (!lockRegions) {
				if (prevMacrostepSeeds != seeds) prevMacrostepSeeds = seeds;
				else relaxationOver = true;
				precisePerformance.recordIteration(swatch.end());
			}
		}
	}

	void relaxSeeds() {
		while (!relaxationOver) {
			relaxSeedsOnce();
			partitionNodes();
		}
	}

	void moveSeedsRandomly() {
		resetState();
		for (int& s : seeds) {
			int random = rand() % nodes[s].arches.size();
			s = nodes[s].arches[random].target;
		}
	}

	bool moveSeedToNode(int nodeId) {
		assert(nodeId >= 0 && nodeId < nodes.size());
		if (isNodeInSeedConfiguration(nodeId)) return false;

		seeds[nodes[nodeId].regionId] = nodeId;
		seedsAtStart = seeds;
		return true;
	}

	bool addSeed(int nodeId) {
		assert(nodeId >= 0 && nodeId < nodes.size());
		if (isNodeInSeedConfiguration(nodeId)) return false;

		seeds.push_back(nodeId);
		seedsAtStart = seeds;
		return true;
	}

	bool removeSeedOfNode(int nodeId) {
		assert(nodeId >= 0 && nodeId < nodes.size());
		if (seeds.size() <= 1) return false;

		seeds.erase(seeds.begin() + nodes[nodeId].regionId);
		seedsAtStart = seeds;
		return true;
	}

	void restoreSeeds() {
		seeds = seedsAtStart;
	}

	void nodewiseRegionAssignments(std::vector<int>& assignments) const {
		assert(assignments.size() == 0);
		for (Node n : nodes)
			assignments.push_back(n.regionId);
	}

	void treeEdges(std::vector<std::pair<int, int>>& treeEdges) const {
		assert(treeEdges.size() == 0);
		for (int i = 0; i < nodes.size(); i++)
			if (nodes[i].parentId != -1)
				treeEdges.push_back(std::pair<int, int>(i, nodes[i].parentId));
	}

	bool isNodeInSeedConfiguration(int nodeId) const {
		assert(nodeId >= 0 && nodeId < nodes.size());
		return nodes[nodeId].inSeedConfiguration;
	}

	int getNodesCount() const {
		return nodes.size();
	}

	const std::vector<int>& getSeeds() const {
		return seeds;
	}

	const PerformanceStatistics& getDijkstraPerformance() const {
		return dijkstraPerformance;
	}

	const PerformanceStatistics& getGreedyPerformance() const {
		return greedyPerformance;
	}

	const PerformanceStatistics& getPrecisePerformance() const {
		return precisePerformance;
	}

	void setSeeds(const std::vector<int>& newSeeds) {
		seeds = newSeeds;
		seedsAtStart = seeds;
	}

	int setSeedsCount(int seedsCount) {
		assert(seedsCount > 0);
		
		int oldCount = seeds.size();
		if (seedsCount == oldCount)
			return seedsCount;
		else if (seedsCount < oldCount)
			return makeSeedsVector(seedsCount, -1);
		else
			return makeSeedsVector(seedsCount, oldCount);
	}
};

#endif