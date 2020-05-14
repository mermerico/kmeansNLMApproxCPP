#include <vector>
#include <random>
#include "expandDims.cpp"
#include "bisecting_kmeans.cpp"
#include "calcW_generic.cpp"
#include "calcC1ChanT.cpp"
#include "highDimBoxFilter.cpp"


void kMeansNLMApprox(float* const I, int numClusters, const float h, const int sizeX, const int sizeY, float* output) {

	constexpr int radius = 1;
	constexpr int patchSize = (2 * radius + 1) * (2 * radius + 1);
	constexpr int numChannels = 3;

	std::vector<float> expandedDimensions(sizeY * sizeX * numChannels * patchSize);
	expandDims(I, radius, sizeX, sizeY, expandedDimensions.data());

	constexpr int numPatchesToSample = 1e3;
	std::minstd_rand0 generator(0);
	std::uniform_int_distribution<> distribution(0, sizeY * sizeX - 1);
	std::vector<int> sampledLocs(numPatchesToSample);
	for (auto& sample : sampledLocs) {
		sample = distribution(generator);
	}
	/*
	std::vector<int> sampledLocs(numPatchesToSample);
	for (int i = 0; i < numPatchesToSample; i++) {
		sampledLocs[i] = i * 90;
	}
	*/
	std::vector<float> sampledPatches(numPatchesToSample * patchSize * numChannels);
	for (int dIdx = 0; dIdx < (patchSize*numChannels); dIdx++) {
		for (int pIdx = 0; pIdx < numPatchesToSample; pIdx++) {
			sampledPatches[pIdx + dIdx*numPatchesToSample] = expandedDimensions[sampledLocs[pIdx] + dIdx*sizeX*sizeY];
		}
	}

	int maxClusters = 100;
	float threshold = 3e-3;
	std::vector<float> clusterCenters = bisecting_kmeans(sampledPatches.data(), numPatchesToSample, patchSize*numChannels, maxClusters, threshold);
	numClusters = clusterCenters.size() / (patchSize*numChannels);

	std::vector<float> W(sizeX * sizeY * numClusters);
	calcW(expandedDimensions.data(), clusterCenters.data(), sizeY, sizeX, patchSize*3, numClusters, h, W.data());

	std::vector<float> C1ChanT = calcC1ChanT(clusterCenters, numClusters, h);

	int S_ignored = 0;
	highDimBoxFilter(I, W.data(), C1ChanT.data(), sizeY, sizeX, numClusters, S_ignored, output);
}