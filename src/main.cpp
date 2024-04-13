#include <chrono>
#include <vector>
#include <map>
#include <exception>
#include <string>
#include <cstddef>
#include <algorithm>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <igl/opengl/glfw/Viewer.h>
#include <igl/opengl/glfw/imgui/ImGuiPlugin.h>
#include <igl/opengl/glfw/imgui/ImGuiMenu.h>
#include <igl/opengl/glfw/imgui/ImGuiHelpers.h>
#include <glm/glm.hpp>
#include <glm/gtx/color_space.hpp>
#include "dijkstra_partitioner.hpp"
#include "model.hpp"
#include "consts.hpp"

#define REGIONS_NUMBER 53


DijkstraPartitioner partitioner;
Model model;
std::chrono::steady_clock::time_point beginTimeStamp;
std::chrono::steady_clock::time_point endTimeStamp;
bool denseGraph = false;
bool showGroundTruth = false;


void beginChrono() {
    beginTimeStamp = std::chrono::steady_clock::now();
}

void endChrono(const std::string taskName) {
    endTimeStamp = std::chrono::steady_clock::now();
    std::cout << taskName << ": " << std::chrono::duration_cast<std::chrono::milliseconds>(endTimeStamp - beginTimeStamp).count() << "ms" << "\n";
}

void groundTruth(igl::opengl::ViewerData& data, std::vector<int>& regionAssignments, Eigen::MatrixXd& barycenters) {
    assert(showGroundTruth);

    std::vector<glm::vec3> nodePos;
    std::vector<float> nodeArea;
    if (data.face_based) {
        model.triangleBarycenters(nodePos);
        model.triangleAreas(nodeArea);
    } else {
        model.verticesPositions(nodePos);
        model.vertexSurroundingAreas(nodeArea);
    }
    const std::vector<int>& seeds = partitioner.getSeeds();
    regionAssignments.clear();
    regionAssignments.resize(nodePos.size());

    for (int node = 0; node < nodePos.size(); node++) {
        float minDist = +INF;
        for (int region = 0; region < seeds.size(); region++) {
            float newDist = glm::distance(nodePos[seeds[region]], nodePos[node]);
            if (newDist < minDist) {
                minDist = newDist;
                regionAssignments[node] = region;
            }
        }
    }

    assert(barycenters.rows() == seeds.size() && barycenters.cols() == 3);
    std::vector<float> regionCount(seeds.size(), 0);
    for (int node = 0; node < nodePos.size(); node++) {
        int region = regionAssignments[node];
        regionCount[region] += nodeArea[node];
        barycenters.row(region) += Eigen::RowVector3d(nodePos[node].x, nodePos[node].y, nodePos[node].z) * nodeArea[node];
    }
    for (int region = 0; region < seeds.size(); region++)
        barycenters.row(region) /= regionCount[region];
}

void regionColors(int regionsNumber, std::vector<glm::vec3>& colors) {
    colors.clear();
    colors.resize(regionsNumber);
    for (int c = 0; c < regionsNumber; c++) {
        glm::vec3 hsvCol = glm::vec3(float(c) / regionsNumber * 360.0f, 1.0f, 1.0f);
        glm::vec3 rgbCol = glm::rgbColor(hsvCol);
        colors[c] = rgbCol;
    }
}

void assignColorsToModel(igl::opengl::ViewerData& data, const std::vector<int>& regionAssignments) {
    Eigen::MatrixXd C(regionAssignments.size(), 3);
    std::vector<glm::vec3> regionCols;
    regionColors(partitioner.getSeeds().size(), regionCols);

    for (int node = 0; node < C.rows(); node++) {
        glm::vec3 nodeColor = regionCols[regionAssignments[node]];
        C.row(node) << nodeColor.x, nodeColor.y, nodeColor.z;
    }
    data.set_colors(C);
}

void plotTrees(igl::opengl::ViewerData& data) {
    std::vector<glm::vec3> nodePos;
    if (data.face_based) model.triangleBarycenters(nodePos);
    else model.verticesPositions(nodePos);

    std::vector<int> seeds = partitioner.getSeeds();
    Eigen::MatrixXd S(seeds.size(), 3);
    for (int s = 0; s < seeds.size(); s++)
        S.row(s) << nodePos[seeds[s]].x, nodePos[seeds[s]].y, nodePos[seeds[s]].z;
    data.add_points(S, Eigen::RowVector3d(1.0, 1.0, 1.0));

    std::vector<std::pair<int, int>> treeEdges;
    partitioner.treeEdges(treeEdges);
    Eigen::MatrixXd P0(treeEdges.size(), 3);
    Eigen::MatrixXd P1(treeEdges.size(), 3);
    for (int e = 0; e < treeEdges.size(); e++) {
        int e0 = treeEdges[e].first;
        int e1 = treeEdges[e].second;
        P0.row(e) << nodePos[e0].x, nodePos[e0].y, nodePos[e0].z;
        P1.row(e) << nodePos[e1].x, nodePos[e1].y, nodePos[e1].z;
    }
    data.add_edges(P0, P1, Eigen::RowVector3d(1.0, 1.0, 1.0));
}

void plotDataOnScreen(igl::opengl::ViewerData& data) {
    data.clear_points();
    data.clear_edges();

    std::vector<int> regionAssignments;
    if (showGroundTruth) {
        Eigen::MatrixXd barycenters(partitioner.getSeeds().size(), 3);
        groundTruth(data, regionAssignments, barycenters);
        data.add_points(barycenters, Eigen::RowVector3d(0, 0, 0));
    } else {
        plotTrees(data);
        partitioner.nodewiseRegionAssignments(regionAssignments);
    }
    assignColorsToModel(data, regionAssignments);
}

void runPartitioner(igl::opengl::ViewerData& data) {
    partitioner.resetState();
    partitioner.partitionNodes();
    plotDataOnScreen(data);
}

void generateSeeds(igl::opengl::ViewerData &data) {
    partitioner.generateRandomSeeds();
    runPartitioner(data);
}

void relaxPartitioner(igl::opengl::ViewerData& data) {
    partitioner.relaxSeeds();
    partitioner.partitionNodes();
    plotDataOnScreen(data);
}

void moveSeedsRandomly(igl::opengl::ViewerData& data) {
    partitioner.moveSeedsRandomly();
    partitioner.partitionNodes();
    plotDataOnScreen(data);
}

// Maybe you can merge switchGraphType and switchGraphDensity into a single function...
void switchGraphType(igl::opengl::ViewerData& data) {
    std::vector<int> newSeeds;
    std::vector<float> nodeWeights;
    std::vector<std::vector<int>> adjacency;
    std::vector<std::vector<float>> edgeWeights;
    if (data.face_based) {
        model.trianglesFromVertices(partitioner.getSeeds(), newSeeds);
        model.triangleAreas(nodeWeights);
        if (denseGraph) model.denseTriangleAdjacency(adjacency, edgeWeights);
        else model.triangleAdjacency(adjacency, edgeWeights);
    } else {
        model.firstVerticesFromTriangles(partitioner.getSeeds(), newSeeds);
        model.vertexSurroundingAreas(nodeWeights);
        if (denseGraph) model.denseVertexAdjacency(adjacency, edgeWeights);
        else model.vertexAdjacency(adjacency, edgeWeights);
    }
    partitioner.setSeeds(newSeeds);
    partitioner.generateNodes(nodeWeights, adjacency, edgeWeights);
    runPartitioner(data);
}

// Maybe you can merge switchGraphType and switchGraphDensity into a single function...
void switchGraphDensity(igl::opengl::ViewerData& data) {
    std::vector<float> nodeWeights;
    std::vector<std::vector<int>> adjacency;
    std::vector<std::vector<float>> edgeWeights;
    if (data.face_based) {
        model.triangleAreas(nodeWeights);
        if (denseGraph) model.denseTriangleAdjacency(adjacency, edgeWeights);
        else model.triangleAdjacency(adjacency, edgeWeights);
    } else {
        model.vertexSurroundingAreas(nodeWeights);
        if (denseGraph) model.denseVertexAdjacency(adjacency, edgeWeights);
        else model.vertexAdjacency(adjacency, edgeWeights);
    }
    partitioner.generateNodes(nodeWeights, adjacency, edgeWeights);
    runPartitioner(data);
}

bool keyboardInputCallback(igl::opengl::glfw::Viewer& viewer, unsigned char key, int modifier) {
    switch (key) {
        case 'C':
        case 'c':
            denseGraph = !denseGraph;
            switchGraphDensity(viewer.data());
            return true;

        case 'F':
        case 'f':
            viewer.data().set_face_based(!viewer.data().face_based);
            switchGraphType(viewer.data());
            return true;

        case 'G':
        case 'g':
            generateSeeds(viewer.data());
            return true;

        case 'N':
        case 'n':
            moveSeedsRandomly(viewer.data());
            return true;

        case 'R':
        case 'r':
            if (modifier == 0) {
                relaxPartitioner(viewer.data());
                return true;
            }
            break;

        case 'V':
        case 'v':
            viewer.core().is_animating = false;
            showGroundTruth = !showGroundTruth;
            plotDataOnScreen(viewer.data());
            return true;
    }
    return false;
}

bool preDrawCallback(igl::opengl::glfw::Viewer& viewer) {
    if (viewer.core().is_animating)
        relaxPartitioner(viewer.data());
    return false;
}

int main(int argc, char* argv[]) {
    std::string modelPath = argc > 1 ? std::string(argv[1]) : "./models/plane.obj";
    if (!model.load(modelPath)) return -1;

    std::vector<float> nodeWeights;
    model.vertexSurroundingAreas(nodeWeights);
    std::vector<std::vector<int>> adjacency;
    std::vector<std::vector<float>> edgeWeights;
    if (denseGraph) model.denseVertexAdjacency(adjacency, edgeWeights);
    else model.vertexAdjacency(adjacency, edgeWeights);

    igl::opengl::glfw::Viewer viewer;
    std::cout << "DijkstraPartitioner usage:\n"
        << "  C,c     Toggle dense graph\n"
        << "  G,g     Generate random seeds\n"
        << "  N,n     Move seeds randomly\n"
        << "  R,r     Relax seeds once\n"
        << "  V,v     Toggle ground truth\n";

    // Import ImGui plugin.
    igl::opengl::glfw::imgui::ImGuiPlugin plugin;
    viewer.plugins.push_back(&plugin);
    igl::opengl::glfw::imgui::ImGuiMenu menu;
    plugin.widgets.push_back(&menu);

    // Viewer setup.
    viewer.data().set_mesh(model.getVerticesMatrix(), model.getFacesMatrix());
    viewer.core().is_animating = false;
    viewer.core().animation_max_fps = 60.;
    viewer.core().lighting_factor = 1;
    viewer.data().point_size = 8.;
    viewer.data().show_lines = false;

    // Callback setup.
    viewer.callback_key_pressed = &keyboardInputCallback; // Do NOT mistake it with viewer.callback_key_down
    viewer.callback_pre_draw = &preDrawCallback;
    menu.callback_draw_viewer_menu = [&]() {
        menu.draw_viewer_menu();
        if (ImGui::CollapsingHeader("New Group", ImGuiTreeNodeFlags_DefaultOpen)) {
            static bool boolVariable = true;
            if (ImGui::Checkbox("bool", &boolVariable)) {
                // do something
                std::cout << "boolVariable: " << std::boolalpha << boolVariable << std::endl;
            }
        }
    };

    partitioner = DijkstraPartitioner(nodeWeights, adjacency, edgeWeights, REGIONS_NUMBER);
    viewer.data().set_face_based(false);
    runPartitioner(viewer.data());
    viewer.launch();

    return 0;
}
