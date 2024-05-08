#include <vector>
#include <map>
#include <string>
#include <algorithm>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <igl/opengl/glfw/Viewer.h>
#include <igl/opengl/glfw/imgui/ImGuiPlugin.h>
#include <igl/opengl/glfw/imgui/ImGuiMenu.h>
#include <igl/opengl/glfw/imgui/ImGuiHelpers.h>
#include <igl/unproject_onto_mesh.h>
#include <glm/glm.hpp>
#include <glm/gtx/color_space.hpp>
#include <glm/gtc/type_ptr.hpp>
#include "dijkstra_partitioner.hpp"
#include "model.hpp"
#include "stopwatch.hpp"
#include "performance_stats.hpp"
#include "consts.hpp"

#define DEFAULT_REGIONS_NUMBER 16

enum GraphType {
    VERTEX,
    VERTEX_DENSE,
    TRIANGLE,
    TRIANGLE_DENSE,
    MIXED,
    GRAPH_TYPE_SIZE,
};

DijkstraPartitioner partitioner;
Model model;
Stopwatch swatch;
GraphType graphType;
float triangulationFactor = 0;
glm::vec3 triangulationColor{0.7f, 0.7f, 0.7f};
bool showGroundTruth = false;
bool isDraggingRegion = false;


inline bool isVertexBased(GraphType type) {
    return type == GraphType::VERTEX || type == GraphType::VERTEX_DENSE;
}

inline bool isTriangleBased(GraphType type) {
    return type == GraphType::TRIANGLE || type == GraphType::TRIANGLE_DENSE;
}

inline bool isMixed(GraphType type) {
    return type == GraphType::MIXED;
}

inline bool isDense(GraphType type) {
    return type == GraphType::VERTEX_DENSE || type == GraphType::TRIANGLE_DENSE;
}

void printPerformanceTestResults(igl::opengl::ViewerData& data, double runningTime) {
    std::cout << "-----------\n";
    std::cout << "Voronoi: " << runningTime << "ms\n";
    std::cout << "Graph type: ";
    switch (graphType) {
        case GraphType::VERTEX:
            std::cout << "vertex-based\n";
            break;
        case GraphType::VERTEX_DENSE:
            std::cout << "vertex-based (dense)\n";
            break;
        case GraphType::TRIANGLE:
            std::cout << "triangle-based\n";
            break;
        case GraphType::TRIANGLE_DENSE:
            std::cout << "triangle-based (dense)\n";
            break;
        case GraphType::MIXED:
            std::cout << "mixed\n";
            break;
    }
    std::cout << "Seeds count: " << partitioner.getSeeds().size() << "\n";
    const PerformanceStatistics& dPerf = partitioner.getDijkstraPerformance();
    const PerformanceStatistics& gPerf = partitioner.getGreedyPerformance();
    const PerformanceStatistics& pPerf = partitioner.getPrecisePerformance();
    auto printStats = [&](const char* header, const PerformanceStatistics& perf) {
        std::cout << header
            << "iterations: " << perf.getIterations()
            << ", min: " << perf.getMinTime() << "ms"
            << ", max: " << perf.getMaxTime() << "ms"
            << ", avg: " << perf.getAvgTime() << "ms\n";
    };
    if (partitioner.greedyRelaxationType != DijkstraPartitioner::GreedyOption::DISABLED) {
        printStats(partitioner.greedyRelaxationType == DijkstraPartitioner::GreedyOption::EXTENDED ? "Greedy (ext):\t" : "Greedy:\t\t", gPerf);
    } else std::cout << "Greedy:\t\tDISABLED\n";
    printStats(partitioner.optimizePreciseRelaxation ? "Precise (opt):\t" : "Precise:\t", pPerf);
    printStats(partitioner.optimizeDijkstra ? "Dijkstra (opt):\t" : "Dijkstra:\t", dPerf);
    std::cout << "\n";
}

void groundTruth(std::vector<int>& regionAssignments, Eigen::MatrixXd& barycenters) {
    assert(showGroundTruth);

    std::vector<glm::vec3> nodePos;
    std::vector<float> nodeArea;
    if (graphType == GraphType::TRIANGLE || graphType == GraphType::TRIANGLE_DENSE) {
        model.trianglesBarycenters(nodePos);
        model.triangleAreas(nodeArea);
    } else { // Ground truth for mixed graph is vertex based.
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

void pseudoDelaunay(igl::opengl::ViewerData& data, const std::vector<int>& regionAssignments) {
    std::vector<glm::vec3> seedPoints;
    const Eigen::MatrixXd& V = model.getVerticesMatrix();
    const Eigen::MatrixXi& F = model.getFacesMatrix();
    for (int seed : partitioner.getSeeds()) {
        glm::vec3 point;
        if (isTriangleBased(graphType))
            model.triangleBarycenter(seed, point);
        else if (isMixed(graphType) && seed >= V.rows())
            model.triangleBarycenter(seed - V.rows(), point);
        else
            model.vertexPosition(seed, point);
        
        seedPoints.push_back(point);
    }

    glm::vec3 mixed;
    glm::vec3 vertexPos;
    auto mixForDelaunay = [&](int vertexId, int region) {
        model.vertexPosition(vertexId, vertexPos);
        mixed = glm::mix(vertexPos, seedPoints[region], triangulationFactor);
    };
    for (int node = 0; node < partitioner.getNodesCount(); node++) {
        int region = regionAssignments[node];

        if (isTriangleBased(graphType)) {
            for (int v = 0; v < F.cols(); v++) {
                mixForDelaunay(F(node, v), region);
                data.V.row(F(node, v)) << mixed.x, mixed.y, mixed.z;
            }
        } else if (isMixed(graphType) && node >= V.rows()) {
            for (int v = 0; v < F.cols(); v++) {
                mixForDelaunay(F(node - V.rows(), v), region);
                data.V.row(F(node - V.rows(), v)) << mixed.x, mixed.y, mixed.z;
            }
        } else {
            mixForDelaunay(node, region);
            data.V.row(node) << mixed.x, mixed.y, mixed.z;
        }
    }
    data.dirty |= igl::opengl::MeshGL::DIRTY_POSITION;
    data.compute_normals();
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
    Eigen::MatrixXd C(isMixed(graphType) ? model.getVerticesMatrix().rows() : regionAssignments.size(), 3);
    std::vector<glm::vec3> regionCols;
    regionColors(partitioner.getSeeds().size(), regionCols);

    for (int node = 0; node < C.rows(); node++) {
        glm::vec3 nodeColor = glm::mix(regionCols[regionAssignments[node]], triangulationColor, triangulationFactor);
        C.row(node) << nodeColor.x, nodeColor.y, nodeColor.z;
    }
    data.set_colors(C);
}

void plotTrees(igl::opengl::ViewerData& data) {
    std::vector<glm::vec3> nodePos;
    if (isTriangleBased(graphType))
        model.trianglesBarycenters(nodePos);
    else if (isVertexBased(graphType))
        model.verticesPositions(nodePos);
    else model.verticesAndTrianglesPositions(nodePos);

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
        Eigen::MatrixXd barycenters = Eigen::MatrixXd::Zero(partitioner.getSeeds().size(), 3);
        groundTruth(regionAssignments, barycenters);
        data.add_points(barycenters, Eigen::RowVector3d(0, 0, 0));
    } else
        partitioner.nodewiseRegionAssignments(regionAssignments);
    if (triangulationFactor > 0)
        pseudoDelaunay(data, regionAssignments);
    if (!showGroundTruth && triangulationFactor <= 0)
        plotTrees(data);
    assignColorsToModel(data, regionAssignments);
}

void initPartitioner() {
    std::vector<float> weights;
    std::vector<std::vector<int>> adjacency;
    std::vector<std::vector<float>> costs;
    switch (graphType) {
        case GraphType::TRIANGLE:
            model.triangleAreas(weights);
            model.triangleAdjacency(adjacency, costs);
            break;
        case GraphType::TRIANGLE_DENSE:
            model.triangleAreas(weights);
            model.denseTriangleAdjacency(adjacency, costs);
            break;
        case GraphType::VERTEX:
            model.vertexSurroundingAreas(weights);
            model.vertexAdjacency(adjacency, costs);
            break;
        case GraphType::VERTEX_DENSE:
            model.vertexSurroundingAreas(weights);
            model.denseVertexAdjacency(adjacency, costs);
            break;
        case GraphType::MIXED:
            model.vertexAndTriangleAreas(weights);
            model.mixedAdjacency(adjacency, costs);
            break;
    }

    int regionsNumber = partitioner.getSeeds().size();
    if (regionsNumber == 0) regionsNumber = DEFAULT_REGIONS_NUMBER;
    partitioner = DijkstraPartitioner(weights, adjacency, costs, regionsNumber);
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

void moveSeedsRandomly(igl::opengl::ViewerData& data) {
    partitioner.moveSeedsRandomly();
    partitioner.partitionNodes();
    plotDataOnScreen(data);
}

void relaxPartitioner(igl::opengl::ViewerData& data, bool plotData = true) {
    swatch.begin();
    partitioner.resetState();
    partitioner.fullRelaxation();
    printPerformanceTestResults(data, swatch.end());
    if (plotData) plotDataOnScreen(data);
}

void relaxPartitionerOnce(igl::opengl::ViewerData& data) {
    partitioner.relaxSeeds();
    partitioner.partitionNodes();
    plotDataOnScreen(data);
}

// Returns true if seeds have been converted.
bool convertSeeds(GraphType targetType, std::vector<int>& newSeeds) {
    if (isTriangleBased(targetType) && isVertexBased(graphType)) {
        model.trianglesFromVertices(partitioner.getSeeds(), newSeeds);
        return true;
    }
    else if (isVertexBased(targetType) && isTriangleBased(graphType)) {
        model.firstVerticesFromTriangles(partitioner.getSeeds(), newSeeds);
        return true;
    }
    else if (isMixed(targetType)) {
        newSeeds = partitioner.getSeeds();
        if (isTriangleBased(graphType)) {
            for (int& s : newSeeds)
                s += model.getVerticesMatrix().rows();
        }
        return true;
    }
    else if (isMixed(graphType)) {
        newSeeds = partitioner.getSeeds();
        for (int& s : newSeeds) {
            if (isTriangleBased(targetType)) {
                if (s >= model.getVerticesMatrix().rows()) {
                    s -= model.getVerticesMatrix().rows();
                }
                else s = model.triangleFromVertex(s);
            }
            else if (s >= model.getVerticesMatrix().rows()) {
                s = model.firstVertexFromTriangle(s - model.getVerticesMatrix().rows());
            }
        }
        return true;
    }

    return false;
}

void setGraphMetrics(GraphType targetType, std::vector<float>& weights, std::vector<std::vector<int>>& adjacency, std::vector<std::vector<float>>& costs) {
    if (isTriangleBased(targetType)) {
        model.triangleAreas(weights);
        if (isDense(targetType)) model.denseTriangleAdjacency(adjacency, costs);
        else model.triangleAdjacency(adjacency, costs);
    } else if (isVertexBased(targetType)) {
        model.vertexSurroundingAreas(weights);
        if (isDense(targetType)) model.denseVertexAdjacency(adjacency, costs);
        else model.vertexAdjacency(adjacency, costs);
    } else {
        model.vertexAndTriangleAreas(weights);
        model.mixedAdjacency(adjacency, costs);
    }
}

void setGraphType(igl::opengl::ViewerData& data, GraphType targetType) {
    if (targetType == graphType) return;

    std::vector<int> newSeeds;
    if (convertSeeds(targetType, newSeeds))
        partitioner.setSeeds(newSeeds);

    std::vector<float> weights;
    std::vector<std::vector<int>> adjacency;
    std::vector<std::vector<float>> costs;
    setGraphMetrics(targetType, weights, adjacency, costs);

    graphType = targetType;
    data.set_face_based(isTriangleBased(targetType));
    partitioner.generateNodes(weights, adjacency, costs);
    runPartitioner(data);
}

void runPerformanceTest(igl::opengl::ViewerData& data) {
    // Save parameters before the test.
    partitioner.restoreSeeds();
    GraphType typeAtStart = graphType;
    DijkstraPartitioner::GreedyOption greedyAtStart = partitioner.greedyRelaxationType;
    bool preciseAtStart = partitioner.optimizePreciseRelaxation;
    bool dijkstraAtStart = partitioner.optimizeDijkstra;

    // Do all tests.
    // Forgive me, Father.
    for (int type = 0; type < (int)GraphType::GRAPH_TYPE_SIZE; type++) {
        setGraphType(data, (GraphType)type);
        for (int greedy = 0; greedy < 3; greedy++)
            for (int precise = 0; precise < 2; precise++)
                for (int dijkstra = 0; dijkstra < 2; dijkstra++) {
                    partitioner.greedyRelaxationType = (DijkstraPartitioner::GreedyOption)greedy;
                    partitioner.optimizePreciseRelaxation = (bool)precise;
                    partitioner.optimizeDijkstra = (bool)dijkstra;
                    relaxPartitioner(data, false);
                    partitioner.restoreSeeds();
                }
    }
    plotDataOnScreen(data);

    // Restore previous parameters.
    setGraphType(data, typeAtStart);
    partitioner.greedyRelaxationType = greedyAtStart;
    partitioner.optimizePreciseRelaxation = preciseAtStart;
    partitioner.optimizeDijkstra = dijkstraAtStart;
}

void viewerSetup(igl::opengl::glfw::Viewer& viewer) {
    setGraphType(viewer.data(), GraphType::VERTEX);
    viewer.data().point_size = 8.;
    viewer.data().show_lines = false;
    viewer.core().is_animating = false;
    viewer.core().animation_max_fps = 120.;
    viewer.core().lighting_factor = 1;
}

bool keyboardInputCallback(igl::opengl::glfw::Viewer& viewer, unsigned char key, int modifier) {
    switch (key) {
        case 'F':
        case 'f':
            setGraphType(viewer.data(), viewer.data().face_based ? GraphType::VERTEX : GraphType::TRIANGLE);
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
            relaxPartitionerOnce(viewer.data());
            return true;
            break;

        case 'V':
        case 'v':
            showGroundTruth = !showGroundTruth;
            plotDataOnScreen(viewer.data());
            return true;
    }
    return false;
}

bool mouseDownCallback(igl::opengl::glfw::Viewer& viewer, int button, int modifier) {
    if (button != 0 /* Left Mouse Button */ || modifier == 0) return false;
    
    float x = viewer.current_mouse_x;
    float y = viewer.core().viewport(3) - viewer.current_mouse_y;
    int pickedTriangle;
    Eigen::Vector3f barycentricCoords;
    if (igl::unproject_onto_mesh(Eigen::Vector2f(x, y), viewer.core().view, viewer.core().proj, viewer.core().viewport, model.getVerticesMatrix(),
        model.getFacesMatrix(), pickedTriangle, barycentricCoords)) {

        int pickedVertex = model.vertexFromBarycentricCoords(pickedTriangle, barycentricCoords);
        int pickedTriangleMixed = pickedTriangle + model.getVerticesMatrix().rows();
        switch (modifier) {
            case IGL_MOD_CONTROL:
                if (isTriangleBased(graphType) && partitioner.addSeed(pickedTriangle) ||
                    (isVertexBased(graphType) || isMixed(graphType)) && partitioner.addSeed(pickedVertex) ||
                    isMixed(graphType) && partitioner.addSeed(pickedTriangleMixed)) {
                    runPartitioner(viewer.data());
                }
                return true;
            case IGL_MOD_CONTROL | IGL_MOD_SHIFT:
                if (isTriangleBased(graphType) && partitioner.removeSeedOfNode(pickedTriangle) ||
                    (isVertexBased(graphType) || isMixed(graphType)) && partitioner.removeSeedOfNode(pickedVertex) ||
                    isMixed(graphType) && partitioner.removeSeedOfNode(pickedTriangleMixed)) {
                    runPartitioner(viewer.data());
                }
                return true;
            case IGL_MOD_ALT:
                if (isTriangleBased(graphType) && partitioner.moveSeedToNode(pickedTriangle) ||
                    (isVertexBased(graphType) || isMixed(graphType)) && partitioner.moveSeedToNode(pickedVertex) ||
                    isMixed(graphType) && partitioner.moveSeedToNode(pickedTriangleMixed)) {
                    isDraggingRegion = true;
                    runPartitioner(viewer.data());
                }
                return true;
        }
    }
    return false;
}

bool mouseMoveCallback(igl::opengl::glfw::Viewer& viewer, int x, int y) {
    if (!isDraggingRegion) return false;

    x = viewer.current_mouse_x;
    y = viewer.core().viewport(3) - viewer.current_mouse_y;
    int pickedTriangle;
    Eigen::Vector3f barycentricCoords;
    if (igl::unproject_onto_mesh(Eigen::Vector2f(x, y), viewer.core().view, viewer.core().proj, viewer.core().viewport, model.getVerticesMatrix(),
        model.getFacesMatrix(), pickedTriangle, barycentricCoords)) {

        int pickedVertex = model.vertexFromBarycentricCoords(pickedTriangle, barycentricCoords);
        int pickedTriangleMixed = pickedTriangle + model.getVerticesMatrix().rows();
        if (isTriangleBased(graphType) && partitioner.moveSeedToNode(pickedTriangle) ||
            (isVertexBased(graphType) || isMixed(graphType)) && partitioner.moveSeedToNode(pickedVertex) ||
            isMixed(graphType) && partitioner.moveSeedToNode(pickedTriangleMixed))
            runPartitioner(viewer.data());
        return true;
    }
    else isDraggingRegion = false;
    return false;
}

bool mouseUpCallback(igl::opengl::glfw::Viewer& viewer, int button, int modifier) {
    isDraggingRegion = false;
    return false;
}

bool preDrawCallback(igl::opengl::glfw::Viewer& viewer) {
    if (viewer.core().is_animating)
        relaxPartitionerOnce(viewer.data());
    return false;
}

int main(int argc, char* argv[]) {
    igl::opengl::glfw::Viewer viewer;
    std::cout << "\nDijkstraPartitioner usage:\n"
        << "  G,g                          Generate random seeds\n"
        << "  N,n                          Move seeds randomly\n"
        << "  R,r                          Relax seeds once\n"
        << "  V,v                          Toggle ground truth\n"
        << "  CTRL + Left Click            Add clicked node to seeds\n"
        << "  SHIFT + CTRL + Left Click    Remove clicked region\n"
        << "  ALT + Left Click             Drag region\n\n";

    // Load mesh from file.
    std::string modelPath = argc > 1 ? std::string(argv[1]) : "./models/plane.obj";
    if (!viewer.load_mesh_from_file(modelPath)) return -1;
    model = Model(viewer.data().V, viewer.data().F, viewer.data().V_normals);

    // Dijkstra partitioner.
    //model.fillTangentSpaceFlat(glm::vec3(2, 0, 0), glm::vec3(0, 1, 0)); // TODO: erase this line later
    initPartitioner();

    // Import ImGui plugin.
    igl::opengl::glfw::imgui::ImGuiPlugin plugin;
    viewer.plugins.push_back(&plugin);
    igl::opengl::glfw::imgui::ImGuiMenu menu;
    plugin.widgets.push_back(&menu);

    // Viewer setup.
    viewerSetup(viewer);
    viewer.callback_key_pressed = &keyboardInputCallback; // Do NOT mistake it with viewer.callback_key_down
    viewer.callback_mouse_down = &mouseDownCallback;
    viewer.callback_mouse_move = &mouseMoveCallback;
    viewer.callback_mouse_up = &mouseUpCallback;
    viewer.callback_pre_draw = &preDrawCallback;

    menu.callback_draw_viewer_window = [&]() {
        //ImGui::SetNextWindowSizeConstraints(ImVec2(192, -1), ImVec2(-1, -1));
        ImGui::GetStyle().WindowMinSize = ImVec2(192, 1);
        menu.draw_viewer_window();
    };

    menu.callback_draw_viewer_menu = [&]() { // I had to copy a lot of ImGuiMenu::draw_viewer_menu since I must modify some of its behaviors.
        // Mesh
        if (ImGui::CollapsingHeader("Mesh", ImGuiTreeNodeFlags_DefaultOpen)) {
            float w = ImGui::GetContentRegionAvail().x;
            float p = ImGui::GetStyle().FramePadding.x;
            if (ImGui::Button("Load##Mesh", ImVec2((w - p) / 2.f, 0))) {
                viewer.open_dialog_load_mesh();
                if (viewer.data_list.size() > 1) {
                    model = Model(viewer.data().V, viewer.data().F, viewer.data().V_normals);
                    while (viewer.selected_data_index != 0)
                        viewer.erase_mesh(0);
                    viewerSetup(viewer);
                    initPartitioner();
                    runPartitioner(viewer.data());
                }
            }
            ImGui::SameLine(0, p);
            if (ImGui::Button("Save##Mesh", ImVec2((w - p) / 2.f, 0)))
                viewer.open_dialog_save_mesh();
        }

        // Viewing options
        if (ImGui::CollapsingHeader("Viewing Options", ImGuiTreeNodeFlags_DefaultOpen)) {
            if (ImGui::Button("Center object", ImVec2(-1, 0)))
                viewer.core().align_camera_center(viewer.data().V, viewer.data().F);
            if (ImGui::Button("Snap canonical view", ImVec2(-1, 0)))
                viewer.snap_to_canonical_quaternion();

            // Zoom
            ImGui::PushItemWidth(80 * menu.menu_scaling());
            ImGui::DragFloat("Zoom", &(viewer.core().camera_zoom), 0.05f, 0.1f, 20.0f);

            // Select rotation type
            int rotation_type = static_cast<int>(viewer.core().rotation_type);
            static Eigen::Quaternionf trackball_angle = Eigen::Quaternionf::Identity();
            static bool orthographic = true;
            if (ImGui::Combo("Camera Type", &rotation_type, "Trackball\0Two Axes\0002D Mode\0\0")) {
                using RT = igl::opengl::ViewerCore::RotationType;
                auto new_type = static_cast<RT>(rotation_type);
                if (new_type != viewer.core().rotation_type) {
                    if (new_type == RT::ROTATION_TYPE_NO_ROTATION) {
                        trackball_angle = viewer.core().trackball_angle;
                        orthographic = viewer.core().orthographic;
                        viewer.core().trackball_angle = Eigen::Quaternionf::Identity();
                        viewer.core().orthographic = true;
                    } else if (viewer.core().rotation_type == RT::ROTATION_TYPE_NO_ROTATION) {
                        viewer.core().trackball_angle = trackball_angle;
                        viewer.core().orthographic = orthographic;
                    }
                    viewer.core().set_rotation_type(new_type);
                }
            }

            // Orthographic view
            ImGui::Checkbox("Orthographic view", &(viewer.core().orthographic));
            ImGui::PopItemWidth();
        }

        // Helpers for making checkboxes quickly.
        auto makeCheckboxWithOptionId = [&](const char* label, unsigned int& option) {
            return ImGui::Checkbox(label,
                [&]() { return viewer.core().is_set(option); },
                [&](bool value) { return viewer.core().set(option, value); }
            );
        };
        // Draw options
        if (ImGui::CollapsingHeader("Draw Options", ImGuiTreeNodeFlags_DefaultOpen)) {
            makeCheckboxWithOptionId("Fill faces", viewer.data().show_faces);
            makeCheckboxWithOptionId("Wireframe", viewer.data().show_lines);
            ImGui::Checkbox("Double sided lighting", &viewer.data().double_sided);
            ImGui::ColorEdit4("Wireframe color", viewer.data().line_color.data(),
                ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_PickerHueWheel);
            ImGui::ColorEdit4("Background", viewer.core().background_color.data(),
                ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_PickerHueWheel);
            ImGui::PushItemWidth(ImGui::GetWindowWidth() * 0.3f);
            ImGui::DragFloat("Shininess", &(viewer.data().shininess), 0.05f, 0.0f, 100.0f);
            ImGui::PopItemWidth();
        }

        // Overlays
        if (ImGui::CollapsingHeader("Overlays", ImGuiTreeNodeFlags_DefaultOpen)) {
            makeCheckboxWithOptionId("Show overlays", viewer.data().show_overlay);
            makeCheckboxWithOptionId("Show overlays depth", viewer.data().show_overlay_depth);
        }

        // Graph
        if (ImGui::CollapsingHeader("Graph", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::PushItemWidth(ImGui::GetWindowWidth() * 0.3f);
            int dummySeedsCount = partitioner.getSeeds().size();
            if (ImGui::DragInt("Seeds count", &(dummySeedsCount), 1.0f, 1, partitioner.getNodesCount() / 2)) {
                partitioner.setSeedsCount(dummySeedsCount);
                runPartitioner(viewer.data());
            }
            ImGui::PopItemWidth();
            if (ImGui::Button("Generate Seeds", ImVec2(-1, 0))) {
                viewer.data().dirty = igl::opengl::MeshGL::DIRTY_ALL;
                generateSeeds(viewer.data());
            }
            if (ImGui::Button("Restore Seeds", ImVec2(-1, 0))) {
                viewer.data().dirty = igl::opengl::MeshGL::DIRTY_ALL;
                partitioner.restoreSeeds();
                runPartitioner(viewer.data());
            }
            if (ImGui::Button("Move Seeds Randomly", ImVec2(-1, 0))) {
                viewer.data().dirty = igl::opengl::MeshGL::DIRTY_ALL;
                moveSeedsRandomly(viewer.data());
            }
            int targetType = (int)graphType;
            ImGui::PushItemWidth(ImGui::GetWindowWidth() * 0.55f);
            if (ImGui::Combo("Graph type", &targetType, "Vertex\0Vertex Dense\0Triangle\0Triangle Dense\0Mixed\0\0"))
                setGraphType(viewer.data(), (GraphType)targetType);
            ImGui::PopItemWidth();
        }

        // Partitioning
        if (ImGui::CollapsingHeader("Partitioning", ImGuiTreeNodeFlags_DefaultOpen)) {
            if (ImGui::Button("Relax Seeds", ImVec2(-1, 0))) {
                viewer.data().dirty = igl::opengl::MeshGL::DIRTY_ALL;
                relaxPartitioner(viewer.data());
            }
            if (ImGui::Button("Relax Seeds Once", ImVec2(-1, 0))) {
                viewer.data().dirty = igl::opengl::MeshGL::DIRTY_ALL;
                relaxPartitionerOnce(viewer.data());
            }
            ImGui::Checkbox("Relax seeds over time", &viewer.core().is_animating);
            ImGui::Combo("Greedy relaxation", (int*)(&partitioner.greedyRelaxationType), "Disabled\0Enabled\0Extended\0\0");
            ImGui::Checkbox("Optimize precise relaxation", &partitioner.optimizePreciseRelaxation);
            ImGui::Checkbox("Optimize Dijkstra", &partitioner.optimizeDijkstra);
            ImGui::PushItemWidth(ImGui::GetWindowWidth() * 0.3f);
            ImGui::InputDouble("Relaxation rate", &(viewer.core().animation_max_fps));
            ImGui::PopItemWidth();
        }

        // Tests
        if (ImGui::CollapsingHeader("Tests", ImGuiTreeNodeFlags_DefaultOpen)) {
            if (ImGui::Button("Run Performance Test", ImVec2(-1, 0)))
                runPerformanceTest(viewer.data());
            if (ImGui::Checkbox("Ground truth", &showGroundTruth)) {
                viewer.data().dirty = igl::opengl::MeshGL::DIRTY_ALL;
                plotDataOnScreen(viewer.data());
            }
            if (ImGui::SliderFloat("Triangulation preview", &triangulationFactor, 0.0f, 1.0f)) {
                if (triangulationFactor <= 0) {
                    viewer.data().set_vertices(model.getVerticesMatrix());
                    viewer.data().compute_normals();
                    viewer.data().show_lines = false;
                } else
                    viewer.data().show_lines = true;
                viewer.data().dirty = igl::opengl::MeshGL::DIRTY_ALL;
                plotDataOnScreen(viewer.data());
            }
            if (ImGui::ColorEdit3("Triangulation color", glm::value_ptr(triangulationColor),
                ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_PickerHueWheel)) {
                viewer.data().dirty = igl::opengl::MeshGL::DIRTY_ALL;
                plotDataOnScreen(viewer.data());
            }
        }
    };

    // Run everything...
    runPartitioner(viewer.data());
    viewer.launch();

    return 0;
}
