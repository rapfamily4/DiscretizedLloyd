#ifndef APPLICATION_H
#define APPLICATION_H

#include <vector>
#include <map>
#include <string>
#include <algorithm>
#include <cstddef>
#include <fstream>
#include <sstream>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <igl/file_dialog_open.h>
#include <igl/opengl/glfw/Viewer.h>
#include <igl/opengl/glfw/imgui/ImGuiPlugin.h>
#include <igl/opengl/glfw/imgui/ImGuiMenu.h>
#include <igl/opengl/glfw/imgui/ImGuiHelpers.h>
#include <igl/unproject_onto_mesh.h>
#include <glm/glm.hpp>
#include <glm/gtx/color_space.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <imgui_internal.h>
#include "dijkstra_partitioner.hpp"
#include "model.hpp"
#include "tangent_space.hpp"
#include "stopwatch.hpp"
#include "performance_stats.hpp"
#include "consts.hpp"

class Application {
private:
    enum GraphType {
        VERTEX,
        VERTEX_DENSE,
        TRIANGLE,
        TRIANGLE_DENSE,
        MIXED,
        GRAPH_TYPE_SIZE,
    };

    const unsigned int DEFAULT_REGIONS_NUMBER = 16;

    igl::opengl::glfw::Viewer viewer;
    igl::opengl::glfw::imgui::ImGuiPlugin guiPlugin;
    igl::opengl::glfw::imgui::ImGuiMenu guiMenu;
    DijkstraPartitioner partitioner;
    Model model;
    Stopwatch swatch;
    GraphType graphType = GraphType::VERTEX;
    float tangentFieldsScale = 0.25f;
    float triangulationFactor = 0;
    glm::vec3 triangulationColor{ 0.7f, 0.7f, 0.7f };
    bool showAdvancedOptions = false;
    bool showTreesOverlay = true;
    bool showTangentFieldsOverlay = true;
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

    void printPerformanceResults(double runningTime) {
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
        }
        else std::cout << "Greedy:\t\tDISABLED\n";
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
        }
        else { // Ground truth for mixed graph is vertex based.
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

    void triangulationPreview(const std::vector<int>& regionAssignments) {
        igl::opengl::ViewerData& data = viewer.data();
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
            }
            else if (isMixed(graphType) && node >= V.rows()) {
                for (int v = 0; v < F.cols(); v++) {
                    mixForDelaunay(F(node - V.rows(), v), region);
                    data.V.row(F(node - V.rows(), v)) << mixed.x, mixed.y, mixed.z;
                }
            }
            else {
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

    void assignColorsToModel(const std::vector<int>& regionAssignments) {
        Eigen::MatrixXd C(isMixed(graphType) ? model.getVerticesMatrix().rows() : regionAssignments.size(), 3);
        std::vector<glm::vec3> regionCols;
        regionColors(partitioner.getSeeds().size(), regionCols);

        for (int node = 0; node < C.rows(); node++) {
            glm::vec3 nodeColor = glm::mix(regionCols[regionAssignments[node]], triangulationColor, triangulationFactor);
            C.row(node) << nodeColor.x, nodeColor.y, nodeColor.z;
        }
        viewer.data().set_colors(C);
    }

    void plotTrees() {
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
        viewer.data().add_points(S, Eigen::RowVector3d(1.0, 1.0, 1.0));

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
        viewer.data().add_edges(P0, P1, Eigen::RowVector3d(1.0, 1.0, 1.0));
    }

    void plotTangentFields() {
        igl::opengl::ViewerData& data = viewer.data();
        const std::vector<TangentSpace>& vfield = model.getTangentField();
        std::vector<glm::vec3> positions;
        model.verticesPositions(positions);
        
        assert(vfield.size() == positions.size());
        Eigen::MatrixXd P(vfield.size(), 3);
        Eigen::MatrixXd T(vfield.size(), 3);
        Eigen::MatrixXd B(vfield.size(), 3);
        for (int i = 0; i < vfield.size(); i++) {
            glm::vec3 t, b;
            glm::vec3& p = positions[i];
            vfield[i].tangent(t);
            vfield[i].bitangent(b);
            t = p + t * tangentFieldsScale;
            b = p + b * tangentFieldsScale;
            P.row(i) << p.x, p.y, p.z;
            T.row(i) << t.x, t.y, t.z;
            B.row(i) << b.x, b.y, b.z;
        }
        data.add_edges(P, T, Eigen::RowVector3d{ 1, 0, 0 });
        data.add_edges(P, B, Eigen::RowVector3d{ 0, 1, 0 });
    }

    void plotOverlays() {
        igl::opengl::ViewerData& data = viewer.data();
        data.clear_points();
        data.clear_edges();

        std::vector<int> regionAssignments;
        if (showGroundTruth) {
            Eigen::MatrixXd barycenters = Eigen::MatrixXd::Zero(partitioner.getSeeds().size(), 3);
            groundTruth(regionAssignments, barycenters);
            data.add_points(barycenters, Eigen::RowVector3d(0, 0, 0));
        }
        else
            partitioner.nodewiseRegionAssignments(regionAssignments);
        if (triangulationFactor > 0)
            triangulationPreview(regionAssignments);
        if (!showGroundTruth && triangulationFactor <= 0) {
            if (showTreesOverlay)
                plotTrees();
            if (showTangentFieldsOverlay && model.hasTangentField())
                plotTangentFields();
        }
        assignColorsToModel(regionAssignments);
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

    void runPartitioner() {
        partitioner.resetState();
        partitioner.partitionNodes();
        plotOverlays();
    }

    void generateSeeds() {
        partitioner.generateRandomSeeds();
        runPartitioner();
    }

    void moveSeedsRandomly() {
        partitioner.moveSeedsRandomly();
        partitioner.partitionNodes();
        plotOverlays();
    }

    void relaxPartitioner(bool plotData = true) {
        swatch.begin();
        partitioner.resetState();
        partitioner.relaxSeeds();
        printPerformanceResults(swatch.end());
        if (plotData) plotOverlays();
    }

    void relaxPartitionerOnce() {
        partitioner.relaxSeedsOnce();
        partitioner.partitionNodes();
        plotOverlays();
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
        }
        else if (isVertexBased(targetType)) {
            model.vertexSurroundingAreas(weights);
            if (isDense(targetType)) model.denseVertexAdjacency(adjacency, costs);
            else model.vertexAdjacency(adjacency, costs);
        }
        else {
            model.vertexAndTriangleAreas(weights);
            model.mixedAdjacency(adjacency, costs);
        }
    }

    void setGraphType(GraphType targetType) {
        std::vector<int> newSeeds;
        if (targetType != graphType && convertSeeds(targetType, newSeeds))
            partitioner.setSeeds(newSeeds);

        std::vector<float> weights;
        std::vector<std::vector<int>> adjacency;
        std::vector<std::vector<float>> costs;
        setGraphMetrics(targetType, weights, adjacency, costs);

        graphType = targetType;
        viewer.data().set_face_based(isTriangleBased(targetType));
        partitioner.generateNodes(weights, adjacency, costs);
        runPartitioner();
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
            setGraphType((GraphType)type);
            for (int greedy = 0; greedy < 3; greedy++)
                for (int precise = 0; precise < 2; precise++)
                    for (int dijkstra = 0; dijkstra < 2; dijkstra++) {
                        partitioner.greedyRelaxationType = (DijkstraPartitioner::GreedyOption)greedy;
                        partitioner.optimizePreciseRelaxation = (bool)precise;
                        partitioner.optimizeDijkstra = (bool)dijkstra;
                        relaxPartitioner(false);
                        partitioner.restoreSeeds();
                    }
        }
        plotOverlays();

        // Restore previous parameters.
        setGraphType(typeAtStart);
        partitioner.greedyRelaxationType = greedyAtStart;
        partitioner.optimizePreciseRelaxation = preciseAtStart;
        partitioner.optimizeDijkstra = dijkstraAtStart;
    }

    void viewerSetup(igl::opengl::glfw::Viewer& viewer) {
        viewer.data().point_size = 8.;
        if (triangulationFactor <= 0.0f)
            viewer.data().show_lines = false;
        viewer.core().is_animating = false;
        viewer.core().animation_max_fps = 120.;
        viewer.core().lighting_factor = 1;
    }

    bool keyboardInputCallback(unsigned char key, int modifier) {
        switch (key) {
        case 'F':
        case 'f':
            setGraphType(viewer.data().face_based ? GraphType::VERTEX : GraphType::TRIANGLE);
            return true;

        case 'G':
        case 'g':
            generateSeeds();
            return true;

        case 'N':
        case 'n':
            moveSeedsRandomly();
            return true;

        case 'R':
        case 'r':
            relaxPartitionerOnce();
            return true;
            break;

        case 'V':
        case 'v':
            showGroundTruth = !showGroundTruth;
            plotOverlays();
            return true;
        }
        return false;
    }

    bool mouseDownCallback(int button, int modifier) {
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
                    runPartitioner();
                }
                return true;
            case IGL_MOD_CONTROL | IGL_MOD_SHIFT:
                if (isTriangleBased(graphType) && partitioner.removeSeedOfNode(pickedTriangle) ||
                    (isVertexBased(graphType) || isMixed(graphType)) && partitioner.removeSeedOfNode(pickedVertex) ||
                    isMixed(graphType) && partitioner.removeSeedOfNode(pickedTriangleMixed)) {
                    runPartitioner();
                }
                return true;
            case IGL_MOD_ALT:
                if (isTriangleBased(graphType) && partitioner.moveSeedToNode(pickedTriangle) ||
                    (isVertexBased(graphType) || isMixed(graphType)) && partitioner.moveSeedToNode(pickedVertex) ||
                    isMixed(graphType) && partitioner.moveSeedToNode(pickedTriangleMixed)) {
                    isDraggingRegion = true;
                    runPartitioner();
                }
                return true;
            }
        }
        return false;
    }

    bool mouseMoveCallback(int x, int y) {
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
                runPartitioner();
            return true;
        }
        else isDraggingRegion = false;
        return false;
    }

    bool mouseUpCallback(int button, int modifier) {
        isDraggingRegion = false;
        return false;
    }

    bool preDrawCallback() {
        if (viewer.core().is_animating)
            relaxPartitionerOnce();
        return false;
    }

    void drawViewerWindowCallback() {
        //ImGui::SetNextWindowSizeConstraints(ImVec2(192, -1), ImVec2(-1, -1));
        ImGui::GetStyle().WindowMinSize = ImVec2(192, 1);
        guiMenu.draw_viewer_window();
    }

    void drawViewerMenuCallback() {
        ImGui::Checkbox("Advanced options", &showAdvancedOptions);
        // Mesh
        if (ImGui::CollapsingHeader("Mesh", ImGuiTreeNodeFlags_DefaultOpen)) {
            float w = ImGui::GetContentRegionAvail().x;
            float p = ImGui::GetStyle().FramePadding.x;
            if (ImGui::Button("Load Mesh", ImVec2((w - p) / 2.f, 0))) {
                viewer.open_dialog_load_mesh();
                if (viewer.data_list.size() > 1) {
                    model = Model(viewer.data().V, viewer.data().F);
                    while (viewer.selected_data_index != 0)
                        viewer.erase_mesh(0);
                    viewerSetup(viewer);
                    initPartitioner();
                    runPartitioner();
                }
            }
            ImGui::SameLine(0, p);
            if (ImGui::Button("Save Mesh", ImVec2((w - p) / 2.f, 0)))
                viewer.open_dialog_save_mesh();
            if (ImGui::Button("Load .vfield", ImVec2(w - p, 0))) {
                if (readTangentSpaceFromFile(igl::file_dialog_open())) {
                    viewerSetup(viewer);
                    initPartitioner();
                    runPartitioner();
                }
            }
        }

        // Viewing options
        if (showAdvancedOptions && ImGui::CollapsingHeader("Viewing Options", ImGuiTreeNodeFlags_DefaultOpen)) {
            if (ImGui::Button("Center object", ImVec2(-1, 0)))
                viewer.core().align_camera_center(viewer.data().V, viewer.data().F);
            if (ImGui::Button("Snap canonical view", ImVec2(-1, 0)))
                viewer.snap_to_canonical_quaternion();

            // Zoom
            ImGui::PushItemWidth(80 * guiMenu.menu_scaling());
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
                    }
                    else if (viewer.core().rotation_type == RT::ROTATION_TYPE_NO_ROTATION) {
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
        if (showAdvancedOptions && ImGui::CollapsingHeader("Draw Options", ImGuiTreeNodeFlags_DefaultOpen)) {
            makeCheckboxWithOptionId("Wireframe", viewer.data().show_lines);
            makeCheckboxWithOptionId("Fill faces", viewer.data().show_faces);
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
            if (ImGui::Checkbox("Show shortest paths", &showTreesOverlay)) {
                plotOverlays();
                viewer.data().dirty = igl::opengl::MeshGL::DIRTY_ALL;
            }
            if (!model.hasTangentField()) {
                ImGui::PushItemFlag(ImGuiItemFlags_Disabled, true);
                ImGui::PushStyleVar(ImGuiStyleVar_Alpha, ImGui::GetStyle().Alpha * 0.5f);
            }
            if (ImGui::Checkbox("Show tangent fields", &showTangentFieldsOverlay)) {
                plotOverlays();
                viewer.data().dirty = igl::opengl::MeshGL::DIRTY_ALL;
            }
            if (ImGui::DragFloat("Tangent fields' scale", &tangentFieldsScale, 0.05f)) {
                plotOverlays();
                viewer.data().dirty = igl::opengl::MeshGL::DIRTY_ALL;
            }
            if (!model.hasTangentField()) {
                ImGui::PopItemFlag();
                ImGui::PopStyleVar();
            }
            if (showAdvancedOptions)
                makeCheckboxWithOptionId("Show overlays depth", viewer.data().show_overlay_depth);
        }

        // Graph
        if (ImGui::CollapsingHeader("Graph", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::PushItemWidth(ImGui::GetWindowWidth() * 0.3f);
            int dummySeedsCount = partitioner.getSeeds().size();
            if (ImGui::DragInt("Seeds count", &(dummySeedsCount), 1.0f, 1, partitioner.getNodesCount() / 2)) {
                partitioner.setSeedsCount(dummySeedsCount);
                runPartitioner();
            }
            ImGui::PopItemWidth();
            if (ImGui::Button("Generate Seeds", ImVec2(-1, 0))) {
                viewer.data().dirty = igl::opengl::MeshGL::DIRTY_ALL;
                generateSeeds();
            }
            if (ImGui::Button("Restore Seeds", ImVec2(-1, 0))) {
                viewer.data().dirty = igl::opengl::MeshGL::DIRTY_ALL;
                partitioner.restoreSeeds();
                runPartitioner();
            }
            if (ImGui::Button("Move Seeds Randomly", ImVec2(-1, 0))) {
                viewer.data().dirty = igl::opengl::MeshGL::DIRTY_ALL;
                moveSeedsRandomly();
            }
            ImGui::PushItemWidth(ImGui::GetWindowWidth() * 0.55f);
            int targetType = (int)graphType;
            if (ImGui::Combo("Graph type", &targetType, "Vertex\0Vertex Dense\0Triangle\0Triangle Dense\0Mixed\0\0"))
                setGraphType((GraphType)targetType);
            if (showAdvancedOptions) {
                int distFunc = (int)model.distanceFunction;
                if (ImGui::Combo("Distance function", &distFunc, "Geodesic\0Manhattan\0Infinite\0\0")) {
                    model.distanceFunction = (Model::DistanceFunction)distFunc;
                    setGraphType(graphType);
                }
            }
            ImGui::PopItemWidth();
        }

        // Partitioning
        if (ImGui::CollapsingHeader("Partitioning", ImGuiTreeNodeFlags_DefaultOpen)) {
            if (ImGui::Button("Relax Seeds", ImVec2(-1, 0))) {
                viewer.data().dirty = igl::opengl::MeshGL::DIRTY_ALL;
                relaxPartitioner();
            }
            if (ImGui::Button("Relax Seeds Once", ImVec2(-1, 0))) {
                viewer.data().dirty = igl::opengl::MeshGL::DIRTY_ALL;
                relaxPartitionerOnce();
            }
            ImGui::Checkbox("Relax seeds over time", &viewer.core().is_animating);
            ImGui::PushItemWidth(ImGui::GetWindowWidth() * 0.3f);
            ImGui::InputDouble("Relaxation rate", &(viewer.core().animation_max_fps));
            ImGui::PopItemWidth();
            if (showAdvancedOptions) {
                ImGui::Combo("Greedy relaxation", (int*)(&partitioner.greedyRelaxationType), "Disabled\0Enabled\0Extended\0\0");
                ImGui::Checkbox("Optimize precise relaxation", &partitioner.optimizePreciseRelaxation);
                ImGui::Checkbox("Optimize Dijkstra", &partitioner.optimizeDijkstra);
            }
        }

        // Tests
        if (ImGui::CollapsingHeader("Tests", ImGuiTreeNodeFlags_DefaultOpen)) {
            if (ImGui::Button("Run Performance Test", ImVec2(-1, 0)))
                runPerformanceTest(viewer.data());
            if (ImGui::Checkbox("Ground truth", &showGroundTruth)) {
                viewer.data().dirty = igl::opengl::MeshGL::DIRTY_ALL;
                plotOverlays();
            }
            if (ImGui::SliderFloat("Delaunay preview", &triangulationFactor, 0.0f, 1.0f)) {
                if (triangulationFactor <= 0) {
                    viewer.data().set_vertices(model.getVerticesMatrix());
                    viewer.data().compute_normals();
                    viewer.data().show_lines = false;
                }
                else
                    viewer.data().show_lines = true;
                viewer.data().dirty = igl::opengl::MeshGL::DIRTY_ALL;
                plotOverlays();
            }
            if (showAdvancedOptions && ImGui::ColorEdit3("Delaunay color", glm::value_ptr(triangulationColor),
                ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_PickerHueWheel)) {
                viewer.data().dirty = igl::opengl::MeshGL::DIRTY_ALL;
                plotOverlays();
            }
        }
    }

    bool isNumberList(std::string line) {
        return std::all_of(line.begin(), line.end(), [](char ch) {
            return std::isdigit(ch) || std::isspace(ch) || ch == '-' || ch == '+' || ch == '.' || ch == 'e';
        });
    }

    bool parseFloats(std::stringstream& stream, std::vector<float>& floats, int amount) {
        floats.clear();
        floats.resize(amount);
        for (int i = 0; i < amount; i++)
            stream >> floats.at(i);
        
        return stream.good() || stream.eof();
    }

    bool readTangentSpaceFromFile(std::string vFieldPath) {
        if (vFieldPath.length() == 0)
            return false;
        
        std::size_t dotIndex = vFieldPath.find_last_of('.');
        if (dotIndex != std::string::npos) {
            std::string extention = vFieldPath.substr(dotIndex + 1);
            if (extention == "vfield" || extention == "VFIELD") {
                std::ifstream ifile{ vFieldPath };
                std::string line;
                bool vFieldFound = false;
                while (std::getline(ifile, line)) {
                    if (line == "k1\t k2\t k1v_x\t k1v_y\t k1v_z\t k2v_x\t k2v_y\t k2v_z") {
                        vFieldFound = true;
                        break;
                    }
                }
                if (vFieldFound) {
                    std::stringstream sstream;
                    const Eigen::MatrixXd& V = model.getVerticesMatrix();
                    std::vector<float> parsedNums;
                    int scannedLines = 0;
                    while (std::getline(ifile, line) && isNumberList(line)) {
                        sstream << line << " ";
                        scannedLines++;
                    }
                    if (scannedLines == V.rows()) {
                        std::vector<TangentSpace> vfield;
                        for (int i = 0; i < scannedLines; i++) {
                            if (parseFloats(sstream, parsedNums, 8)) {
                                glm::vec3 tDir{ parsedNums[2], parsedNums[3], parsedNums[4] };
                                glm::vec3 bDir{ parsedNums[5], parsedNums[6], parsedNums[7] };
                                float tMag = parsedNums[0];
                                float bMag = parsedNums[1];
                                vfield.emplace_back(tMag, bMag, tDir, bDir);
                            }
                            else {
                                std::cout << "Parse error in " << vFieldPath << " at tangent space " << i << ".\n";
                                return false;
                            }
                        }
                        model.setTangentField(vfield);
                        return true;
                    } else
                        std::cout << "Invalid number of tangent spaces in " << vFieldPath << ": found " << scannedLines << ", required " << V.rows() << ".\n";
                } else
                    std::cout << "Tangent spaces not found in " << vFieldPath << "\n";
            } else
                std::cout << "File " << vFieldPath << " has invalid extension: expected .vfield, .VFIELD\n";
        } else
            std::cout << "File " << vFieldPath << " has no extension: expected .vfield, .VFIELD\n";
        
        return false;
    }

public:
    Application() {}

    void printUsage() {
        std::cout << "\nDijkstraPartitioner usage:\n"
            << "  G,g                          Generate random seeds\n"
            << "  N,n                          Move seeds randomly\n"
            << "  R,r                          Relax seeds once\n"
            << "  V,v                          Toggle ground truth\n"
            << "  CTRL + Left Click            Add clicked node to seeds\n"
            << "  SHIFT + CTRL + Left Click    Remove clicked region\n"
            << "  ALT + Left Click             Drag region\n\n";
    }

    bool init(std::string modelPath = "./models/plane.obj") {
        // Load mesh from file.
        if (!viewer.load_mesh_from_file(modelPath)) return false;
        model = Model(viewer.data().V, viewer.data().F);

        // Dijkstra partitioner.
        //model.fillTangentSpaceFlat(glm::vec3(2, 0, 0), glm::vec3(0, 1, 0)); // TODO: erase this line later
        initPartitioner();

        // Import ImGui plugin.
        viewer.plugins.push_back(&guiPlugin);
        guiPlugin.widgets.push_back(&guiMenu);

        // Viewer setup.
        viewerSetup(viewer);
        viewer.callback_key_pressed = [this](igl::opengl::glfw::Viewer& viewer, unsigned int key, int modifier) { return keyboardInputCallback(key, modifier); };
        viewer.callback_mouse_down = [this](igl::opengl::glfw::Viewer& viewer, int button, int modifier) { return mouseDownCallback(button, modifier); };
        viewer.callback_mouse_move = [this](igl::opengl::glfw::Viewer& viewer, int x, int y) { return mouseMoveCallback(x, y); };
        viewer.callback_mouse_up = [this](igl::opengl::glfw::Viewer& viewer, int button, int modifier) { return mouseUpCallback(button, modifier); };
        viewer.callback_pre_draw = [this](igl::opengl::glfw::Viewer& viewer) { return preDrawCallback(); };
        guiMenu.callback_draw_viewer_window = [this]() { drawViewerWindowCallback(); };
        guiMenu.callback_draw_viewer_menu = [this]() { drawViewerMenuCallback(); };

        return true;
    }

    void launch() {
        runPartitioner();
        viewer.launch();
    }
};

#endif // !APPLICATION_H
