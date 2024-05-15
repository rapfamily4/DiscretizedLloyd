#ifndef MODEL_H
#define MODEL_H

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <igl/readOBJ.h>
#include <igl/readPLY.h>
#include <igl/adjacency_list.h>
#include <glm/glm.hpp>
#include <glm/gtx/vector_angle.hpp>
#include <glm/gtx/transform.hpp>
#include <tangent_space.hpp>

class Model {
private:
    struct HalfEdge {
        int i;
        int j;

        HalfEdge() : i{ -1 }, j{ -1 } {}
        HalfEdge(int i, int j) : i{ i }, j{ j } { }

        HalfEdge flip() const { return HalfEdge(j, i); }
        bool operator<(const HalfEdge& other) const {
            if (i < other.i) return true;
            if (i > other.i) return false;
            return j < other.j;
        }
    };

    Eigen::MatrixXd V;
    Eigen::MatrixXi F;
    std::vector<TangentSpace> vfield;


    void vertexTriangleAdjacency(std::vector<std::vector<int>>& VT) {
        VT.clear();
        VT.resize(V.rows());
        for (int t = 0; t < F.rows(); t++) // t-th triangle.
            for (int v = 0; v < F.cols(); v++) // v-th vertex of the triangle.
                VT[F(t, v)].push_back(t);
    }

    void tangentFieldFromTriangle(int t, TangentSpace& field) {
        TangentSpace ijAvg;
        TangentSpace::average(vfield[F(t, 0)], vfield[F(t, 1)], ijAvg);
        TangentSpace::average(ijAvg, vfield[F(t, 2)], field);
    }

    float distanceGivenFuctionType(const glm::vec3& delta) {
        switch (distanceFunction) {
            case GEODESIC_D:
                return glm::length(delta);
            case MANHATTAN_D:
                return std::abs(delta.x) + std::abs(delta.y) + std::abs(delta.z);
            case INFINITE_D:
                return std::max(std::max(std::abs(delta.x), std::abs(delta.y)), std::abs(delta.z));
        }
        return -1;
    }

    float weightedDistanceGivenFunctionType(const TangentSpace& space, const glm::vec3& delta) {
        switch (distanceFunction) {
            case GEODESIC_D:
                return space.euclideanDistance(delta);
            case MANHATTAN_D:
                return space.manhattanDistance(delta);
            case INFINITE_D:
                return space.infiniteDistance(delta);
        }
        return -1;
    }

    float oppositePointsDistance(glm::vec3 vertexDelta, HalfEdge trianglesBase) {
        glm::vec3 base = glm::vec3(
            V(trianglesBase.i, 0) - V(trianglesBase.j, 0),
            V(trianglesBase.i, 1) - V(trianglesBase.j, 1),
            V(trianglesBase.i, 2) - V(trianglesBase.j, 2)
        );
        float sqrDoubleArea = glm::length2(glm::cross(base, vertexDelta));
        float sqrDot = glm::dot(vertexDelta, base);
        sqrDot *= sqrDot;
        float sqrBaseLength = glm::length2(base);
        return std::sqrt((sqrDoubleArea + sqrDot) / sqrBaseLength);
    }

    float verticesDistance(int i, int j) {
        assert(i < V.rows() && j < V.rows());
        
        glm::vec3 vertex = glm::vec3(V(i, 0), V(i, 1), V(i, 2));
        glm::vec3 neighbor = glm::vec3(V(j, 0), V(j, 1), V(j, 2));
        glm::vec3 delta = neighbor - vertex;
        if (hasTangentField()) {
            TangentSpace avgSpace;
            TangentSpace::average(vfield[i], vfield[j], avgSpace);
            return weightedDistanceGivenFunctionType(avgSpace, delta);
        } else return distanceGivenFuctionType(delta);
    }

    float oppositeVerticesDistance(int i, int j, HalfEdge trianglesBase) {
        glm::vec3 delta = glm::vec3(
            V(i, 0) - V(j, 0),
            V(i, 1) - V(j, 1),
            V(i, 2) - V(j, 2)
        );
        float dist = oppositePointsDistance(delta, trianglesBase);
        delta = glm::normalize(-delta) * dist;
        if (hasTangentField()) {
            TangentSpace avgSpace;
            TangentSpace::average(vfield[i], vfield[j], avgSpace);
            return weightedDistanceGivenFunctionType(avgSpace, delta);
        } else return distanceGivenFuctionType(delta);
    }

    float trianglesDistance(int i, int j, HalfEdge trianglesBase) {
        glm::vec3 iBarycenter, jBarycenter;
        triangleBarycenter(i, iBarycenter);
        triangleBarycenter(j, jBarycenter);
        float dist = oppositePointsDistance(iBarycenter - jBarycenter, trianglesBase);
        glm::vec3 delta = glm::normalize(jBarycenter - iBarycenter) * dist;
        if (hasTangentField()) {
            TangentSpace iAvgSpace, jAvgSpace, avgSpace;
            tangentFieldFromTriangle(i, iAvgSpace);
            tangentFieldFromTriangle(j, jAvgSpace);
            TangentSpace::average(iAvgSpace, jAvgSpace, avgSpace);
            return weightedDistanceGivenFunctionType(avgSpace, delta);
        } else return distanceGivenFuctionType(delta);
    }

    float vertexTriangleDistance(int v, int t) {
        glm::vec3 vPoint, tBarycenter;
        vPoint = glm::vec3(V(v, 0), V(v, 1), V(v, 2));
        triangleBarycenter(t, tBarycenter);
        glm::vec3 delta = tBarycenter - vPoint;
        if (hasTangentField()) {
            TangentSpace tAvgSpace, avgSpace;
            tangentFieldFromTriangle(t, tAvgSpace);
            TangentSpace::average(vfield[v], tAvgSpace, avgSpace);
            return weightedDistanceGivenFunctionType(avgSpace, delta);
        } else return distanceGivenFuctionType(delta);
    }

public:
    enum DistanceFunction {
        GEODESIC_D,
        MANHATTAN_D,
        INFINITE_D,
    };

    DistanceFunction distanceFunction = DistanceFunction::GEODESIC_D;


    Model() {}
    Model(Eigen::MatrixXd vertices, Eigen::MatrixXi triangles) :
        V{ vertices },
        F{ triangles } {
    }

    void vertexSurroundingAreas(std::vector<float>& vertexAreas) {
        vertexAreas.clear();
        vertexAreas.resize(V.rows(), 0.0f);
        for (int t = 0; t < F.rows(); t++) {
            float thirdOfArea = triangleArea(t) / 3;
            vertexAreas[F(t, 0)] += thirdOfArea;
            vertexAreas[F(t, 1)] += thirdOfArea;
            vertexAreas[F(t, 2)] += thirdOfArea;
        }
    }

    void vertexAdjacency(std::vector<std::vector<int>>& adjacency, std::vector<std::vector<float>>& distances) {
        adjacency.clear();
        igl::adjacency_list(F, adjacency);
        assert(adjacency.size() == V.rows());
        distances.clear();
        for (int i = 0; i < adjacency.size(); i++) {
            distances.push_back(std::vector<float>());
            for (int n : adjacency[i]) {
                distances[i].push_back(verticesDistance(i, n));
            }
        }
    }

    void denseVertexAdjacency(std::vector<std::vector<int>>& adjacency, std::vector<std::vector<float>>& distances) {
        vertexAdjacency(adjacency, distances);
        std::map<HalfEdge, int> edges;
        for (int t = 0; t < F.rows(); t++) { // t-th triangle.
            for (int v = 0; v < F.cols(); v++) { // v-th vertex of the triangle.
                int i = F(t, v), j = F(t, (v + 1) % F.cols());
                int oppositeVert0 = F(t, (v + 2) % F.cols());
                HalfEdge edge(i, j);
                HalfEdge flipped = edge.flip();

                if (edges.find(flipped) != edges.end()) {
                    int n = edges[flipped];
                    assert(t != n);
                    int oppositeVert1;
                    for (int nv : F.row(n))
                        if (nv != i && nv != j) {
                            oppositeVert1 = nv;
                            break;
                        }
                    adjacency[oppositeVert0].push_back(oppositeVert1);
                    adjacency[oppositeVert1].push_back(oppositeVert0);
                    float w = oppositeVerticesDistance(oppositeVert0, oppositeVert1, edge);
                    distances[oppositeVert0].push_back(w);
                    distances[oppositeVert1].push_back(w);
                }
                else
                    edges[edge] = t;
            }
        }
    }

    float triangleArea(int id) {
        assert(id < F.rows());
        int i = F(id, 0), j = F(id, 1), k = F(id, 2);
        glm::vec3 vi = glm::vec3(V(i, 0), V(i, 1), V(i, 2));
        glm::vec3 vj = glm::vec3(V(j, 0), V(j, 1), V(j, 2));
        glm::vec3 vk = glm::vec3(V(k, 0), V(k, 1), V(k, 2));
        return 0.5f * glm::length(glm::cross((vj - vi), (vk - vi)));
    }

    void triangleAreas(std::vector<float>& areas) {
        areas.clear();
        areas.resize(F.rows());
        for (int t = 0; t < F.rows(); t++) {
            areas[t] = triangleArea(t);
        }
    }

    void triangleAdjacency(std::vector<std::vector<int>>& adjacency, std::vector<std::vector<float>>& distances) {
        adjacency.clear();
        adjacency.resize(F.rows());
        distances.clear();
        distances.resize(F.rows());
        std::map<HalfEdge, int> edges;

        for (int t = 0; t < F.rows(); t++) { // t-th triangle.
            for (int v = 0; v < F.cols(); v++) { // v-th vertex of the triangle.
                int i = F(t, v), j = F(t, (v + 1) % F.cols());
                HalfEdge edge(i, j);
                HalfEdge flipped = edge.flip();
                if (edges.find(flipped) != edges.end()) {
                    int n = edges[flipped];
                    assert(t != n);
                    adjacency[t].push_back(n);
                    adjacency[n].push_back(t);
                    float dist = trianglesDistance(t, n, edge);
                    distances[t].push_back(dist);
                    distances[n].push_back(dist);
                }
                else edges[edge] = t;
            }
        }
    }

    void denseTriangleAdjacency(std::vector<std::vector<int>>& adjacency, std::vector<std::vector<float>>& distances) {
        triangleAdjacency(adjacency, distances);
        std::vector<std::vector<int>> VT;
        vertexTriangleAdjacency(VT);
        for (int t = 0; t < F.rows(); t++) { // t-th triangle.
            for (int v = 0; v < F.cols(); v++) { // v-th vertex of the triangle.
                int vId = F(t, v);
                for (int n : VT[vId]) // Triangle n, neighbor of vertex vId.
                    if (n != t && std::find(adjacency[t].begin(), adjacency[t].end(), n) == adjacency[t].end()) {
                        adjacency[t].push_back(n);
                        float dist = vertexTriangleDistance(vId, t) + vertexTriangleDistance(vId, n);
                        distances[t].push_back(dist);
                    }
            }
        }
    }

    void vertexAndTriangleAreas(std::vector<float>& areas) {
        vertexSurroundingAreas(areas);
        std::vector<float> trAreas;
        triangleAreas(trAreas);
        areas.insert(areas.end(), trAreas.begin(), trAreas.end());
        assert(areas.size() == F.rows() + V.rows());
    }

    void mixedAdjacency(std::vector<std::vector<int>>& adjacency, std::vector<std::vector<float>>& distances) {
        vertexAdjacency(adjacency, distances);
        std::vector<std::vector<int>> trAdjacency;
        std::vector<std::vector<float>> trDistances;
        triangleAdjacency(trAdjacency, trDistances);
        adjacency.insert(adjacency.end(), trAdjacency.begin(), trAdjacency.end());
        distances.insert(distances.end(), trDistances.begin(), trDistances.end());

        for (int t = 0; t < F.rows(); t++)
            for (int& adj : adjacency[t + V.rows()])
                adj += V.rows();

        for (int t = 0; t < F.rows(); t++) {
            for (int v = 0; v < 3; v++) {
                int vId = F(t, v);
                adjacency[t + V.rows()].push_back(vId);
                adjacency[vId].push_back(t + V.rows());
                float dist = vertexTriangleDistance(vId, t);
                distances[t + V.rows()].push_back(dist);
                distances[vId].push_back(dist);
            }
        }
        assert(adjacency.size() == F.rows() + V.rows() && distances.size() == F.rows() + V.rows());
    }

    int firstVertexFromTriangle(int triangleId) {
        // Is this even useful?
        assert(triangleId < F.rows());
        return F(triangleId, 0);
    }

    void firstVerticesFromTriangles(const std::vector<int>& triangleIds, std::vector<int>& vertexIds) {
        // NB: there may be duplicate vertices in vertexIds
        vertexIds.clear();
        for (int t : triangleIds)
            vertexIds.push_back(firstVertexFromTriangle(t));
        assert(vertexIds.size() == triangleIds.size());
    }

    int triangleFromVertex(int vertexId) {
        for (int i = 0; i < F.size(); i++)
            if (F(i) == vertexId)
                return i % F.rows(); // F is column major.
        return -1;
    }

    void trianglesFromVertices(const std::vector<int>& vertexIds, std::vector<int>& triangleIds) {
        triangleIds.clear();
        for (int v : vertexIds)
            triangleIds.push_back(triangleFromVertex(v));
        assert(vertexIds.size() == triangleIds.size());
    }

    int vertexFromBarycentricCoords(int triangleId, Eigen::Vector3f& barycentricCoords) {
        float maxCoord = -INF;
        int result = F(triangleId, 0);
        for (int c = 0; c < barycentricCoords.size(); c++) {
            if (barycentricCoords(c) > maxCoord) {
                maxCoord = barycentricCoords(c);
                result = F(triangleId, c);
            }
        }
        return result;
    }

    void vertexPosition(int id, glm::vec3& position) {
        position.x = V(id, 0);
        position.y = V(id, 1);
        position.z = V(id, 2);
    }

    void verticesPositions(std::vector<glm::vec3>& positions) {
        positions.clear();
        positions.resize(V.rows());
        for (int v = 0; v < V.rows(); v++)
            positions[v] = glm::vec3(V(v, 0), V(v, 1), V(v, 2));
    }

    void triangleBarycenter(int id, glm::vec3& barycenter) {
        assert(id < F.rows());
        barycenter = glm::vec3(0.0f);
        for (int i = 0; i < 3; i++)
            barycenter += glm::vec3(V(F(id, i), 0), V(F(id, i), 1), V(F(id, i), 2));
        barycenter /= 3;
    }

    void trianglesBarycenters(std::vector<glm::vec3>& barycenters) {
        barycenters.clear();
        barycenters.resize(F.rows(), glm::vec3(0.0f));
        for (int t = 0; t < F.rows(); t++) {
            for (int i = 0; i < 3; i++)
                barycenters[t] += glm::vec3(V(F(t, i), 0), V(F(t, i), 1), V(F(t, i), 2));
            barycenters[t] /= 3;
        }
    }

    void verticesAndTrianglesPositions(std::vector<glm::vec3>& positions) {
        verticesPositions(positions);
        std::vector<glm::vec3> trPositions;
        trianglesBarycenters(trPositions);
        positions.insert(positions.end(), trPositions.begin(), trPositions.end());
        assert(positions.size() == F.rows() + V.rows());
    }

    bool hasTangentField() {
        return vfield.size() > 0;
    }

    void clearTangentField() {
        vfield.clear();
    }

    void setTangentField(const std::vector<TangentSpace>& vfield) {
        assert(vfield.size() == V.rows());
        this->vfield = vfield;
    }

    const Eigen::MatrixXd& getVerticesMatrix() const {
        return V;
    }

    const Eigen::MatrixXi& getFacesMatrix() const {
        return F;
    }

    const std::vector<TangentSpace>& getTangentField() const {
        return vfield;
    }
};

#endif