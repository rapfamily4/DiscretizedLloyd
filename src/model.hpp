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

class Model {
private:
    struct HalfEdge {
        int i;
        int j;
        float length;
        HalfEdge() : i{ -1 }, j{ -1 }, length{ 0.0f } {}
        HalfEdge(int i, int j, float length) : i{ i }, j{ j }, length{ length } { }
        HalfEdge flip() const { return HalfEdge(j, i, length); }
        bool operator<(const HalfEdge& other) const {
            if (i < other.i) return true;
            if (i > other.i) return false;
            return j < other.j;
        }
    };

    Eigen::MatrixXd V;
    Eigen::MatrixXi F;

    float edgeLength(int i, int j) {
        assert(i < V.rows() && j < V.rows());
        glm::vec3 vi = glm::vec3(V(i, 0), V(i, 1), V(i, 2));
        glm::vec3 vj = glm::vec3(V(j, 0), V(j, 1), V(j, 2));
        return glm::distance(vi, vj);
    }

    void vertexTriangleAdjacency(std::vector<std::vector<int>>& VT) {
        VT.clear();
        VT.resize(V.rows());
        for (int t = 0; t < F.rows(); t++) // t-th triangle.
            for (int v = 0; v < F.cols(); v++) // v-th vertex of the triangle.
                VT[F(t, v)].push_back(t);
    }

    float oppositeVerticesDistance(glm::vec3 vertexDelta, HalfEdge trianglesBase) {
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

    float oppositeVerticesDistance(int vertexId0, int vertexId1, HalfEdge trianglesBase) {
        glm::vec3 delta = glm::vec3(
            V(vertexId0, 0) - V(vertexId1, 0),
            V(vertexId0, 1) - V(vertexId1, 1),
            V(vertexId0, 2) - V(vertexId1, 2)
        );
        return oppositeVerticesDistance(delta, trianglesBase);
    }

public:
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
            glm::vec3 vertex = glm::vec3((float)V(i, 0), (float)V(i, 1), (float)V(i, 2));
            for (int j = 0; j < adjacency[i].size(); j++) {
                // i-th vertex, j-th neighbor of index adjacency[i][j].
                int nId = adjacency[i][j];
                glm::vec3 neighbor = glm::vec3((float)V(nId, 0), (float)V(nId, 1), (float)V(nId, 2));
                distances[i].push_back(glm::distance(vertex, neighbor));
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
                HalfEdge edge(i, j, edgeLength(i, j));
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
        glm::vec3 tBarycenter;
        glm::vec3 nBarycenter;
        std::map<HalfEdge, int> edges;

        for (int t = 0; t < F.rows(); t++) { // t-th triangle.
            triangleBarycenter(t, tBarycenter);
            for (int v = 0; v < F.cols(); v++) { // v-th vertex of the triangle.
                int i = F(t, v), j = F(t, (v + 1) % F.cols());
                HalfEdge edge(i, j, edgeLength(i, j));
                HalfEdge flipped = edge.flip();
                if (edges.find(flipped) != edges.end()) {
                    int n = edges[flipped];
                    assert(t != n);
                    adjacency[t].push_back(n);
                    adjacency[n].push_back(t);
                    //float w = (triangleArea(t) + triangleArea(n)) / edge.length;
                    triangleBarycenter(n, nBarycenter);
                    float w = oppositeVerticesDistance(tBarycenter - nBarycenter, edge);
                    distances[t].push_back(w);
                    distances[n].push_back(w);
                }
                else edges[edge] = t;
            }
        }
    }

    void denseTriangleAdjacency(std::vector<std::vector<int>>& adjacency, std::vector<std::vector<float>>& distances) {
        triangleAdjacency(adjacency, distances);
        std::vector<std::vector<int>> VT;
        vertexTriangleAdjacency(VT);
        glm::vec3 tBarycenter;
        glm::vec3 nBarycenter;

        for (int t = 0; t < F.rows(); t++) { // t-th triangle.
            triangleBarycenter(t, tBarycenter);
            for (int v = 0; v < F.cols(); v++) { // v-th vertex of the triangle.
                int vId = F(t, v);
                glm::vec3 vPoint = glm::vec3(V(vId, 0), V(vId, 1), V(vId, 2));
                for (int n : VT[vId]) // Triangle n, neighbor of vertex vId.
                    if (n != t && std::find(adjacency[t].begin(), adjacency[t].end(), n) == adjacency[t].end()) {
                        adjacency[t].push_back(n);
                        triangleBarycenter(n, nBarycenter);
                        float w = glm::length(vPoint - tBarycenter) + glm::length(vPoint - nBarycenter);
                        distances[t].push_back(w);
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

        glm::vec3 tBarycenter;
        for (int t = 0; t < F.rows(); t++) {
            triangleBarycenter(t, tBarycenter);
            for (int v = 0; v < 3; v++) {
                int vId = F(t, v);
                adjacency[t + V.rows()].push_back(vId);
                adjacency[vId].push_back(t + V.rows());
                glm::vec3 vPoint = glm::vec3(V(vId, 0), V(vId, 1), V(vId, 2));
                float w = glm::length(vPoint - tBarycenter);
                distances[t + V.rows()].push_back(w);
                distances[vId].push_back(w);
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

    void triangleBarycenters(std::vector<glm::vec3>& barycenters) {
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
        triangleBarycenters(trPositions);
        positions.insert(positions.end(), trPositions.begin(), trPositions.end());
        assert(positions.size() == F.rows() + V.rows());
    }

    const Eigen::MatrixXd& getVerticesMatrix() const {
        return V;
    }

    const Eigen::MatrixXi& getFacesMatrix() const {
        return F;
    }
};

#endif