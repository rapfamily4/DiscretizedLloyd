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
    Eigen::MatrixXd N;
    Eigen::MatrixXd T;
    Eigen::MatrixXd B;


    void vertexTriangleAdjacency(std::vector<std::vector<int>>& VT) {
        VT.clear();
        VT.resize(V.rows());
        for (int t = 0; t < F.rows(); t++) // t-th triangle.
            for (int v = 0; v < F.cols(); v++) // v-th vertex of the triangle.
                VT[F(t, v)].push_back(t);
    }

    void tangentVectorsFromVertices(int i, int j, glm::vec3& tangent, glm::vec3& bitangent) {
        tangent = glm::vec3(
            (T(i, 0) + T(j, 0)) * 0.5f,
            (T(i, 1) + T(j, 1)) * 0.5f,
            (T(i, 2) + T(j, 2)) * 0.5f
        );
        bitangent = glm::vec3(
            (B(i, 0) + B(j, 0)) * 0.5f,
            (B(i, 1) + B(j, 1)) * 0.5f,
            (B(i, 2) + B(j, 2)) * 0.5f
        );
    }

    void tangentVectorsFromTriangle(int t, glm::vec3& tangent, glm::vec3& bitangent) {
        int i = F(t, 0), j = F(t, 1), k = F(t, 2);
        tangent = glm::vec3(
            (T(i, 0) + T(j, 0) + T(k, 0)) / 3.0f,
            (T(i, 1) + T(j, 1) + T(k, 1)) / 3.0f,
            (T(i, 2) + T(j, 2) + T(k, 2)) / 3.0f
        );
        bitangent = glm::vec3(
            (B(i, 0) + B(j, 0) + B(k, 0)) / 3.0f,
            (B(i, 1) + B(j, 1) + B(k, 1)) / 3.0f,
            (B(i, 2) + B(j, 2) + B(k, 2)) / 3.0f
        );
    }

    float verticesDistance(int i, int j) {
        assert(i < V.rows() && j < V.rows());
        
        glm::vec3 vertex = glm::vec3(V(i, 0), V(i, 1), V(i, 2));
        glm::vec3 neighbor = glm::vec3(V(j, 0), V(j, 1), V(j, 2));
        if (T.rows() != 0 && B.rows() != 0) {
            glm::vec3 tangent, bitangent;
            tangentVectorsFromVertices(i, j, tangent, bitangent);
            float tDot = glm::dot(tangent, neighbor - vertex);
            float bDot = glm::dot(bitangent, neighbor - vertex);
            return std::sqrt(tDot * tDot + bDot * bDot);
        }
        else return glm::distance(vertex, neighbor);
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

    float oppositeVerticesDistance(int i, int j, HalfEdge trianglesBase) {
        glm::vec3 delta = glm::vec3(
            V(i, 0) - V(j, 0),
            V(i, 1) - V(j, 1),
            V(i, 2) - V(j, 2)
        );
        float dist = oppositePointsDistance(delta, trianglesBase);
        if (T.rows() != 0 && B.rows() != 0) {
            glm::vec3 tangent, bitangent;
            tangentVectorsFromVertices(i, j, tangent, bitangent);
            delta = glm::normalize(-delta) * dist;
            float tDot = glm::dot(tangent, delta);
            float bDot = glm::dot(bitangent, delta);
            return std::sqrt(tDot * tDot + bDot * bDot);
        }
        else return dist;
    }

    float trianglesDistance(int i, int j, HalfEdge trianglesBase) {
        glm::vec3 iBarycenter, jBarycenter;
        triangleBarycenter(i, iBarycenter);
        triangleBarycenter(j, jBarycenter);
        float dist = oppositePointsDistance(iBarycenter - jBarycenter, trianglesBase);
        if (T.rows() != 0 && B.rows() != 0) {
            glm::vec3 iTangent, jTangent, avgTangent, iBitangent, jBitangent, avgBitangent;
            tangentVectorsFromTriangle(i, iTangent, iBitangent);
            tangentVectorsFromTriangle(j, jTangent, jBitangent);
            avgTangent = (iTangent + jTangent) * 0.5f;
            avgBitangent = (iBitangent + jBitangent) * 0.5f;
            glm::vec3 delta = glm::normalize(jBarycenter - iBarycenter) * dist;
            float tDot = glm::dot(avgTangent, delta);
            float bDot = glm::dot(avgBitangent, delta);
            return std::sqrt(tDot * tDot + bDot * bDot);
        }
        else return dist;
    }

    float vertexTriangleDistance(int v, int t) {
        glm::vec3 vPoint, tBarycenter;
        vPoint = glm::vec3(V(v, 0), V(v, 1), V(v, 2));
        triangleBarycenter(t, tBarycenter);
        if (T.rows() != 0 && B.rows() != 0) {
            glm::vec3 tTangent, avgTangent, tBitangent, avgBitangent;
            tangentVectorsFromTriangle(t, tTangent, tBitangent);
            avgTangent = glm::vec3(
                (T(v, 0) + tTangent.x) * 0.5f,
                (T(v, 1) + tTangent.y) * 0.5f,
                (T(v, 2) + tTangent.z) * 0.5f
            );
            avgBitangent = glm::vec3(
                (B(v, 0) + tBitangent.x) * 0.5f,
                (B(v, 1) + tBitangent.y) * 0.5f,
                (B(v, 2) + tBitangent.z) * 0.5f
            );
            glm::vec3 delta = tBarycenter - vPoint;
            float tDot = glm::dot(avgTangent, delta);
            float bDot = glm::dot(avgBitangent, delta);
            return std::sqrt(tDot * tDot + bDot * bDot);
        }
        else return glm::length(tBarycenter - vPoint);
    }

public:
    Model() {}
    Model(Eigen::MatrixXd vertices, Eigen::MatrixXi triangles, Eigen::MatrixXd verticesNormals) :
        V{ vertices },
        F{ triangles },
        N{ verticesNormals } {
    }

    void clearTangentSpace() {
        T.conservativeResize(0, 3);
        B.conservativeResize(0, 3);
    }

    void fillTangentSpaceFlat(glm::vec3& tangent, glm::vec3& bitangent) {
        T.resize(V.rows(), 3);
        B.resize(V.rows(), 3);
        for (int i = 0; i < V.rows(); i++) {
            T.row(i) << tangent.x, tangent.y, tangent.z;
            B.row(i) << bitangent.x, bitangent.y, bitangent.z;
        }
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

    const Eigen::MatrixXd& getVerticesMatrix() const {
        return V;
    }

    const Eigen::MatrixXi& getFacesMatrix() const {
        return F;
    }
};

#endif