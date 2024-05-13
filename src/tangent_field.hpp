#ifndef TANGENT_FIELD_HPP
#define TANGENT_FIELD_HPP

#include <cmath>
#include <glm/glm.hpp>

class TangentField {
private:
    float tMagnitude;
    float bMagnitude;
    glm::vec3 tDirection;
    glm::vec3 bDirection;

public:
    TangentField() : tMagnitude{ -1 }, bMagnitude{ -1 }, tDirection{ 0.0 }, bDirection{ 0.0 } {}
    TangentField(float tMagnitude, float bMagnitude, glm::vec3 tDirection, glm::vec3 bDirection) :
        tMagnitude{ tMagnitude }, bMagnitude{ bMagnitude }, tDirection{ tDirection }, bDirection{ bDirection } {}

    float euclideanDistance(glm::vec3 delta) const {
        float tDot = glm::dot(tMagnitude * tDirection, delta);
        float bDot = glm::dot(bMagnitude * bDirection, delta);
        return std::sqrt(tDot * tDot + bDot * bDot);
    }

    float manhattanDistance(glm::vec3 delta) const {
        float tDot = glm::dot(tMagnitude * tDirection, delta);
        float bDot = glm::dot(bMagnitude * bDirection, delta);
        return std::abs(tDot) + std::abs(bDot);
    }

    static void average(const TangentField& field0, const TangentField& field1, TangentField& avg) {
        std::vector<float> dots(4, 0);
        dots[0] = glm::dot(field0.tDirection, field1.tDirection);
        dots[1] = glm::dot(field0.tDirection, field1.bDirection);
        dots[2] = glm::dot(field0.tDirection, -field1.tDirection);
        dots[3] = glm::dot(field0.tDirection, -field1.bDirection);
        float dotMax = -INF;
        int dotMaxIndex = -1;
        for (int i = 0; i < 4; i++) {
            if (dots[i] > dotMax) {
                dotMax = dots[i];
                dotMaxIndex = i;
            }
        }
        glm::vec3 tMixedDir, bMixedDir;
        float tMixedMag = (field0.tMagnitude + field1.tMagnitude) * 0.5f;
        float bMixedMag = (field0.bMagnitude + field1.bMagnitude) * 0.5f;
        switch (dotMaxIndex) {
        case 0:
            tMixedDir = glm::normalize((field0.tDirection + field1.tDirection) * 0.5f) * tMixedMag;
            bMixedDir = glm::normalize((field0.bDirection + field1.bDirection) * 0.5f) * bMixedMag;
            break;
        case 1:
            tMixedDir = glm::normalize((field0.tDirection + field1.bDirection) * 0.5f) * tMixedMag;
            bMixedDir = glm::normalize((field0.bDirection - field1.tDirection) * 0.5f) * bMixedMag;
            break;
        case 2:
            tMixedDir = glm::normalize((field0.tDirection - field1.tDirection) * 0.5f) * tMixedMag;
            bMixedDir = glm::normalize((field0.bDirection - field1.bDirection) * 0.5f) * bMixedMag;
            break;
        case 3:
            tMixedDir = glm::normalize((field0.tDirection - field1.bDirection) * 0.5f) * tMixedMag;
            bMixedDir = glm::normalize((field0.bDirection + field1.tDirection) * 0.5f) * bMixedMag;
            break;
        }
        avg = TangentField(tMixedMag, bMixedMag, tMixedDir, bMixedDir);
    }

    void tangent(glm::vec3& t) const {
        t = tDirection * tMagnitude;
    }

    void bitangent(glm::vec3& b) const {
        b = bDirection * bMagnitude;
    }
};

#endif // !TANGENT_FIELD_HPP
