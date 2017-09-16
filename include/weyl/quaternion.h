#pragma once
#ifndef _WEYL_QUATERNION_H_
#define _WEYL_QUATERNION_H_

#include "weyl/vector.h"
#include "weyl/matrix.h"

namespace weyl
{
    /// \class quaternion
    /// \brief Quaternion with generic floating point type
    template <typename T>
    class quaternion
    {
    public:
        typedef vector<T, 3> vtype;

    public:
        quaternion() : s(static_cast<T>(1.0)) {}

        quaternion(const quaternion& other) : s(other.s), v(other.v) {}

        quaternion(T s, const vtype& v) : s(s), v(v) {}

        quaternion(const vtype& angles) { from_euler(angles); }

        quaternion(const vtype& a, const vtype& b) { from_rotation(a,b); }

        ~quaternion() {}

        quaternion& operator=(const quaternion& other) {
            s = other.s;
            v = other.v;
            return *this;
        }

        quaternion& operator+=(const quaternion& other) {
            s += other.s;
            v += other.v;
            return *this;
        }

        quaternion& operator-=(const quaternion& other) {
            s -= other.s;
            v -= other.v;
            return *this;
        }

        quaternion& operator*=(const quaternion& other) {
            T s_old = s;
            s = (s * other.s) - dot(v, other.v);
            v = (s_old * other.v) + (v * other.s) + cross(v, other.v);
            return *this;
        }

        quaternion& operator*=(float c) {
            s *= c;
            v *= c;
            return *this;
        }

        operator matrix<T, 3, 3>() const {
            vec3 v2 = v * v;
            return matrix<T, 3, 3>({
                { 1.0f - 2.0f * (v2[1] + v2[2]), 2.0f * (v[0]*v[1] + s*v[2]), 2.0f * (v[0]*v[2] - s*v[1]) },
                { 2.0f * (v[0]*v[1] - s*v[2]), 1.0f - 2.0f * (v2[0] + v2[2]), 2.0f * (v[1]*v[2] + s*v[0]) },
                { 2.0f * (v[0]*v[2] + s*v[1]), 2.0f * (v[1]*v[2] - s*v[0]), 1.0f - 2.0f * (v2[0] + v2[1]) }
            }).transpose();
        }

        operator matrix<T, 4, 4>() const {
            matrix<T, 3, 3> m33 = operator matrix<T, 3, 3>();
            matrix<T, 4, 4> m44(0.0f);
            for (size_t i = 0; i < 3; ++i)
            for (size_t j = 0; j < 3; ++j)
                m44[i][j] = m33[i][j];
            m44[3][3] = 1.0f;
            return m44;
        }

        bool operator==(const quaternion& other) const { return (s == other.s) && (v == other.v); }

        bool operator!=(const quaternion& other) const { return !operator==(other); }

        void from_euler(const vtype& angles) {
            float c1 = cos(angles[0]*0.5f);
            float s1 = sin(angles[0]*0.5f);
            float c2 = cos(angles[1]*0.5f);
            float s2 = sin(angles[1]*0.5f);
            float c3 = cos(angles[2]*0.5f);
            float s3 = sin(angles[2]*0.5f);
            s     = c1*c2*c3 + s1*s2*s3;
            v[0] = s1*c2*c3 - c1*s2*s3;
            v[1] = c1*s2*c3 + s1*c2*s3;
            v[2] = c1*c2*s3 - s1*s2*c3;

            // Just to be safe...
            normalize();
        }

        static void axis_angle(quaternion& q, const vtype& axis, float angle) {
            float half_angle = angle * 0.5f;
            q.s = cos(half_angle);
            q.v = axis * sin(half_angle);
            q.normalize();
        }

        static void axis_angle(quaternion& q, const vtype& axis) {
            float angle = length(axis);
            q = quaternion<T>::axis_angle(abs(angle) > std::numeric_limits<float>::epsilon() ? axis / angle : vec3(0.0f), angle);
        }

        static quaternion axis_angle(const vtype& axis, float angle) {
            quaternion q;
            axis_angle(q, axis, angle);
            return q;
        }

        static quaternion axis_angle(const vtype& axis) {
            quaternion q;
            axis_angle(q, axis);
            return q;
        }

        // Adapted from: http://lolengine.net/blog/2013/09/18/beautiful-maths-quaternion-from-vectors
        void from_rotation(const vtype& a, const vtype& b) {

            float norm_a_norm_b = sqrt(dot(a, a) * dot(b, b));
            if (norm_a_norm_b < std::numeric_limits<float>::epsilon()) {
                v = vec3(0.0f, 1.0f, 0.0f);
                s = 0.0f;
                return;
            }
            float cos_theta = dot(a, b) / norm_a_norm_b;
            if (cos_theta < -1.0f + std::numeric_limits<float>::epsilon()) {
                v = vec3(0.0f, 1.0f, 0.0f);
                s = 0.0f;
                return;
            }

            s = sqrt(0.5f * (1.f + cos_theta));
            v = cross(a, b) / (norm_a_norm_b * 2.f * s);

            // Unsure that this is necessary...
            normalize();
        }


        quaternion& normalize() {
            T m_inv = 1.0f / magnitude();
            s *= m_inv;
            v *= m_inv;
            return *this;
        }

        quaternion& conjugate() {
            v *= -1.0f;
            return *this;
        }

        quaternion normalized() const {
            quaternion q(*this);
            return q.normalize();
        }

        quaternion conjugated() const {
            quaternion q(*this);
            return q.conjugate();
        }

        T magnitude() const {
            return sqrt(dot(*this, *this));
        }

    public:
        T s;
        vtype v;
    };

    template <typename T>
    T dot(const quaternion<T>& a, const quaternion<T>& b) {
        return (a.s * b.s) + (a.v[0] * b.v[0]) + (a.v[1] * b.v[1]) + (a.v[2] * b.v[2]);
    }

    template <typename T>
    quaternion<T> normalize(quaternion<T> q) {
        return q.normalized();
    }

    template <typename T>
    quaternion<T> operator+(quaternion<T> a, const quaternion<T>& b) {
        return a += b;
    }

    template <typename T>
    quaternion<T> operator-(quaternion<T> a, const quaternion<T>& b) {
        return a -= b;
    }

    template <typename T>
    quaternion<T> operator*(quaternion<T> a, const quaternion<T>& b) {
        return a *= b;
    }

    template <typename T>
    quaternion<T> operator*(quaternion<T> q, T c) {
        return q *= c;
    }

    template <typename T>
    quaternion<T> operator*(T c, quaternion<T> q) {
        return q *= c;
    }

    template <typename T>
    vector<T, 3> operator*(const quaternion<T>& q, const vector<T, 3>& v) {
        return ((q * quaternion<T>(0.0f, v)) * q.conjugated()).v;
    }

    template <typename T>
    matrix<T, 3, 3> operator*(const quaternion<T>& q, const matrix<T, 3, 3>& m) {
        matrix<T, 3, 3> qm(q);
        return (qm * m) * qm.transpose();
    }
}

#endif
