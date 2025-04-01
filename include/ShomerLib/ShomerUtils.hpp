#pragma once
#include <Eigen/Dense>
#include <array>
#include <vector>
#include <algorithm>

using Vec2 = Eigen::Vector2d;

/**
 * @brief Computes the squared norm of a 2D vector.
 * 
 * This function calculates the squared norm (magnitude squared) of the given
 * 2D vector `v`. It avoids computing the square root, making it faster for
 * use cases where the exact norm is not required, such as selecting the 
 * maximum element based on magnitude.
 * 
 * @param v The 2D vector for which the squared norm is to be computed.
 * @return The squared norm of the vector `v`.
 */
inline double fast_norm(const Vec2& v) {
    return v.squaredNorm(); // Avoid sqrt for faster max element selection
}

/**
 * @brief Computes the midpoint between two points in a list of 2D vectors.
 * 
 * @param candidate_target_list A vector of Vec2 objects representing the list of 2D points.
 * @param p1p2 An array of two indices (uint8_t) specifying the positions of the points in the list
 *             for which the midpoint is to be calculated.
 * @return Vec2 The midpoint between the two specified points.
 * 
 * @note The indices in p1p2 must be valid and within the bounds of candidate_target_list.
 */
inline Vec2 FindMidpoint(const std::vector<Vec2>& candidate_target_list, const std::array<uint8_t, 2>& p1p2) {
    return (candidate_target_list[p1p2[0]] + candidate_target_list[p1p2[1]]) * 0.5;
}

/**
 * @brief Computes the Euclidean distances between a given midpoint and two candidate target points.
 * 
 * This function calculates the distances from the specified midpoint to two points in the 
 * candidate target list, identified by their indices in the `p3p4` array. The computed distances 
 * are stored in the `short_lengths` array.
 * 
 * @param midpoint A 2D vector representing the midpoint.
 * @param candidate_target_list A vector of 2D vectors representing the candidate target points.
 * @param p3p4 An array of two indices specifying the positions of the target points in the 
 *             `candidate_target_list` to compute distances to.
 * @param short_lengths An array of two doubles where the computed distances will be stored. 
 *                      `short_lengths[0]` will hold the distance to the first target point 
 *                      (indexed by `p3p4[0]`), and `short_lengths[1]` will hold the distance 
 *                      to the second target point (indexed by `p3p4[1]`).
 */
inline void Midpoint2P3P4(const Vec2& midpoint, const std::vector<Vec2>& candidate_target_list, 
                           const std::array<uint8_t, 2>& p3p4, std::array<double, 2>& short_lengths) {
    short_lengths[0] = (midpoint - candidate_target_list[p3p4[0]]).norm();
    short_lengths[1] = (midpoint - candidate_target_list[p3p4[1]]).norm();
}

/**
 * @brief Determines the indices of two points (p1 and p2) based on the cross products
 *        of vectors relative to a reference vector.
 *
 * This function calculates the cross products of the reference vector `v_p3p4` with
 * two other vectors `v_p3pa` and `v_p3pb`. It uses the signs of these cross products
 * to determine the order of the indices `p1` and `p2` from the input array `p1p2`.
 * If the cross products have opposite signs, the function assigns the indices
 * accordingly and returns true. Otherwise, it returns false.
 *
 * @param v_p3p4 The reference vector.
 * @param v_p3pa The vector to the first point.
 * @param v_p3pb The vector to the second point.
 * @param p1p2 An array containing two indices to be assigned to `p1` and `p2`.
 * @param[out] p1 The first output index, determined based on the cross product.
 * @param[out] p2 The second output index, determined based on the cross product.
 * @return true if the cross products have opposite signs and indices are assigned;
 *         false otherwise.
 */
inline bool FindP1P2Indices(const Vec2& v_p3p4, const Vec2& v_p3pa, const Vec2& v_p3pb,
                      const std::array<uint8_t, 2>& p1p2, uint8_t& p1, uint8_t& p2) {
    double cross_pa = v_p3p4.x() * v_p3pa.y() - v_p3p4.y() * v_p3pa.x();
    double cross_pb = v_p3p4.x() * v_p3pb.y() - v_p3p4.y() * v_p3pb.x();

    if (cross_pa * cross_pb < 0) {
        p1 = (cross_pa > 0) ? p1p2[1] : p1p2[0];
        p2 = (cross_pa > 0) ? p1p2[0] : p1p2[1];
        return true;
    }
    return false;
}

/**
 * @brief Sorts a list of candidate target points based on tetrahedron geometry and camera intrinsics.
 *
 * This function identifies and sorts four points from the given candidate target list such that they
 * form a tetrahedron-like structure in the image plane. The sorted points are transformed using the
 * provided camera intrinsics and stored in the output parameter `sortedImagePoints`.
 *
 * @param candidate_target_list A vector of 2D points (Vec2) representing the candidate target points.
 * @param cam_intrinsics A vector of camera intrinsic parameters in the form:
 *                       [fx, cx, fy, cy], where fx and fy are the focal lengths, and cx and cy are
 *                       the principal point offsets.
 * @param sortedImagePoints An Eigen matrix (4x2) to store the sorted and transformed image points.
 *                          Each row corresponds to a sorted point in the image plane.
 *                          The points are ordered as p1, p2, p3, and p4.
 * 
 * @return true if the sorting and transformation were successful, false otherwise.
 *
 * @note The function assumes that the input `candidate_target_list` contains at least four points.
 *       The function uses geometric properties such as distances and midpoints to determine the
 *       sorting order.
 */
inline bool SortTargetsUsingTetrahedronGeometry(const std::vector<Vec2>& candidate_target_list, const std::vector<double>& cam_intrinsics,
                                         Eigen::Matrix<double, 4, 2>& sortedImagePoints) {
    constexpr std::array<std::array<uint8_t, 2>, 6> idx_lookup_table = {{
        {0,1}, {0,2}, {0,3}, {1,2}, {1,3}, {2,3}
    }};

    constexpr std::array<std::array<uint8_t, 2>, 6> not_idx_lookup_table = {{
        {2,3}, {1,3}, {1,2}, {0,3}, {0,2}, {0,1}
    }};

    std::array<double, 6> lengths;
    for (size_t i = 0; i < 6; ++i) {
        lengths[i] = fast_norm(candidate_target_list[idx_lookup_table[i][0]] - candidate_target_list[idx_lookup_table[i][1]]);
    }

    auto p1p2_idx = std::distance(lengths.begin(), std::max_element(lengths.begin(), lengths.end()));
    const auto& p1p2 = idx_lookup_table[p1p2_idx];
    Vec2 midpoint = FindMidpoint(candidate_target_list, p1p2);
    const auto& p3p4 = not_idx_lookup_table[p1p2_idx];

    std::array<double, 2> short_lengths;
    Midpoint2P3P4(midpoint, candidate_target_list, p3p4, short_lengths);

    uint8_t p3 = short_lengths[0] < short_lengths[1] ? p3p4[0] : p3p4[1];
    uint8_t p4 = short_lengths[0] < short_lengths[1] ? p3p4[1] : p3p4[0];

    Vec2 v_p3p4 = candidate_target_list[p3] - candidate_target_list[p4];
    Vec2 v_p3pa = candidate_target_list[p3] - candidate_target_list[p1p2[0]];
    Vec2 v_p3pb = candidate_target_list[p3] - candidate_target_list[p1p2[1]];

    uint8_t p1, p2;
    if (FindP1P2Indices(v_p3p4, v_p3pa, v_p3pb, p1p2, p1, p2)) {
        sortedImagePoints << candidate_target_list[p1].x()*cam_intrinsics[0] + cam_intrinsics[1], candidate_target_list[p1].y()*cam_intrinsics[2] + cam_intrinsics[3],
                             candidate_target_list[p2].x()*cam_intrinsics[0] + cam_intrinsics[1], candidate_target_list[p2].y()*cam_intrinsics[2] + cam_intrinsics[3],
                             candidate_target_list[p3].x()*cam_intrinsics[0] + cam_intrinsics[1], candidate_target_list[p3].y()*cam_intrinsics[2] + cam_intrinsics[3],
                             candidate_target_list[p4].x()*cam_intrinsics[0] + cam_intrinsics[1], candidate_target_list[p4].y()*cam_intrinsics[2] + cam_intrinsics[3];
        return true;
    }

    return false;
}
