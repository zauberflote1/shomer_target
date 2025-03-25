#pragma once
#include <Eigen/Dense>
#include <array>
#include <vector>
#include <algorithm>

using Vec2 = Eigen::Vector2d;

inline double fast_norm(const Vec2& v) {
    return v.squaredNorm(); // Avoid sqrt for faster max element selection
}

inline Vec2 FindMidpoint(const std::vector<Vec2>& candidate_target_list, const std::array<uint8_t, 2>& p1p2) {
    return (candidate_target_list[p1p2[0]] + candidate_target_list[p1p2[1]]) * 0.5;
}

inline void Midpoint2P3P4(const Vec2& midpoint, const std::vector<Vec2>& candidate_target_list, 
                           const std::array<uint8_t, 2>& p3p4, std::array<double, 2>& short_lengths) {
    short_lengths[0] = (midpoint - candidate_target_list[p3p4[0]]).norm();
    short_lengths[1] = (midpoint - candidate_target_list[p3p4[1]]).norm();
}

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
