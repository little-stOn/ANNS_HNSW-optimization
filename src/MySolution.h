#ifndef CPP_SOLUTION_H
#define CPP_SOLUTION_H

#include <vector>
#include <utility>
#include <cmath>
#include <cstdint>
#include <atomic>
#include <memory>
#include <cstring>
#include <immintrin.h>
#include <algorithm>
#include <mutex>
#include <string>
#include <iostream>

// Aligned SpinLock to prevent false sharing
struct alignas(64) SpinLock
{
    std::atomic_flag flag = ATOMIC_FLAG_INIT;

    inline void lock()
    {
        while (flag.test_and_set(std::memory_order_acquire))
        {
#if defined(__SSE2__)
            _mm_pause();
#endif
        }
    }

    inline void unlock()
    {
        flag.clear(std::memory_order_release);
    }
};

class Solution
{
public:
    Solution();
    ~Solution();

    void build(int d, const std::vector<float> &base);
    void search(const std::vector<float> &query, int *res);

    void clear_distance_counter();
    uint64_t get_distance_count() const;
    double get_avg_distance_count() const;

    // SQ16 settings
    void set_sq16_params(bool enable = true, bool use_avx2 = true);

    // Interface for evaluate.cpp compatibility
    bool load_graph(const std::string &filename);
    void save_graph(const std::string &filename);
    std::vector<int> get_degree_distribution();
    void set_gamma(float g);
    void set_ef_search(int ef);

private:
    // HNSW parameters
    int M;
    int maxM0;
    int ef_construction;
    int ef_search;
    float m_log_factor;
    int dim;
    int dim_aligned;
    int num_points;

    // Gamma search parameter
    float search_gamma;

    // Base data
    std::vector<float> base_data;

    // --- Graph Storage (Standard) ---
    char *graph_mem;
    std::vector<size_t> node_offsets;
    std::vector<int> levels;

    size_t size_l0_entry;
    size_t size_lx_entry;

    inline int *get_neighbor_link_ptr(int node_idx, int level) const;

    // Concurrency
    std::vector<SpinLock> node_locks;
    std::atomic<int> max_level_atomic;
    std::atomic<int> enter_point_atomic;

    inline SpinLock &get_lock(int id);

    // Flat L0 Graph (Disabled in this Ablation)
    std::vector<int> flat_l0;
    int stride_l0;
    void finalize_graph();

    // Global State
    int max_level;
    int enter_point;

    // --- Thread Context ---
    struct SearchContext
    {
        std::vector<uint16_t> visited_tags;
        uint16_t curr_visit_tag;
        std::vector<int> local_nbrs_copy;
        std::vector<int16_t> q_int_buffer;
        uint64_t local_dist_count;

        SearchContext() : curr_visit_tag(0), local_dist_count(0) {}

        void resize(int n_points, int d_aligned)
        {
            if (visited_tags.size() != (size_t)n_points)
            {
                visited_tags.assign(n_points, 0);
                curr_visit_tag = 0;
            }
            size_t alloc_len = d_aligned + 32;
            if (q_int_buffer.size() < alloc_len)
            {
                q_int_buffer.resize(alloc_len);
            }
        }

        inline void reset_visited()
        {
            curr_visit_tag++;
            if (curr_visit_tag == 0)
            {
                curr_visit_tag = 1;
                std::memset(visited_tags.data(), 0, visited_tags.size() * sizeof(uint16_t));
            }
        }

        inline bool is_visited(int id) const
        {
            return visited_tags[id] == curr_visit_tag;
        }

        inline void mark_visited(int id)
        {
            visited_tags[id] = curr_visit_tag;
        }
    };

    float L2_query_point(const std::vector<float> &q, int idx) const;
    const float *point_ptr(int idx) const;

    float L2_sq16_int_avx2(const int16_t *q_int, int idx) const;
    float L2_sq16_int_scalar(const int16_t *q_int, int idx) const;

    template <bool USE_SQ>
    inline float get_dist_template(const std::vector<float> &q_float, const int16_t *q_int, int idx, SearchContext &ctx);

    void connect_new_node(int cur, const std::vector<std::pair<float, int>> &cand, int level, SearchContext &ctx);

    // [ABLATION] Note: USE_FLAT will be forced to false in implementation
    template <bool USE_SQ, bool USE_FLAT, bool IS_BUILD, bool USE_GAMMA>
    std::vector<std::pair<float, int>> search_layer_template(
        const std::vector<float> &q_float,
        const int16_t *q_int,
        int enter, int ef, int k_search, int level, SearchContext &ctx);

    std::vector<int> select_neighbors_heuristic(int new_node, const std::vector<std::pair<float, int>> &cand, int Mlevel, SearchContext &ctx);

    int compute_dynamic_M(int baseM, int cur, int level);

    mutable std::atomic<uint64_t> dist_comp_counter{0};
    mutable std::atomic<uint64_t> query_counter{0};

    bool sq16_enabled;
    bool sq16_use_avx2;

    float g_min_val;
    float g_scale;
    std::vector<int16_t> sq16_data;

    void compute_sq16_params_global();
    void encode_base_with_sq16_global();

    void *aligned_alloc_wrapper(size_t size, size_t alignment);
    void aligned_free_wrapper(void *p);
};

#endif