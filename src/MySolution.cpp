#include "MySolution.h"
#include <random>
#include <cmath>
#include <algorithm>
#include <vector>
#include <thread>
#include <immintrin.h>
#include <omp.h>
#include <queue>
#include <iostream>
#include <fstream>

#if defined(_MSC_VER) || defined(__MINGW32__)
#include <malloc.h>
#else
#include <stdlib.h>
#endif

using namespace std;

// --- Global Counters for evaluate.cpp ---
#ifdef COUNT_DIST
extern std::atomic<unsigned long long> g_dist_calc_count;
#endif

#ifdef TEST_GRAPH
extern std::atomic<unsigned long long> g_acc;
extern std::atomic<unsigned long long> g_tot;
#endif

// Simplified AVX2 float distance
__attribute__((target("avx2,fma"))) static float L2_vec_avx2_fma(const float *pa, const float *pb, int dim)
{
    __m256 sum = _mm256_setzero_ps();
    int i = 0;
    for (; i + 8 <= dim; i += 8)
    {
        __m256 a = _mm256_loadu_ps(pa + i);
        __m256 b = _mm256_loadu_ps(pb + i);
        __m256 d = _mm256_sub_ps(a, b);
        sum = _mm256_fmadd_ps(d, d, sum);
    }
    float dist = 0.0f;
    __m128 lo = _mm256_castps256_ps128(sum);
    __m128 hi = _mm256_extractf128_ps(sum, 1);
    lo = _mm_add_ps(lo, hi);
    lo = _mm_hadd_ps(lo, lo);
    lo = _mm_hadd_ps(lo, lo);
    dist = _mm_cvtss_f32(lo);
    for (; i < dim; ++i)
    {
        float d = pa[i] - pb[i];
        dist += d * d;
    }
    return dist;
}

Solution::Solution()
{
    M = 0;
    maxM0 = 144;
    ef_construction = 432;
    ef_search = 221;
    dim = 0;
    dim_aligned = 0;
    num_points = 0;
    max_level = -1;
    max_level_atomic = -1;
    enter_point = -1;
    enter_point_atomic = -1;
    m_log_factor = 1.0f / log(1.0f * M);
    search_gamma = 0.172f;
    sq16_enabled = true;
    sq16_use_avx2 = true;
    graph_mem = nullptr;
    dist_comp_counter = 0;
    query_counter = 0;
}

Solution::~Solution()
{
    if (graph_mem)
    {
        aligned_free_wrapper(graph_mem);
        graph_mem = nullptr;
    }
}

void *Solution::aligned_alloc_wrapper(size_t size, size_t alignment)
{
#if defined(_MSC_VER) || defined(__MINGW32__)
    return _aligned_malloc(size, alignment);
#else
    void *p = nullptr;
    if (posix_memalign(&p, alignment, size) != 0)
        return nullptr;
    return p;
#endif
}

void Solution::aligned_free_wrapper(void *p)
{
#if defined(_MSC_VER) || defined(__MINGW32__)
    _aligned_free(p);
#else
    free(p);
#endif
}

void Solution::set_sq16_params(bool enable, bool use_avx2)
{
    sq16_enabled = enable;
    sq16_use_avx2 = use_avx2;
}

void Solution::set_gamma(float g)
{
    search_gamma = g;
}

void Solution::set_ef_search(int ef)
{
    ef_search = ef;
}

std::vector<int> Solution::get_degree_distribution()
{
    std::vector<int> degrees;
    if (num_points == 0)
        return degrees;

    degrees.reserve(num_points);
    // [ABLATION] Always read from graph_mem, ignoring flat_l0
    for (int i = 0; i < num_points; ++i)
    {
        int *ptr = get_neighbor_link_ptr(i, 0);
        degrees.push_back(*ptr);
    }
    return degrees;
}

inline SpinLock &Solution::get_lock(int id)
{
    return node_locks[id];
}

const float *Solution::point_ptr(int idx) const
{
    return &base_data[idx * dim];
}

float Solution::L2_query_point(const std::vector<float> &q, int idx) const
{
    return L2_vec_avx2_fma(q.data(), point_ptr(idx), dim);
}

// ==========================================================
// SQ16 Implementation
// ==========================================================

void Solution::compute_sq16_params_global()
{
    if (base_data.empty())
        return;

    float min_v = std::numeric_limits<float>::max();
    float max_v = std::numeric_limits<float>::lowest();

#pragma omp parallel for reduction(min : min_v) reduction(max : max_v)
    for (size_t i = 0; i < base_data.size(); ++i)
    {
        float val = base_data[i];
        if (val < min_v)
            min_v = val;
        if (val > max_v)
            max_v = val;
    }

    g_min_val = min_v;
    float range = max_v - min_v;
    if (range < 1e-9f)
        range = 1.0f;
    g_scale = 32000.0f / range;
}

void Solution::encode_base_with_sq16_global()
{
    if (base_data.empty())
        return;

    dim_aligned = (dim + 15) & ~15;
    size_t total_elements = (size_t)num_points * dim_aligned;
    sq16_data.resize(total_elements);
    const float offset_target = -16000.0f;

#pragma omp parallel for
    for (int i = 0; i < num_points; ++i)
    {
        const float *src = &base_data[i * dim];
        int16_t *dst = &sq16_data[i * dim_aligned];

        for (int d = 0; d < dim; ++d)
        {
            float val = src[d];
            float encoded = (val - g_min_val) * g_scale + offset_target;
            if (encoded < -32767.0f)
                encoded = -32767.0f;
            if (encoded > 32767.0f)
                encoded = 32767.0f;
            dst[d] = (int16_t)(encoded + 0.5f);
        }
        for (int d = dim; d < dim_aligned; ++d)
            dst[d] = 0;
    }
}

float Solution::L2_sq16_int_scalar(const int16_t *q_int, int idx) const
{
    const int16_t *d_int = &sq16_data[idx * dim_aligned];
    float dist = 0.0f;
    for (int i = 0; i < dim; ++i)
    {
        int diff = (int)q_int[i] - (int)d_int[i];
        dist += (float)(diff * diff);
    }
    return dist;
}

__attribute__((target("avx2"))) float Solution::L2_sq16_int_avx2(const int16_t *q_int, int idx) const
{
    const int16_t *d_int = &sq16_data[idx * dim_aligned];
    __m256i sum1 = _mm256_setzero_si256();
    __m256i sum2 = _mm256_setzero_si256();
    int i = 0;
    for (; i + 32 <= dim_aligned; i += 32)
    {
        __m256i v_d1 = _mm256_loadu_si256((const __m256i *)(d_int + i));
        __m256i v_q1 = _mm256_loadu_si256((const __m256i *)(q_int + i));
        __m256i diff1 = _mm256_sub_epi16(v_q1, v_d1);
        sum1 = _mm256_add_epi32(sum1, _mm256_madd_epi16(diff1, diff1));

        __m256i v_d2 = _mm256_loadu_si256((const __m256i *)(d_int + i + 16));
        __m256i v_q2 = _mm256_loadu_si256((const __m256i *)(q_int + i + 16));
        __m256i diff2 = _mm256_sub_epi16(v_q2, v_d2);
        sum2 = _mm256_add_epi32(sum2, _mm256_madd_epi16(diff2, diff2));
    }
    __m256i sum_total = _mm256_add_epi32(sum1, sum2);
    for (; i < dim_aligned; i += 16)
    {
        __m256i v_d = _mm256_loadu_si256((const __m256i *)(d_int + i));
        __m256i v_q = _mm256_loadu_si256((const __m256i *)(q_int + i));
        __m256i diff = _mm256_sub_epi16(v_q, v_d);
        sum_total = _mm256_add_epi32(sum_total, _mm256_madd_epi16(diff, diff));
    }
    __m128i sum_128 = _mm_add_epi32(_mm256_castsi256_si128(sum_total), _mm256_extracti128_si256(sum_total, 1));
    __m128i sum_64 = _mm_add_epi32(sum_128, _mm_shuffle_epi32(sum_128, _MM_SHUFFLE(1, 0, 3, 2)));
    __m128i sum_32 = _mm_add_epi32(sum_64, _mm_shuffle_epi32(sum_64, _MM_SHUFFLE(0, 3, 0, 1)));
    return (float)_mm_cvtsi128_si32(sum_32);
}

template <bool USE_SQ>
inline float Solution::get_dist_template(const std::vector<float> &q_float, const int16_t *q_int, int idx, SearchContext &ctx)
{
    ctx.local_dist_count++;
    if (idx < 0 || idx >= num_points)
        return std::numeric_limits<float>::infinity();

    if (USE_SQ)
    {
        if (sq16_use_avx2)
            return L2_sq16_int_avx2(q_int, idx);
        else
            return L2_sq16_int_scalar(q_int, idx);
    }
    else
    {
        return L2_query_point(q_float, idx);
    }
}

// ==========================================================
// Graph Logic
// ==========================================================

inline int *Solution::get_neighbor_link_ptr(int node_idx, int level) const
{
    size_t offset = node_offsets[node_idx];
    if (level == 0)
        return (int *)(graph_mem + offset);
    return (int *)(graph_mem + offset + size_l0_entry + (size_t)(level - 1) * size_lx_entry);
}

int Solution::compute_dynamic_M(int baseM, int cur, int level)
{
    return (level == 0) ? maxM0 : M;
}

vector<int> Solution::select_neighbors_heuristic(int new_node, const vector<pair<float, int>> &cand, int Mlevel, SearchContext &ctx)
{
    if (cand.empty())
        return {};
    vector<pair<float, int>> sorted = cand;
    sort(sorted.begin(), sorted.end());
    vector<int> result;
    result.reserve(Mlevel);
    for (const auto &entry : sorted)
    {
        if ((int)result.size() >= Mlevel)
            break;
        int c = entry.second;
        if (c == new_node)
            continue;
        bool good = true;
        float dist_c_new = entry.first;
        for (int r : result)
        {
            float d_cr = L2_vec_avx2_fma(point_ptr(r), point_ptr(c), dim);
            if (d_cr < dist_c_new)
            {
                good = false;
                break;
            }
        }
        if (good)
            result.push_back(c);
    }
    return result;
}

void Solution::connect_new_node(int cur, const vector<pair<float, int>> &cand, int level, SearchContext &ctx)
{
    int M_dyn = compute_dynamic_M(maxM0, cur, level);
    vector<int> selected = select_neighbors_heuristic(cur, cand, M_dyn, ctx);

    {
        auto &lock = get_lock(cur);
        lock.lock();
        int *ptr = get_neighbor_link_ptr(cur, level);
        *ptr = (int)selected.size();
        memcpy(ptr + 1, selected.data(), selected.size() * sizeof(int));
        lock.unlock();
    }

    for (int nb : selected)
    {
        auto &lock = get_lock(nb);
        lock.lock();
        int *ptr = get_neighbor_link_ptr(nb, level);
        int cnt = *ptr;
        int *links = ptr + 1;
        bool exists = false;
        for (int i = 0; i < cnt; ++i)
            if (links[i] == cur)
            {
                exists = true;
                break;
            }
        if (exists)
        {
            lock.unlock();
            continue;
        }

        if (cnt < M_dyn)
        {
            links[cnt] = cur;
            *ptr = cnt + 1;
        }
        else
        {
            vector<pair<float, int>> candidates;
            candidates.reserve(cnt + 1);
            for (int i = 0; i < cnt; ++i)
                candidates.emplace_back(L2_vec_avx2_fma(point_ptr(nb), point_ptr(links[i]), dim), links[i]);
            candidates.emplace_back(L2_vec_avx2_fma(point_ptr(nb), point_ptr(cur), dim), cur);
            vector<int> new_links = select_neighbors_heuristic(nb, candidates, M_dyn, ctx);
            *ptr = (int)new_links.size();
            memcpy(links, new_links.data(), new_links.size() * sizeof(int));
        }
        lock.unlock();
    }
}

// ----------------------------------------------------------------------------------
// Core Search Layer (Templated)
// ----------------------------------------------------------------------------------
template <bool USE_SQ, bool USE_FLAT, bool IS_BUILD, bool USE_GAMMA>
vector<pair<float, int>> Solution::search_layer_template(
    const vector<float> &q_float,
    const int16_t *q_int,
    int enter, int ef, int k_search, int level, SearchContext &ctx)
{
    using P = pair<float, int>;
    if (enter < 0 || enter >= num_points)
        return {};

    float d0 = get_dist_template<USE_SQ>(q_float, q_int, enter, ctx);
    ctx.mark_visited(enter);

    // [ABLATION] Force flat_base to nullptr to ensure no usage
    const int *flat_base = nullptr;
    const int16_t *sq_base = sq16_data.data();
    const int PREFETCH_OFFSET = 8;

    if constexpr (USE_GAMMA)
    {
        priority_queue<P> Bk;
        priority_queue<P, vector<P>, greater<P>> C;

        Bk.push({d0, enter});
        C.push({d0, enter});

        while (!C.empty())
        {
            P curr = C.top();
            C.pop();

            if (Bk.size() >= k_search && curr.first > (1.0f + search_gamma) * Bk.top().first)
                break;

            int c_node = curr.second;
            const int *nbrs = nullptr;
            int nbr_cnt = 0;

            if (USE_FLAT)
            {
                // [ABLATION] This path should NOT be taken if template USE_FLAT=false
                // But logically leaving it here for template consistency
                const int *ptr = flat_base + (size_t)c_node * stride_l0;
                nbr_cnt = ptr[0];
                nbrs = ptr + 1;
                if (nbr_cnt > 0 && USE_SQ)
                    _mm_prefetch((const char *)(sq_base + (size_t)nbrs[0] * dim_aligned), _MM_HINT_T0);
            }
            else
            {
                // [ABLATION] This is the path we want to test (Pointer chasing in graph_mem)
                int *ptr = get_neighbor_link_ptr(c_node, level);
                nbr_cnt = *ptr;
                nbrs = ptr + 1;
            }

            for (int i = 0; i < nbr_cnt; ++i)
            {
                if (i + PREFETCH_OFFSET < nbr_cnt)
                {
                    int pf_node = nbrs[i + PREFETCH_OFFSET];
                    if (USE_SQ)
                        _mm_prefetch((const char *)(sq_base + (size_t)pf_node * dim_aligned), _MM_HINT_T0);
                    else
                        _mm_prefetch((const char *)point_ptr(pf_node), _MM_HINT_T0);

                    // [ABLATION] Disabled prefetching of flat buffer
                    if (USE_FLAT)
                        _mm_prefetch((const char *)(flat_base + (size_t)pf_node * stride_l0), _MM_HINT_T0);

                    _mm_prefetch((const char *)(&ctx.visited_tags[pf_node]), _MM_HINT_T0);
                }

                int neighbor = nbrs[i];
                if (ctx.is_visited(neighbor))
                    continue;

                ctx.mark_visited(neighbor);
                float d = get_dist_template<USE_SQ>(q_float, q_int, neighbor, ctx);

                if ((int)Bk.size() < k_search || d < Bk.top().first)
                {
                    Bk.push({d, neighbor});
                    if ((int)Bk.size() > k_search)
                        Bk.pop();
                }

                if ((int)Bk.size() < k_search || d < (1.0f + search_gamma) * Bk.top().first)
                {
                    C.push({d, neighbor});
                }
            }
        }

        vector<P> res;
        res.reserve(Bk.size());
        while (!Bk.empty())
        {
            res.push_back(Bk.top());
            Bk.pop();
        }
        reverse(res.begin(), res.end());
        return res;
    }
    else
    {
        priority_queue<P, vector<P>, greater<P>> C;
        priority_queue<P> W;

        C.push({d0, enter});
        W.push({d0, enter});

        while (!C.empty())
        {
            P curr_1 = C.top();
            C.pop();

            if (curr_1.first > W.top().first && (int)W.size() >= ef)
                break;

            int c_node = curr_1.second;
            const int *nbrs = nullptr;
            int nbr_cnt = 0;

            if (USE_FLAT)
            {
                // Unreachable in ablation
                const int *ptr = flat_base + (size_t)c_node * stride_l0;
                nbr_cnt = ptr[0];
                nbrs = ptr + 1;
            }
            else
            {
                if (IS_BUILD)
                {
                    int *ptr = get_neighbor_link_ptr(c_node, level);
                    nbr_cnt = *ptr;
                    if (ctx.local_nbrs_copy.size() < (size_t)nbr_cnt)
                        ctx.local_nbrs_copy.resize(nbr_cnt);
                    memcpy(ctx.local_nbrs_copy.data(), ptr + 1, nbr_cnt * sizeof(int));
                    nbrs = ctx.local_nbrs_copy.data();
                }
                else
                {
                    int *ptr = get_neighbor_link_ptr(c_node, level);
                    nbr_cnt = *ptr;
                    nbrs = ptr + 1;
                }
            }

            for (int i = 0; i < nbr_cnt; ++i)
            {
                if (i + PREFETCH_OFFSET < nbr_cnt)
                {
                    int pf_node = nbrs[i + PREFETCH_OFFSET];
                    if (USE_SQ)
                        _mm_prefetch((const char *)(sq_base + (size_t)pf_node * dim_aligned), _MM_HINT_T0);
                    else
                        _mm_prefetch((const char *)point_ptr(pf_node), _MM_HINT_T0);

                    if (USE_FLAT)
                        _mm_prefetch((const char *)(flat_base + (size_t)pf_node * stride_l0), _MM_HINT_T0);

                    _mm_prefetch((const char *)(&ctx.visited_tags[pf_node]), _MM_HINT_T0);
                }

                int neighbor = nbrs[i];
                if (ctx.is_visited(neighbor))
                    continue;

                ctx.mark_visited(neighbor);
                float d = get_dist_template<USE_SQ>(q_float, q_int, neighbor, ctx);

                if ((int)W.size() < ef || d < W.top().first)
                {
                    W.push({d, neighbor});
                    if ((int)W.size() > ef)
                        W.pop();
                    C.push({d, neighbor});
                }
            }
        }

        vector<P> res;
        res.reserve(W.size());
        while (!W.empty())
        {
            res.push_back(W.top());
            W.pop();
        }
        reverse(res.begin(), res.end());
        return res;
    }
}

// ==========================================================
// Build & Search Public API
// ==========================================================

void Solution::finalize_graph()
{
    // [ABLATION] Do NOT build flat_l0.
    // This leaves flat_l0 empty and stride_l0 potentially uninitialized (but unused)
    stride_l0 = maxM0 + 1;
    flat_l0.clear();
    // flat_l0.shrink_to_fit(); // Save memory explicitly
}

void Solution::build(int d, const vector<float> &base)
{
    dim = d;
    num_points = (int)base.size() / dim;
    base_data = base;

    if (sq16_enabled)
    {
        compute_sq16_params_global();
        encode_base_with_sq16_global();
    }

    levels.resize(num_points);
    node_locks.clear();
    node_locks = vector<SpinLock>(num_points);

    std::mt19937 rng(100);
    std::uniform_real_distribution<double> unif(0, 1);
    for (int i = 0; i < num_points; ++i)
    {
        int l = 0;
        double r = unif(rng);
        while (r < 0.5 && l < 20)
        {
            r = unif(rng);
            l++;
        }
        levels[i] = l;
    }

    size_l0_entry = sizeof(int) + sizeof(int) * (maxM0 + 1);
    size_lx_entry = sizeof(int) + sizeof(int) * (M + 1);
    node_offsets.resize(num_points);
    size_t total_mem = 0;
    for (int i = 0; i < num_points; ++i)
    {
        node_offsets[i] = total_mem;
        total_mem += size_l0_entry + (size_t)levels[i] * size_lx_entry;
    }
    total_mem = (total_mem + 63) & ~63;

    if (graph_mem)
        aligned_free_wrapper(graph_mem);
    graph_mem = (char *)aligned_alloc_wrapper(total_mem, 64);
    memset(graph_mem, 0, total_mem);

    enter_point_atomic = 0;
    max_level_atomic = levels[0];

    int hw = thread::hardware_concurrency();
    if (hw == 0)
        hw = 4;

    atomic<int> gid(1);
    vector<thread> pool;
    for (int t = 0; t < hw; ++t)
    {
        pool.emplace_back([&]()
                          {
            SearchContext ctx;
            ctx.resize(num_points, dim_aligned);
            vector<float> q(dim);
            
            while(true) {
                int i = gid.fetch_add(1);
                if (i >= num_points) break;
                
                memcpy(q.data(), point_ptr(i), dim*sizeof(float));
                ctx.reset_visited();
                
                int ep = enter_point_atomic.load(memory_order_relaxed);
                int L_max = max_level_atomic.load(memory_order_relaxed);
                int cur_L = levels[i];
                
                for(int l = L_max; l > cur_L; --l) {
                     auto res = search_layer_template<false, false, true, false>(q, nullptr, ep, 1, 0, l, ctx);
                     if (!res.empty()) ep = res[0].second;
                }
                
                for(int l = min(cur_L, L_max); l >= 0; --l) {
                     // Note: Build always uses standard graph (USE_FLAT=false)
                     auto res = search_layer_template<false, false, true, false>(q, nullptr, ep, ef_construction, 0, l, ctx);
                     connect_new_node(i, res, l, ctx);
                     if(!res.empty()) ep = res[0].second;
                }
                
                int cur_g_max = max_level_atomic.load();
                if (cur_L > cur_g_max) {
                    while(cur_L > max_level_atomic.load()) {
                         int old = max_level_atomic.load();
                         if (max_level_atomic.compare_exchange_weak(old, cur_L)) {
                             enter_point_atomic.store(i);
                         }
                    }
                }
            } });
    }
    for (auto &th : pool)
        th.join();

    max_level = max_level_atomic.load();
    enter_point = enter_point_atomic.load();
    finalize_graph();
}

void Solution::search(const vector<float> &query, int *res)
{
    const int K = 10;
    static thread_local SearchContext ctx;

    ctx.resize(num_points, dim_aligned);
    ctx.reset_visited();
    ctx.local_dist_count = 0;

    int16_t *q_int_ptr = nullptr;
    bool use_sq = false;

    if (sq16_enabled && !sq16_data.empty())
    {
        use_sq = true;
        q_int_ptr = ctx.q_int_buffer.data();
        const float offset_target = -16000.0f;
        for (int i = 0; i < dim; ++i)
        {
            float val = query[i];
            float encoded = (val - g_min_val) * g_scale + offset_target;
            if (encoded < -32767.0f)
                encoded = -32767.0f;
            if (encoded > 32767.0f)
                encoded = 32767.0f;
            q_int_ptr[i] = (int16_t)(encoded + 0.5f);
        }
        for (int i = dim; i < dim_aligned; ++i)
            q_int_ptr[i] = 0;
    }

    if (enter_point == -1 || num_points == 0)
    {
        for (int i = 0; i < K; ++i)
            res[i] = -1;
        query_counter++;
        return;
    }

    int ep = enter_point;

    // Top Layers (Always standard graph)
    for (int l = max_level; l >= 1; --l)
    {
        vector<pair<float, int>> cand;
        if (use_sq)
            cand = search_layer_template<true, false, false, false>(query, q_int_ptr, ep, 1, 0, l, ctx);
        else
            cand = search_layer_template<false, false, false, false>(query, q_int_ptr, ep, 1, 0, l, ctx);
        if (!cand.empty())
            ep = cand[0].second;
    }

    // Bottom Layer
    // [ABLATION] Force USE_FLAT = false here
    // Original optimized was: <..., true, false, true>
    // Ablation version is:    <..., false, false, true>
    ctx.reset_visited();
    vector<pair<float, int>> finals;

    if (use_sq)
    {
        finals = search_layer_template<true, false, false, true>(query, q_int_ptr, ep, ef_search, K, 0, ctx);
    }
    else
    {
        finals = search_layer_template<false, false, false, true>(query, q_int_ptr, ep, ef_search, K, 0, ctx);
    }

    int got = min((int)finals.size(), K);
    for (int i = 0; i < K; ++i)
    {
        if (i < got)
            res[i] = finals[i].second;
        else
            res[i] = -1;
    }

    dist_comp_counter += ctx.local_dist_count;
    query_counter++;
#ifdef COUNT_DIST
    g_dist_calc_count += ctx.local_dist_count;
#endif
}

void Solution::clear_distance_counter()
{
    dist_comp_counter = 0;
    query_counter = 0;
}

uint64_t Solution::get_distance_count() const { return dist_comp_counter; }

double Solution::get_avg_distance_count() const
{
    uint64_t q = query_counter.load();
    if (q == 0)
        return 0.0;
    return (double)dist_comp_counter.load() / (double)q;
}

// ==========================================================
// Serialization (Load/Save) - Modified for Ablation
// ==========================================================

void Solution::save_graph(const std::string &filename)
{
    std::ofstream out(filename, std::ios::binary);
    if (!out)
    {
        std::cerr << "Error: Cannot open " << filename << " for writing.\n";
        return;
    }

    uint32_t magic = 0x484E5357;
    out.write((char *)&magic, sizeof(magic));
    out.write((char *)&num_points, sizeof(num_points));
    out.write((char *)&dim, sizeof(dim));
    out.write((char *)&maxM0, sizeof(maxM0));
    out.write((char *)&M, sizeof(M));
    out.write((char *)&ef_construction, sizeof(ef_construction));
    out.write((char *)&enter_point, sizeof(enter_point));
    out.write((char *)&max_level, sizeof(max_level));

    out.write((char *)&sq16_enabled, sizeof(sq16_enabled));
    out.write((char *)&g_min_val, sizeof(g_min_val));
    out.write((char *)&g_scale, sizeof(g_scale));
    out.write((char *)&search_gamma, sizeof(search_gamma));

    out.write((char *)levels.data(), levels.size() * sizeof(int));
    out.write((char *)base_data.data(), base_data.size() * sizeof(float));

    size_t sq_size = sq16_data.size();
    out.write((char *)&sq_size, sizeof(sq_size));
    if (sq_size > 0)
        out.write((char *)sq16_data.data(), sq_size * sizeof(int16_t));

    // [ABLATION] Write dummy size 0 for flat_l0 to maintain file structure compatibility
    // if we wanted to read it back into the original code, but effectively disabling it here.
    size_t flat_size = 0;
    out.write((char *)&flat_size, sizeof(flat_size));
    // No data written

    size_t total_mem = 0;
    if (num_points > 0)
    {
        total_mem = node_offsets[num_points - 1] + size_l0_entry + (size_t)levels[num_points - 1] * size_lx_entry;
        total_mem = (total_mem + 63) & ~63;
    }
    out.write((char *)&total_mem, sizeof(total_mem));
    if (total_mem > 0 && graph_mem)
        out.write(graph_mem, total_mem);

    out.close();
}

bool Solution::load_graph(const std::string &filename)
{
    std::ifstream in(filename, std::ios::binary);
    if (!in)
        return false;

    uint32_t magic;
    in.read((char *)&magic, sizeof(magic));
    if (magic != 0x484E5357)
        return false;

    in.read((char *)&num_points, sizeof(num_points));
    in.read((char *)&dim, sizeof(dim));
    in.read((char *)&maxM0, sizeof(maxM0));
    in.read((char *)&M, sizeof(M));
    in.read((char *)&ef_construction, sizeof(ef_construction));
    in.read((char *)&enter_point, sizeof(enter_point));
    in.read((char *)&max_level, sizeof(max_level));

    in.read((char *)&sq16_enabled, sizeof(sq16_enabled));
    in.read((char *)&g_min_val, sizeof(g_min_val));
    in.read((char *)&g_scale, sizeof(g_scale));
    in.read((char *)&search_gamma, sizeof(search_gamma));

    m_log_factor = 1.0f / log(1.0f * M);
    dim_aligned = (dim + 15) & ~15;
    size_l0_entry = sizeof(int) + sizeof(int) * (maxM0 + 1);
    size_lx_entry = sizeof(int) + sizeof(int) * (M + 1);

    levels.resize(num_points);
    in.read((char *)levels.data(), num_points * sizeof(int));

    node_offsets.resize(num_points);
    size_t computed_mem = 0;
    for (int i = 0; i < num_points; ++i)
    {
        node_offsets[i] = computed_mem;
        computed_mem += size_l0_entry + (size_t)levels[i] * size_lx_entry;
    }
    computed_mem = (computed_mem + 63) & ~63;

    base_data.resize(num_points * dim);
    in.read((char *)base_data.data(), base_data.size() * sizeof(float));

    size_t sq_size;
    in.read((char *)&sq_size, sizeof(sq_size));
    sq16_data.resize(sq_size);
    if (sq_size > 0)
        in.read((char *)sq16_data.data(), sq_size * sizeof(int16_t));

    // [ABLATION] Read flat_size but ignore the data if it exists (for compatibility)
    size_t flat_size;
    in.read((char *)&flat_size, sizeof(flat_size));
    if (flat_size > 0)
    {
        // Skip over the data in the file
        in.seekg(flat_size * sizeof(int), std::ios::cur);
    }
    flat_l0.clear(); // Ensure it's empty
    stride_l0 = maxM0 + 1;

    size_t total_mem;
    in.read((char *)&total_mem, sizeof(total_mem));

    if (graph_mem)
        aligned_free_wrapper(graph_mem);
    if (total_mem > 0)
    {
        graph_mem = (char *)aligned_alloc_wrapper(total_mem, 64);
        in.read(graph_mem, total_mem);
    }
    else
        graph_mem = nullptr;

    node_locks.clear();
    node_locks = std::vector<SpinLock>(num_points);
    max_level_atomic.store(max_level);
    enter_point_atomic.store(enter_point);

    return true;
}