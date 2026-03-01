#include "MySolution.h"
#include <iostream>
#include <vector>
#include <fstream>
#include <chrono>
#include <string>
#include <random>
#include <cmath>

// 模拟项目中的read_data函数 - 修改为只读取前1000个数据
void read_data(int cas, std::vector<float> &base_, int &dimension_, int &pointEnum)
{
    std::string filePath;

    if (cas == 0)
    {
        filePath = "glove_base.txt";
        pointEnum = 1183514; // 测试数据量
        dimension_ = 100;
    }
    else if (cas == 1)
    {
        filePath = "sift_base.txt";
        pointEnum = 100000;
        dimension_ = 128;
    }
    else
    {
        std::cerr << "Invalid dataset case: " << cas << std::endl;
        return;
    }

    base_.resize(dimension_ * pointEnum);
    std::ifstream fin(filePath);
    if (!fin.is_open())
    {
        std::cerr << "Error opening " << filePath << std::endl;
        return;
    }
    std::ios::sync_with_stdio(false); // 禁用同步
    fin.tie(NULL);                    // 解绑流

    for (int i = 0; i < pointEnum; ++i)
    {
        for (int j = 0; j < dimension_; ++j)
        {
            int index = i * dimension_ + j;
            if (!(fin >> base_[index]))
            {
                std::cerr << "Error reading data at point " << i << ", dimension " << j << std::endl;
                fin.close();
                return;
            }
        }
    }
    fin.close();

    std::cout << "Loaded dataset case " << cas << ": " << filePath << std::endl;
    std::cout << "Vectors: " << pointEnum << ", Dimension: " << dimension_ << std::endl;
}

// 计算Top-K准确率
float compute_recall(const std::vector<int> &bf_res, const std::vector<int> &hnsw_res, int K)
{
    int hit = 0;
    for (int i = 0; i < K; i++)
    {
        for (int j = 0; j < K; j++)
        {
            if (bf_res[i] == hnsw_res[j])
            {
                hit++;
                break;
            }
        }
    }
    return float(hit) / K;
}

// 检查文件是否存在
bool check_file_exists(const std::string &filename)
{
    std::ifstream file(filename);
    return file.good();
}

// 测试指定数据集
void test_dataset_case(int cas, const std::string &dataset_name)
{
    std::cout << "\n=== Testing Dataset: " << dataset_name << " (Case " << cas << ") ===" << std::endl;

    // 1. 读取 Base 数据
    std::vector<float> base_;
    int dimension_, pointEnum;
    read_data(cas, base_, dimension_, pointEnum);

    if (base_.empty())
    {
        std::cout << "Data loading failed, skipping test" << std::endl;
        return;
    }

    // 2. 构建 HNSW 索引
    std::cout << "\n--- HNSW Search Test ---" << std::endl;
    Solution hnsw_solution;
    auto hnsw_build_start = std::chrono::high_resolution_clock::now();

    hnsw_solution.build(dimension_, base_);

    auto hnsw_build_end = std::chrono::high_resolution_clock::now();
    double hnsw_build_time = std::chrono::duration<double>(hnsw_build_end - hnsw_build_start).count();
    std::cout << "[HNSW] Index build time: " << hnsw_build_time << " seconds" << std::endl;

    // 3. 读取 Query 数据
    std::string filePath1 = "query.txt";
    std::ifstream fin1(filePath1);
    if (!fin1.is_open())
    {
        std::cerr << "Error opening " << filePath1 << std::endl;
        return;
    }
    std::ios::sync_with_stdio(false);
    fin1.tie(nullptr);

    // 4. 读取 Truth 数据 (用于计算 Recall)
    std::string filePath2 = "truth.txt";
    std::ifstream fin2(filePath2);
    if (!fin2.is_open())
    {
        std::cerr << "Error opening " << filePath2 << std::endl;
        return;
    }
    std::ios::sync_with_stdio(false);
    fin2.tie(nullptr);

    std::cout << "\n--- HNSW Search Test on query.txt ---" << std::endl;

    const int dimension = 100; // 注意：这里硬编码了100，如果是SIFT需要改为128或者动态获取
    const int query_count = 10000;

    std::vector<std::vector<float>> queries;
    queries.resize(query_count, std::vector<float>(dimension));
    for (int i = 0; i < query_count; ++i)
    {
        for (int j = 0; j < dimension; ++j)
        {
            fin1 >> queries[i][j];
        }
    }

    std::vector<std::vector<int>> groundtruth;
    int groundtruth_size = 100;
    int K = 10;
    groundtruth.resize(query_count, std::vector<int>(K));
    for (int i = 0; i < query_count; ++i)
    {
        for (int j = 0; j < groundtruth_size; ++j)
        {
            int val;
            fin2 >> val;
            if (j < K)
            {
                groundtruth[i][j] = val;
            }
        }
    }
    fin1.close();
    fin2.close();

    std::vector<std::vector<int>> hnsw_res(query_count, std::vector<int>(K));
    double hnsw_total_time_sift = 0.0;

    // [修改点 1]：在开始批量查询前，重置计数器
    hnsw_solution.clear_distance_counter();

    for (int i = 0; i < query_count; i++)
    {
        auto start = std::chrono::high_resolution_clock::now();

        // [修改点 2]：search 函数内部会自动统计次数，不需要外部累加
        hnsw_solution.search(queries[i], hnsw_res[i].data());

        auto end = std::chrono::high_resolution_clock::now();
        hnsw_total_time_sift += std::chrono::duration<double>(end - start).count();

        if (i % 1000 == 0)
        {
            std::cout << "HNSW query " << i << " completed" << std::endl;
        }
    }

    double hnsw_avg_time2 = hnsw_total_time_sift / query_count;
    std::cout << "[HNSW] Average search time on query.txt: " << hnsw_avg_time2 << " seconds, QPS: " << 1.0 / hnsw_avg_time2 << std::endl;

    // [修改点 3]：调用接口获取平均距离计算次数并打印
    std::cout << "[HNSW] Average distance computations per query: " << hnsw_solution.get_avg_distance_count() << std::endl;

    // 计算 Recall
    double recall_sum2 = 0.0;
    for (int i = 0; i < query_count; ++i)
    {
        float recall = compute_recall(groundtruth[i], hnsw_res[i], K);
        recall_sum2 += recall;
    }
    double avg_recall2 = recall_sum2 / query_count;
    std::cout << "Top-" << K << " Average recall (HNSW vs Ground Truth): " << avg_recall2 << std::endl;
}

int main()
{
    std::cout << "Vector Search Algorithm Performance Comparison Test" << std::endl;
    std::cout << "===================================================" << std::endl;

    bool has_real_data = false;

    // Test GLOVE dataset
    if (check_file_exists("glove_base.txt"))
    {
        test_dataset_case(0, "GLOVE");
        has_real_data = true;
    }
    else
    {
        std::cout << "\nGLOVE dataset file not found: glove_base.txt" << std::endl;
    }

    // if (check_file_exists("glove_base.txt"))
    // {
    //     test_dataset_case(0, "GLOVE");
    //     has_real_data = true;
    // }
    // else
    // {
    //     std::cout << "\nGLOVE dataset file not found: glove_base.txt" << std::endl;
    // }

    if (!has_real_data)
    {
        std::cout << "\nNo valid dataset available for testing!" << std::endl;
        std::cout << "Please ensure glove_base.txt file exists in current directory" << std::endl;
    }

    std::cout << "\nTest completed!" << std::endl;
    return 0;
}