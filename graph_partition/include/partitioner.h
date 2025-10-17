//
// Created by Zilliz_wmz on 2022/6/6.
//
#pragma once

#include <omp.h>
#include <oneapi/tbb/concurrent_queue.h>
#include <algorithm>
#include <atomic>
#include <bitset>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <ostream>
#include <queue>
#include <random>
#include <set>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#include "filesystem"
#include "freq_relayout.h"
#include "../../include/pq_table.h"
#include "../../include/distance.h"
#include "../../include/pq_flash_index_utils.h"
#include "../../include/utils.h"
#include <type_traits>

#ifndef INF
#define INF 0xffffffff
#endif  // INF

#ifndef READ_U64
#define READ_U64(stream, val) stream.read((char *)&val, sizeof(_u64))
#endif  // !READ_U64
#ifndef ROUND_UP
#define ROUND_UP(X, Y) (((uint64_t)(X) / (Y)) + ((uint64_t)(X) % (Y) != 0)) * (Y)
#endif  // !ROUND_UP
#ifndef SECTOR_LEN
#define SECTOR_LEN (_u64)4096
#endif  // !SECTOR_LEN

namespace GP {

using concurrent_queue = oneapi::tbb::concurrent_bounded_queue<unsigned>;
namespace fs = std::filesystem;
using _u64 = unsigned long int;
using _u32 = unsigned int;
using VecT = uint8_t;

inline size_t get_file_size(const std::string &fname) {
  std::ifstream reader(fname, std::ios::binary | std::ios::ate);
  if (!reader.fail() && reader.is_open()) {
    size_t end_pos = reader.tellg();
    reader.close();
    return end_pos;
  } else {
    std::cout << "Could not open file: " << fname << std::endl;
    return 0;
  }
}

// DiskANN changes how the meta is stored in the first sector of
// the _disk.index file after commit id 8bb74ff637cb2a77c99b71368ade68c62b7ca8e0
// (exclusive) It returns <is_new_version, vector of metas uint64_ts>
std::pair<bool, std::vector<_u64>> get_disk_index_meta(const std::string &path) {
  std::ifstream fin(path, std::ios::binary);

  int meta_n, meta_dim;
  const int expected_new_meta_n = 9;
  const int expected_new_meta_n_with_reorder_data = 12;
  const int old_meta_n = 11;
  bool is_new_version = true;
  std::vector<_u64> metas;

  fin.read((char *)(&meta_n), sizeof(int));
  fin.read((char *)(&meta_dim), sizeof(int));

  if (meta_n == expected_new_meta_n || meta_n == expected_new_meta_n_with_reorder_data) {
    metas.resize(meta_n);
    fin.read((char *)(metas.data()), sizeof(_u64) * meta_n);
  } else {
    is_new_version = false;
    metas.resize(old_meta_n);
    fin.seekg(0, std::ios::beg);
    fin.read((char *)(metas.data()), sizeof(_u64) * old_meta_n);
  }
  fin.close();
  return {is_new_version, metas};
}

template<typename T>
class graph_partitioner {
 public:
  graph_partitioner(const char *indexName, const char *data_type = "uint8",
                    bool load_disk = true, unsigned BS = 1, bool visual = false,
                    std::string freq_file = std::string(""), unsigned cut = INF,
                    bool build_knn_graph = false) {
    _visual = visual;
    _build_knn_graph = build_knn_graph;
    std::srand(static_cast<unsigned int>(std::time(nullptr)));

    // check file size
    size_t actual_size = get_file_size(indexName);
    size_t expected_size;
    auto meta_pair = get_disk_index_meta(indexName);
    if (meta_pair.first) {
      expected_size = meta_pair.second.back();
    } else {
      expected_size = meta_pair.second.front();
    }
    if (actual_size != expected_size) {
      std::cout << "index file not match!" << std::endl;
      exit(-1);
    }

    _rd = new std::random_device();
    _gen = new std::mt19937((*_rd)());
    _dis = new std::uniform_real_distribution<>(0, 1);
    if (load_disk) {
      load_disk_index<T>(indexName, BS);
    } else {
      load_vamana(indexName);
    }
    cursize = _nd / 1000;

    if (!freq_file.empty()) {
      if (!fs::exists(freq_file)) {
        std::cout << "No such freq file!" << std::endl;
        exit(-1);
      }
      read_freq(_freq_list, _freq_nei_list, freq_file);
      if (_freq_list.size() != _nd) {
        std::cout << "freq not match, freq file has node " << _freq_list.size() << " but index file nodes is " << _nd
                  << std::endl;
        exit(-1);
      }
      relayout_adj(_freq_nei_list, full_graph);
    }
    
    // Build KNN graph if requested (before copying to direct_graph)
    if (_build_knn_graph) {
      std::cout << "Building KNN graph with k=" << C << "..." << std::endl;
      this->build_knn_graph();
    }
    
    // copy to direct_graph
    direct_graph.clear();
    direct_graph.resize(full_graph.size());
#pragma omp parallel for
    for (unsigned i = 0; i < _nd; i++) {
      direct_graph[i].assign(full_graph[i].begin(), full_graph[i].end());
    }
    // cut graph
    if(cut !=INF){
      std::cout << "direct graph will be cut, it degree become "<<cut << std::endl;
    }
#pragma omp parallel for
    for (unsigned i = 0; i < _nd; i++) {
      if (cut < direct_graph[i].size()) {
        direct_graph[i].resize(cut);
      }
    }
    // reverse graph
    std::vector<std::mutex> ms(_nd);
    reverse_graph.resize(_nd);
#pragma omp parallel for shared(reverse_graph, direct_graph)
    for (unsigned i = 0; i < _nd; i++) {
      for (unsigned j = 0; j < direct_graph[i].size(); j++) {
        std::lock_guard<std::mutex> lock(ms[direct_graph[i][j]]);
        reverse_graph[direct_graph[i][j]].emplace_back(i);
      }
    }
    std::cout << "reverse graph done." << std::endl;
    for (unsigned i = 0; i < _partition_number; i++) {
      pmutex.push_back(std::make_unique<std::mutex>());
    }
    //   undirect_graph.resize(_nd);
    //     E = 0;
    // #pragma omp parallel for schedule(dynamic, 100)
    //     for (unsigned i = 0; i < _nd; i++) {
    //       std::set<unsigned> ne;
    //       for (auto n : direct_graph[i]) {
    //         ne.insert(n);
    //       }
    //       for (auto n : reverse_graph[i]) {
    //         ne.insert(n);
    //       }
    //       for (auto n : ne) {
    //         undirect_graph[i].push_back(n);
    //       }
    // #pragma omp atomic
    //       E += undirect_graph[i].size();
    //     }
  }

  void cout_step() {
    if (!_visual) {
      return;
    }

#pragma omp atomic
    cur++;
    if ((cur + 0) % cursize == 0) {
      std::cout << (double)(cur + 0) / _nd * 100 << "%    \r";
      std::cout.flush();
    }
  }

  void build_knn_graph() {
    if (!_build_knn_graph) return;
    
    std::cout << "Initializing KNN graph construction using quantized coordinates..." << std::endl;
    
    std::cout << "Computing KNN for " << _nd << " nodes using Best First Search..." << std::endl;
    
    // Phase 1: Parallel computation of KNN results (thread-safe read-only access)
    std::vector<std::vector<unsigned>> knn_results(_nd);
    
    #pragma omp parallel for schedule(dynamic, 100)
    for (_u64 i = 0; i < _nd; ++i) {
      if (i % 100000 == 0) {
        #pragma omp critical
        {
          std::cout << "Processing node " << i << "/" << _nd << std::endl;
        }
      }
      
      // Best First Search to find k nearest neighbors (read-only access to full_graph)
      knn_results[i] = best_first_search_knn(i, C/2);
    }
    
    // Phase 2: Sequential update of graph structure (thread-safe)
    std::cout << "Updating graph structure with KNN results..." << std::endl;
    for (_u64 i = 0; i < _nd; ++i) {
      full_graph[i].clear();
      full_graph[i].assign(knn_results[i].begin(), knn_results[i].end());
    }
    
    std::cout << "KNN graph construction completed. Average degree: " 
              << (double)std::accumulate(full_graph.begin(), full_graph.end(), 0, 
                  [](_u64 sum, const std::vector<unsigned>& neighbors) {
                    return sum + neighbors.size();
                  }) / _nd << std::endl;
  }
  
  /**
   * Best First Search to find k nearest neighbors using quantized coordinates
   */
  std::vector<unsigned> best_first_search_knn(_u64 query_id, _u64 k) {
    // Candidate structure for priority queue
    struct Candidate {
      _u64 node_id;
      float distance;
      
      Candidate(_u64 id, float dist) : node_id(id), distance(dist) {}
      
      // Priority queue is max-heap, so we reverse the comparison for min-heap
      bool operator>(const Candidate& other) const {
        return distance > other.distance;
      }
      // Define operator< so default priority_queue becomes a max-heap by distance
      bool operator<(const Candidate& other) const {
        return distance < other.distance;
      }
    };
    
    // Min-heap over candidates to expand (smallest distance first)
    std::priority_queue<Candidate, std::vector<Candidate>, std::greater<Candidate>> pq;
    // Max-heap to maintain current k nearest neighbors (largest distance on top)
    std::priority_queue<Candidate> knn_heap;
    std::set<_u64> visited;
    std::vector<unsigned> knn_neighbors;
    knn_neighbors.reserve(k);
    
    // If using disk PQ, inflate query once to a temporary float vector for reuse
    std::unique_ptr<float[]> query_inflated;
    if (_use_disk_pq && _n_chunks > 0) {
      query_inflated.reset(new float[_dim]());
      const _u8* query_code = _pq_codes.data() + query_id * _n_chunks;
      _pq_table.inflate_vector(const_cast<_u8*>(query_code), query_inflated.get());
    }
    
    // Initialize with direct neighbors
    for (auto neighbor : full_graph[query_id]) {
      if (neighbor != query_id) {
        float dist;
        if (_use_disk_pq && query_inflated) {
          const _u8* nb_code = _pq_codes.data() + (_u64)neighbor * _n_chunks;
          dist = _pq_table.l2_distance(query_inflated.get(), const_cast<_u8*>(nb_code));
        } else {
          dist = compute_quantized_distance(query_id, neighbor);
        }
        pq.push(Candidate(neighbor, dist));
      }
    }
    
    // Best First Search
    const _u64 max_hop = 200;  // Limit search depth
    _u64 hop = 0;
    while (!pq.empty() && hop < max_hop) {
      Candidate current = pq.top();
      pq.pop();
      
      if (visited.count(current.node_id)) continue;
      visited.insert(current.node_id);
      hop++;
      
      // Maintain top-k nearest neighbors using a max-heap by distance
      if (knn_heap.size() < k) {
        knn_heap.push(current);
      } else if (current.distance < knn_heap.top().distance) {
        knn_heap.pop();
        knn_heap.push(current);
      }
      
      // Expand: add neighbors of current node
      for (auto next_neighbor : full_graph[current.node_id]) {
        if (next_neighbor != query_id && !visited.count(next_neighbor)) {
          float dist;
          if (_use_disk_pq && query_inflated) {
            const _u8* nb_code = _pq_codes.data() + (_u64)next_neighbor * _n_chunks;
            dist = _pq_table.l2_distance(query_inflated.get(), const_cast<_u8*>(nb_code));
          } else {
            dist = compute_quantized_distance(query_id, next_neighbor);
          }
          pq.push(Candidate(next_neighbor, dist));
        }
      }
    }
    
    // Extract results from max-heap into output vector
    while (!knn_heap.empty()) {
      knn_neighbors.push_back(knn_heap.top().node_id);
      knn_heap.pop();
    }
    
    return knn_neighbors;
  }
  
  
  /**
   * Compute quantized distance between two nodes
   */
  float compute_quantized_distance(_u64 node1, _u64 node2) {
    // strictly use disk PQ path; otherwise return large value
    if (_use_disk_pq && _n_chunks > 0 && !_pq_codes.empty()) {
      std::unique_ptr<float[]> node1_fp(new float[_dim]());
      const _u8* code1 = _pq_codes.data() + node1 * _n_chunks;
      _pq_table.inflate_vector(const_cast<_u8*>(code1), node1_fp.get());
      const _u8* code2 = _pq_codes.data() + node2 * _n_chunks;
      return _pq_table.l2_distance(node1_fp.get(), const_cast<_u8*>(code2));
    }
    return std::numeric_limits<float>::max();
  }
  /**
   * load vamana graph index from disk
   * @param filename
   */
  void load_vamana(const char *filename, bool sample = false) {
    std::cout << "Reading index file: " << filename << "... " << std::flush;
    std::ifstream in;
    in.exceptions(std::ifstream::failbit | std::ifstream::badbit);
    try {
      in.open(filename, std::ios::binary);
      size_t expected_file_size;
      in.read((char *)&expected_file_size, sizeof(uint64_t));
      in.read((char *)&_width, sizeof(unsigned));
      in.read((char *)&_ep, sizeof(unsigned));
      std::cout << "Loading vamana index " << filename << "..." << std::flush;

      size_t cc = 0;
      unsigned nodes = 0;
      while (in.peek() != EOF) {
        unsigned k;
        in.read((char *)&k, sizeof(unsigned));
        cc += k;
        ++nodes;
        std::vector<unsigned> tmp(k);
        in.read((char *)tmp.data(), k * sizeof(unsigned));
        direct_graph.emplace_back(tmp);
        if (nodes % 10000000 == 0) std::cout << "." << std::flush;
      }
      if (direct_graph.size() != _nd) {
        std::cout << "graph vertex size error!\n";
        exit(-1);
      }
      if (sample) {
        std::cout << "cut adj" << std::endl;
        for (unsigned i = 0; i < _nd; i++) {
          std::vector<unsigned> tmp;
          tmp.reserve(10);
          for (unsigned j = 0; j < 20 && j < direct_graph[i].size(); j++) {
            tmp.push_back(direct_graph[i][j]);
          }
          direct_graph[i].clear();
          direct_graph[i].assign(tmp.begin(), tmp.end());
        }
      }
      C = 12;
      _partition_number = ROUND_UP(_nd, C) / C;
      reverse_graph.resize(_nd);
      std::vector<std::mutex> ms(_nd);
#pragma omp parallel for shared(reverse_graph, direct_graph)
      for (unsigned i = 0; i < _nd; i++) {
        for (unsigned j = 0; j < direct_graph[i].size(); j++) {
          std::lock_guard<std::mutex> lock(ms[direct_graph[i][j]]);
          reverse_graph[direct_graph[i][j]].emplace_back(i);
        }
      }
      std::cout << "done. Index has " << nodes << " nodes and " << cc << " out-edges" << std::endl;
      for (unsigned i = 0; i < _partition_number; i++) {
        pmutex.push_back(std::make_unique<std::mutex>());
      }
    } catch (std::system_error &e) {
      exit(-1);
    }
  }

  template<typename U>
  void load_disk_index(const char *index_name, int BS = 1) {
    std::cout << "loading disk index file: " << index_name << "... " << std::flush;
    std::ifstream in;
    in.exceptions(std::ifstream::failbit | std::ifstream::badbit);

    try {
      _u64 expected_npts;
      auto meta_pair = get_disk_index_meta(index_name);

      if (meta_pair.first) {
        // new version
        expected_npts = meta_pair.second.front();
      } else {
        expected_npts = meta_pair.second[1];
      }
      _nd = expected_npts;
      _dim = meta_pair.second[1];

      _max_node_len = meta_pair.second[3];
      C = meta_pair.second[4];

      _partition_number = ROUND_UP(_nd, C) / C;

      in.open(index_name, std::ios::binary);
      in.seekg(SECTOR_LEN, std::ios::beg);
      
      full_graph.resize(_nd);
      if (_build_knn_graph) {
        // Load PQ metadata from _pq_pivots.bin and compressed codes from _pq_compressed.bin
        // Build paths relative to the directory of the index file, e.g. <dir>/_pq_pivots.bin
        fs::path index_path(index_name);
        std::string index_dir = index_path.parent_path().string();
        if (index_dir.empty()) index_dir = ".";
        std::string pq_pivots = (fs::path(index_dir) / "_pq_pivots.bin").string();
        std::string pq_codes  = (fs::path(index_dir) / "_pq_compressed.bin").string();

        // load pq pivots (codebooks) and infer n_chunks
        try {
          _pq_table.load_pq_centroid_bin(pq_pivots.c_str(), 0);
          _n_chunks = _pq_table.get_num_chunks();
        } catch (...) {
          std::cout << "Failed to load PQ pivots from: " << pq_pivots << std::endl;
          exit(-1);
        }
        if (_n_chunks == 0) {
          std::cout << "Invalid PQ n_chunks (0) from pivots." << std::endl;
          exit(-1);
        }
        // load compressed pq codes (nd * n_chunks, uint8) via DiskANN loader
        {
          _u8* codes_ptr = nullptr;
          _u64 nr = 0, nc = 0;
          try {
            diskann::load_bin<_u8>(pq_codes, codes_ptr, nr, nc);
          } catch (std::exception &e) {
            std::cout << "Failed to load PQ codes from: " << pq_codes << " error: " << e.what() << std::endl;
            exit(-1);
          }
          if (nr != _nd || nc != _n_chunks) {
            std::cout << "PQ codes shape mismatch. expect (nd, n_chunks)=(" << _nd << "," << _n_chunks
                      << ") but got (" << nr << "," << nc << ")" << std::endl;
            delete[] codes_ptr;
            exit(-1);
          }
          _pq_codes.resize((_u64)nr * nc);
          std::memcpy(_pq_codes.data(), codes_ptr, (_u64)nr * nc * sizeof(_u8));
          delete[] codes_ptr;
        }
        _use_disk_pq = true;
        std::cout << "Loaded PQ: chunks=" << _n_chunks << ", codes=" << _pq_codes.size() << std::endl;
      }
      _u64 des = 0;
      
      // 流式读取每个分区，避免一次性分配大量内存
      for (unsigned i = 0; i < _partition_number; i++) {
        std::unique_ptr<char[]> sector_buf = std::make_unique<char[]>(SECTOR_LEN);
        in.read(sector_buf.get(), SECTOR_LEN);
        
        for (unsigned j = 0; j < C && i * C + j < _nd; j++) {
          std::unique_ptr<char[]> node_buf = std::make_unique<char[]>(_max_node_len);
          memcpy(node_buf.get(), sector_buf.get() + j * _max_node_len, _max_node_len);
          unsigned &nnbr = *(unsigned *)(node_buf.get() + _dim * sizeof(U));
          unsigned *nhood_buf = (unsigned *)(node_buf.get() + (_dim * sizeof(U)) + sizeof(unsigned));
          std::vector<unsigned> tmp(nnbr);
          des += nnbr;
          memcpy((char *)tmp.data(), nhood_buf, nnbr * sizeof(unsigned));
          full_graph[i * C + j].assign(tmp.begin(), tmp.end());
          
          // no longer synthesize PQ data here; codes already loaded from disk
        }
      }
      in.close();
      std::cout << "avg degree: " << (double)des / _nd << std::endl;
      C = (SECTOR_LEN * BS) / _max_node_len;
      _partition_number = ROUND_UP(_nd, C) / C;
      std::cout << "_nd: " << _nd << " _dim:" << _dim << " C:" << C << " pn:" << _partition_number << std::endl;
      std::cout << "load index over." << std::endl;
    } catch (std::system_error &e) {
      std::cout << "open file " << index_name << " error!" << std::endl;
      exit(-1);
    }
  }

  /**
   * save the partition result
   * @tparam T
   * @param filename
   * @param partition
   */
  void save_partition(const char *filename) {
    // re_id2pid();
    std::ofstream writer(filename, std::ios::binary | std::ios::out);
    std::cout << "writing bin: " << filename << std::endl;
    writer.write((char *)&C, sizeof(_u64));
    writer.write((char *)&_partition_number, sizeof(_u64));
    writer.write((char *)&_nd, sizeof(_u64));
    std::cout << "_partition_num: " << _partition_number << " C: " << C << " _nd: " << _nd << std::endl;
    for (unsigned i = 0; i < _partition_number; i++) {
      auto p = _partition[i];
      unsigned s = p.size();
      writer.write((char *)&s, sizeof(unsigned));
      writer.write((char *)p.data(), sizeof(unsigned) * s);
    }
    std::vector<unsigned> id2pidv(_nd);
    for (auto n : id2pid) {
      id2pidv[n.first] = n.second;
    }
    writer.write((char *)id2pidv.data(), sizeof(unsigned) * _nd);
  }

  /**
   * load partition from disk
   * @param filename
   */
  void load_partition(const char *filename) {
    std::ifstream reader(filename, std::ios::binary);
    reader.read((char *)&C, sizeof(_u64));
    reader.read((char *)&_partition_number, sizeof(_u64));
    reader.read((char *)&_nd, sizeof(_u64));
    std::cout << "load partition _partition_num: " << _partition_number << ", C: " << C << std::endl;
    _partition.clear();
    auto tmp = new unsigned[C];
    for (unsigned i = 0; i < _partition_number; i++) {
      unsigned c;
      reader.read((char *)&c, sizeof(unsigned));
      reader.read((char *)tmp, c * sizeof(unsigned));
      std::vector<unsigned> tt;
      tt.reserve(C);
      for (unsigned j = 0; j < c; j++) {
        tt.push_back(*(tmp + j));
      }
      _partition.push_back(tt);
    }
    delete[] tmp;
    re_id2pid();
  }
  void re_id2pid() {
    id2pid.clear();
    for (unsigned i = 0; i < _partition_number; i++) {
      for (unsigned j = 0; j < _partition[i].size(); j++) {
        id2pid[_partition[i][j]] = i;
      }
    }
  }
  /**
   * count the id overlap according to the graph partitioning
   */
  void partition_statistic() {
    std::vector<unsigned> overlap(_nd, 0);
    std::vector<unsigned> blk_neighbor_overlap(_partition_number, 0);
    double overlap_ratio = 0;

#pragma omp parallel for schedule(dynamic, 100) reduction(+ : overlap_ratio)
    for (size_t i = 0; i < _partition_number; i++) {
      std::unordered_set<unsigned> neighbors;
      unsigned blk_neighbor_num = 0;
      for (size_t j = 0; j < _partition[i].size(); j++) {
        blk_neighbor_num += full_graph[_partition[i][j]].size();
        std::unordered_set<unsigned> ne;
        for (unsigned &x : full_graph[_partition[i][j]]) {
          neighbors.insert(x);
          ne.insert(x);
        }
        blk_neighbor_overlap[i] = blk_neighbor_num - neighbors.size();
        for (size_t z = 0; z < _partition[i].size(); z++) {
          if (_partition[i][j] == _partition[i][z]) continue;
          if (ne.find(_partition[i][z]) != ne.end()) {
            overlap[_partition[i][j]]++;
          }
        }
        overlap_ratio +=
            (_partition[i].size() == 1 ? 0 : (1.0 * overlap[_partition[i][j]] / (_partition[i].size() - 1)));
      }
    }
    unsigned max_overlaps = 0;
    unsigned min_overlaps = std::numeric_limits<unsigned>::max();
    double ave_overlap_ratio = 0;
    std::map<unsigned, unsigned> overlap_count;
    for (size_t i = 0; i < _nd; i++) {
      if (overlap_count.count(overlap[i])) {
        overlap_count[overlap[i]]++;
      } else {
        overlap_count[overlap[i]] = 1;
      }
      if (overlap[i] > max_overlaps) max_overlaps = overlap[i];
      if (overlap[i] < min_overlaps) min_overlaps = overlap[i];
    }
    ave_overlap_ratio = overlap_ratio / (double)_nd;
    for (auto &it : overlap_count) {
      std::cout << "each id, overlap number " << it.first << ", count: " << it.second << std::endl;
    }
    std::cout << "each id, max overlaps: " << max_overlaps << std::endl;
    std::cout << "each id, min overlaps: " << min_overlaps << std::endl;
    std::cout << "each id, average overlap ratio: " << ave_overlap_ratio << std::endl;
  }

  unsigned select_partition(unsigned i) {
#pragma omp atomic
    select_nums++;

    float maxn = 0.0;
    unsigned res = INF;
    std::unordered_map<unsigned, unsigned> pcount;
    unsigned tpid = 0;
    for (auto n : direct_graph[i]) {
      unsigned pid = id2pid[n];
      if (pid == INF) continue;
      pcount[pid] = pcount[pid] + 1;
      if (tpid < pid) {
        tpid = pid;
      }
    }
    for (auto n : reverse_graph[i]) {
      unsigned pid = id2pid[n];
      if (pid == INF) continue;
      pcount[pid] = pcount[pid] + 1;
      if (tpid < pid) {
        tpid = pid;
      }
    }
    for (auto c : pcount) {
      unsigned pid = c.first;
      float cnt = c.second;
      std::lock_guard<std::mutex> lock(*pmutex[pid]);
      double s = _partition[pid].size();
      cnt *= (1 - s / C);
      if (cnt > maxn && _partition[pid].size() < C) {
        res = pid;
        maxn = cnt;
      }
    }
    pcount.clear();
    if (res == INF) {
#pragma omp atomic
      select_free++;
      res = getUnfilled();
    }
    return res;
  }

  unsigned getUnfilled() {
#pragma omp atomic
    getUnfilled_nums++;
    unsigned res;
    do {
      free_q.pop(res);
    } while (_partition[res].size() == C);
    return res;
  }

  // graph partition
  void graph_partition(const char *filename, int k, int lock_nums = 0) {
    for (unsigned i = 0; i < _nd; i++) {
      id2pid[i] = INF;
    }
    _partition.clear();
    _partition.resize(_partition_number);
    std::unordered_set<unsigned> vis;
    std::vector<unsigned> init_stream;
    init_stream.reserve(_nd);
    if (!_freq_list.empty()) {
      for (auto p : _freq_list) {
        init_stream.emplace_back(p.first);
      }
    } else {
      init_stream.resize(_nd);
      std::iota(init_stream.begin(), init_stream.end(), 0);
    }
    _lock_nodes.clear();
    _lock_pids.clear();
    _lock_nodes.resize(_nd, false);
    _lock_pids.resize(_partition_number, false);
    unsigned pid = 0;
    vis.clear();
    if (lock_nums) {
      std::cout << "lock first " << lock_nums << " nodes at init stage." << std::endl;
    }
    for (auto i : init_stream) {
      if (vis.count(i)) {
        lock_nums--;
        continue;  // has insert into partition
      }
      if (_partition[pid].size() == C) {
        ++pid;
      }
      vis.insert(i);
      _partition[pid].push_back(i);
      id2pid[i] = pid;
      if (lock_nums > 0) {
        _lock_pids[pid] = true;
      }
      for (unsigned s : full_graph[i]) {
        if (vis.count(s)) continue;
        if (_partition[pid].size() == C) {
          ++pid;
          break;
        }
        _partition[pid].push_back(s);
        id2pid[s] = pid;
        vis.insert(s);
      }
      if (lock_nums) --lock_nums;
    }
    int s = 0;
    for (unsigned i = 0; i < _partition_number; i++) {
      if (!_lock_pids[i]) break;
      for (unsigned s : _partition[i]) {
        _lock_nodes[s] = true;
      }
      s++;
    }
    if (_lock_pids[0]) {
      std::cout << "finally, it locks partition nums: " << s << " locks nodes num: " << s * C << std::endl;
    }

    std::cout << "init over." << std::endl;

    build_undirected_graph();
    for (int i = 0; i < k; i++) {
      auto t0 = omp_get_wtime();
      graph_partition_min_cut_round();
      auto t1 = omp_get_wtime();
      ivf_time += (t1 - t0);
      partition_statistic();
      auto ivf_file_name = std::string(filename) + std::string(".ivf") + std::to_string(i + 1);
      save_partition(ivf_file_name.c_str());
      std::cout << "ivf time: " << t1 - t0 << " round: " << i + 1 << std::endl;
    }
    save_partition(filename);
    std::cout << "select pid nums" << select_nums << " get unfilled partition nums: " << getUnfilled_nums << std::endl;
    std::cout << "total ivf time: " << ivf_time << std::endl;
    
    // 计算并保存partition度中心性统计（替代简单的入度统计）
    calculate_and_save_partition_centrality(filename);
    
    // 计算并保存基于PageRank的page重要性排序
    calculate_and_save_page_pagerank(filename);
  }
  void graph_partition_LDG() {
    free_q.clear();
#pragma omp parallel for
    for (unsigned i = 0; i < _partition_number; i++) {
      if (_lock_pids[i]) continue;
      _partition[i].clear();
      free_q.push(i);
    }

    cur = 0;
    std::cout << "start" << std::endl;
    std::vector<unsigned> stream(_nd);
    std::iota(stream.begin(), stream.end(), 0);
    auto rng = std::default_random_engine{};
    std::shuffle(std::begin(stream), std::end(stream), rng);
    auto start = omp_get_wtime();
#pragma omp parallel for schedule(dynamic)
    for (unsigned i = 0; i < _nd; i++) {
      size_t n = stream[i];
      if (_lock_nodes[n]) continue;
      sync(n);
      cout_step();
    }
    auto end = omp_get_wtime();
    std::cout << "ivf time: " << end - start << " round: " << round << std::endl;
    ivf_time += end - start;
    round++;
  }
  unsigned sync(unsigned i) {
    unsigned pid = select_partition(i);
    pmutex[pid]->lock();

    while (_partition[pid].size() == C) {
      pmutex[pid]->unlock();
      pid = select_partition(i);
      pmutex[pid]->lock();
    }
    _partition[pid].emplace_back(i);
    id2pid[i] = pid;
    unsigned s = _partition[pid].size();
    pmutex[pid]->unlock();

    if (s != C) {
      free_q.push(pid);
    }

    return pid;
  }

  // 基于direct_graph与reverse_graph构建无向图
  void build_undirected_graph() {
    undirect_graph.clear();
    undirect_graph.resize(_nd);
#pragma omp parallel for schedule(dynamic, 100)
    for (unsigned i = 0; i < _nd; i++) {
      std::unordered_set<unsigned> ne;
      ne.reserve(direct_graph[i].size() + reverse_graph[i].size());
      for (auto n : direct_graph[i]) {
        if (n != i) ne.insert(n);
      }
      for (auto n : reverse_graph[i]) {
        if (n != i) ne.insert(n);
      }
      undirect_graph[i].assign(ne.begin(), ne.end());
    }
    std::cout << "undirected graph built." << std::endl;
  }

  // 边权：距离的反比，越近越大。仅在无向图存在边时计权
  inline float edge_weight(unsigned a, unsigned b) {
    if (a == b) return 0.0f;
    const auto &adj = undirect_graph[a];
    bool connected = false;
    for (auto v : adj) {
      if (v == b) { connected = true; break; }
    }
    if (!connected) return 0.0f;
    float d = 0.0f;
    if (_use_disk_pq && _n_chunks > 0) {
      static thread_local std::vector<float> a_fp;
      static thread_local _u64 last_id = std::numeric_limits<_u64>::max();
      if (last_id != a) {
        if (a_fp.size() != _dim) a_fp.assign(_dim, 0.0f);
        const _u8 *code1 = _pq_codes.data() + (_u64)a * _n_chunks;
        _pq_table.inflate_vector(const_cast<_u8*>(code1), a_fp.data());
        last_id = a;
      }
      const _u8 *code2 = _pq_codes.data() + (_u64)b * _n_chunks;
      d = _pq_table.l2_distance(a_fp.data(), const_cast<_u8*>(code2));
    } else {
      d = compute_quantized_distance(a, b);
    }
    if (!std::isfinite(d) || d <= 0.f) d = 1e-6f;
    return 1.0f / (1e-6f + d);
  }
 
  // 为每个分区挑选Top-M最重的邻接分区，并做贪心配对，生成互不冲突的分区对
  std::vector<std::pair<unsigned, unsigned>> build_disjoint_heavy_pairs(unsigned topM_per_partition = 1) {
    std::vector<std::vector<std::pair<unsigned, float>>> heavy_neighbors(_partition_number);
#pragma omp parallel for schedule(dynamic, 64)
    for (int pid = 0; pid < (int)_partition_number; ++pid) {
      if (_lock_pids.size() && _lock_pids[pid]) continue;
      const auto &nodes = _partition[pid];
      if (nodes.empty()) continue;
      std::unordered_map<unsigned, float> cut_w;
      for (auto u : nodes) {
        for (auto v : undirect_graph[u]) {
          unsigned pj = id2pid[v];
          if (pj == INF || pj == (unsigned)pid) continue;
          if (_lock_pids.size() && _lock_pids[pj]) continue;
          float w = edge_weight(u, v);
          if (w <= 0) continue;
          cut_w[pj] += w;
        }
      }
      std::vector<std::pair<unsigned, float>> list(cut_w.begin(), cut_w.end());
      std::sort(list.begin(), list.end(), [](const auto &a, const auto &b){ return a.second > b.second; });
      if (list.size() > topM_per_partition) list.resize(topM_per_partition);
      heavy_neighbors[pid] = std::move(list);
    }

    std::vector<char> used(_partition_number, 0);
    std::vector<std::pair<unsigned, unsigned>> pairs;
    pairs.reserve(_partition_number / 2);
    for (unsigned i = 0; i < _partition_number; ++i) {
      if (used[i]) continue;
      if (_lock_pids.size() && _lock_pids[i]) continue;
      const auto &nei = heavy_neighbors[i];
      for (const auto &pr : nei) {
        unsigned j = pr.first;
        if (i == j || used[j]) continue;
        if (_lock_pids.size() && _lock_pids[j]) continue;
        pairs.emplace_back(i, j);
        used[i] = used[j] = 1;
        break;
      }
    }
    return pairs;
  }

  // 单轮KL/FM交换优化：保持每块容量不变
  void graph_partition_min_cut_round() {
    auto pairs = build_disjoint_heavy_pairs(1);
#pragma omp parallel for schedule(dynamic, 64)
    for (int idx = 0; idx < (int)pairs.size(); ++idx) {
      auto pr = pairs[idx];
      kl_optimize_pair(pr.first, pr.second);
    }
  }

  // KL在两个分区之间做等量交换
  void kl_optimize_pair(unsigned pid_a, unsigned pid_b) {
    auto &A = _partition[pid_a];
    auto &B = _partition[pid_b];
    if (A.empty() || B.empty()) return;
    if (A.size() != B.size()) return;  // 简化：仅处理等大小块
    const size_t P = A.size();

    std::unordered_set<unsigned> setA(A.begin(), A.end());
    std::unordered_set<unsigned> setB(B.begin(), B.end());

    std::vector<float> D_A(P, 0.0f), D_B(P, 0.0f);
    auto recompute_D = [&](bool forA) {
      if (forA) {
#pragma omp parallel for schedule(dynamic, 32)
        for (int ii = 0; ii < (int)P; ++ii) {
          unsigned u = A[ii];
          float in_w = 0.0f, ex_w = 0.0f;
          for (auto v : undirect_graph[u]) {
            if (setA.count(v)) {
              in_w += edge_weight(u, v);
            } else if (setB.count(v)) {
              ex_w += edge_weight(u, v);
            }
          }
          D_A[ii] = ex_w - in_w;
        }
      } else {
#pragma omp parallel for schedule(dynamic, 32)
        for (int ii = 0; ii < (int)P; ++ii) {
          unsigned u = B[ii];
          float in_w = 0.0f, ex_w = 0.0f;
          for (auto v : undirect_graph[u]) {
            if (setB.count(v)) {
              in_w += edge_weight(u, v);
            } else if (setA.count(v)) {
              ex_w += edge_weight(u, v);
            }
          }
          D_B[ii] = ex_w - in_w;
        }
      }
    };

    recompute_D(true);
    recompute_D(false);

    std::vector<int> chosenA; chosenA.reserve(P);
    std::vector<int> chosenB; chosenB.reserve(P);
    std::vector<float> gains; gains.reserve(P);
    std::vector<char> lockedA(P, 0), lockedB(P, 0);

    for (size_t step = 0; step < P; ++step) {
      int best_ai = -1, best_bi = -1;
      float best_gain = -std::numeric_limits<float>::infinity();
      for (size_t ai = 0; ai < P; ++ai) {
        if (lockedA[ai]) continue;
        unsigned a = A[ai];
        for (size_t bi = 0; bi < P; ++bi) {
          if (lockedB[bi]) continue;
          unsigned b = B[bi];
          float g = D_A[ai] + D_B[bi] - 2.0f * edge_weight(a, b);
          if (g > best_gain) {
            best_gain = g; best_ai = (int)ai; best_bi = (int)bi;
          }
        }
      }
      if (best_ai < 0 || best_bi < 0) break;
      chosenA.push_back(best_ai);
      chosenB.push_back(best_bi);
      gains.push_back(best_gain);
      lockedA[best_ai] = 1; lockedB[best_bi] = 1;

      unsigned va = A[best_ai], vb = B[best_bi];
      setA.erase(va); setB.erase(vb);
      setA.insert(vb); setB.insert(va);
      recompute_D(true);
      recompute_D(false);
    }

    float best_sum = -std::numeric_limits<float>::infinity();
    int best_k = -1; float acc = 0.0f;
    for (size_t i = 0; i < gains.size(); ++i) {
      acc += gains[i];
      if (acc > best_sum) { best_sum = acc; best_k = (int)i + 1; }
    }
    if (best_k <= 0 || best_sum <= 1e-6f) return;

    std::unordered_set<unsigned> newA(A.begin(), A.end());
    std::unordered_set<unsigned> newB(B.begin(), B.end());
    for (int i = 0; i < best_k; ++i) {
      unsigned va = A[chosenA[i]];
      unsigned vb = B[chosenB[i]];
      newA.erase(va); newB.erase(vb);
      newA.insert(vb); newB.insert(va);
    }

    std::vector<unsigned> A_new; A_new.reserve(P);
    std::vector<unsigned> B_new; B_new.reserve(P);
    for (auto x : newA) A_new.push_back(x);
    for (auto x : newB) B_new.push_back(x);
    if (A_new.size() != P || B_new.size() != P) return;

    A.swap(A_new); B.swap(B_new);
    for (auto x : A) id2pid[x] = pid_a;
    for (auto x : B) id2pid[x] = pid_b;
  }

  /**
   * 基于Page级别图计算PageRank值来选择要缓存的pages
   * 理论：PageRank能识别在图中具有全局重要性的页面
   */
  void calculate_and_save_page_pagerank(const char* filename) {
    std::cout << "开始构建Page级别图并计算PageRank..." << std::endl;
    auto calc_start = omp_get_wtime();
    
    // Step 1: 构建Page级别的加权图
    struct PageEdge {
      unsigned target_page;
      float weight;
      PageEdge(unsigned target, float w) : target_page(target), weight(w) {}
    };
    
    std::vector<std::vector<PageEdge>> page_graph(_partition_number);
    std::vector<std::unordered_map<unsigned, float>> page_edge_weights(_partition_number);
    
    std::cout << "构建Page图: 统计Page间的连接权重..." << std::endl;
    
    // 遍历所有节点，构建page间的连接 - OpenMP并行化
    #pragma omp parallel
    {
      // 每个线程维护自己的局部权重映射
      std::vector<std::unordered_map<unsigned, float>> local_page_edge_weights(_partition_number);
      
      #pragma omp for schedule(dynamic, 1000)
      for (unsigned node_id = 0; node_id < _nd; node_id++) {
        auto it_a = id2pid.find(node_id);
        if (it_a == id2pid.end()) continue;
        unsigned page_a = it_a->second;  // 当前节点所在的page/partition
        if (page_a >= _partition_number || page_a == INF) continue;
        
        // 遍历该节点的所有邻居
        for (unsigned neighbor : full_graph[node_id]) {
          auto it_b = id2pid.find(neighbor);
          if (it_b == id2pid.end()) continue;
          unsigned page_b = it_b->second;  // 邻居节点所在的page/partition
          if (page_b >= _partition_number || page_b == INF) continue;
          
          if (page_a != page_b) {  // 跨page的边
            local_page_edge_weights[page_a][page_b] += 1.0f;  // 累加权重到局部映射
          }
        }
      }
      
      // 合并所有线程的局部权重映射到全局映射
      #pragma omp critical
      {
        for (unsigned page_a = 0; page_a < _partition_number; page_a++) {
          for (const auto& edge : local_page_edge_weights[page_a]) {
            page_edge_weights[page_a][edge.first] += edge.second;
          }
        }
      }
    }
    
    // 将权重map转换为邻接表 - OpenMP并行化
    std::vector<float> page_outdegree(_partition_number, 0.0f);
    #pragma omp parallel for schedule(static)
    for (unsigned page_a = 0; page_a < _partition_number; page_a++) {
      for (const auto& edge : page_edge_weights[page_a]) {
        unsigned page_b = edge.first;
        float weight = edge.second;
        page_graph[page_a].emplace_back(page_b, weight);
        page_outdegree[page_a] += weight;
      }
    }
    
    std::cout << "Page图构建完成. 共有 " << _partition_number << " 个pages" << std::endl;
    
    // Step 2: 计算PageRank
    std::vector<float> pagerank(_partition_number, 1.0f / _partition_number);
    std::vector<float> new_pagerank(_partition_number);
    const float damping_factor = 0.85f;
    const float tolerance = 1e-6f;
    const int max_iterations = 100;
    
    std::cout << "开始PageRank迭代计算..." << std::endl;
    
    for (int iter = 0; iter < max_iterations; iter++) {
      // 初始化新的PageRank值 - OpenMP并行化
      #pragma omp parallel for schedule(static)
      for (unsigned i = 0; i < _partition_number; i++) {
        new_pagerank[i] = (1.0f - damping_factor) / _partition_number;
      }
      
      // PageRank传播 - OpenMP并行化
      #pragma omp parallel for schedule(static)
      for (unsigned page_a = 0; page_a < _partition_number; page_a++) {
        if (page_outdegree[page_a] > 0) {
          float contribution = damping_factor * pagerank[page_a] / page_outdegree[page_a];
          for (const auto& edge : page_graph[page_a]) {
            #pragma omp atomic
            new_pagerank[edge.target_page] += contribution * edge.weight;
          }
        } else {
          // 处理悬挂页面：将权重平均分配给所有页面
          float dangling_contribution = damping_factor * pagerank[page_a] / _partition_number;
          for (unsigned i = 0; i < _partition_number; i++) {
            #pragma omp atomic
            new_pagerank[i] += dangling_contribution;
          }
        }
      }
      
      // 检查收敛 - OpenMP并行化使用reduction
      float diff = 0.0f;
      #pragma omp parallel for reduction(+:diff)
      for (unsigned i = 0; i < _partition_number; i++) {
        diff += std::abs(new_pagerank[i] - pagerank[i]);
      }
      
      // 更新pagerank向量 - OpenMP并行化
      #pragma omp parallel for schedule(static)
      for (unsigned i = 0; i < _partition_number; i++) {
        pagerank[i] = new_pagerank[i];
      }
      
      if (iter % 10 == 0) {
        std::cout << "PageRank迭代 " << iter << ", 差值: " << diff << std::endl;
      }
      
      if (diff < tolerance) {
        std::cout << "PageRank收敛于第 " << iter + 1 << " 次迭代" << std::endl;
        break;
      }
    }
    
    // Step 3: 按PageRank值排序
    std::vector<std::pair<float, unsigned>> pagerank_pairs(_partition_number);
    // 并行构建PageRank排序对 - OpenMP并行化
    #pragma omp parallel for schedule(static)
    for (unsigned i = 0; i < _partition_number; i++) {
      pagerank_pairs[i] = {pagerank[i], i};
    }
    
    // 按PageRank值降序排序
    std::sort(pagerank_pairs.begin(), pagerank_pairs.end(), 
              [](const std::pair<float, unsigned>& a, const std::pair<float, unsigned>& b) {
                return a.first > b.first;  // 降序排序
              });
    
    // 输出统计信息
    std::cout << "Page PageRank统计（前10个）:" << std::endl;
    int stat_size = (int)(_partition_number/5);
    for (int i = 0; i < std::min(10, stat_size); i++) {
      unsigned page_id = pagerank_pairs[i].second;
      std::cout << "Page " << page_id 
                << ": PageRank=" << pagerank_pairs[i].first
                << " (出度=" << page_outdegree[page_id] 
                << ", 大小=" << _partition[page_id].size() << ")" << std::endl;
    }
    
    auto calc_end = omp_get_wtime();
    std::cout << "Page PageRank计算耗时: " << (calc_end - calc_start) << " 秒" << std::endl;
    
    // Step 4: 保存到文件
    std::string output_filename = std::string(filename) + "_top_pagerank_pages.txt";
    std::ofstream output_file(output_filename);
    if (output_file.is_open()) {
      output_file << "# Top " << stat_size << " pages ranked by PageRank score\n";
      output_file << "# Format: page_id pagerank_score out_degree page_size\n";
      for (int i = 0; i < stat_size; i++) {
        unsigned page_id = pagerank_pairs[i].second;
        output_file << page_id << " " << pagerank_pairs[i].first 
                   << " " << page_outdegree[page_id] << " " << _partition[page_id].size() << std::endl;
      }
      output_file.close();
      std::cout << "PageRank最高的page ID已保存到文件: " << output_filename << std::endl;
    } else {
      std::cout << "无法创建输出文件: " << output_filename << std::endl;
    }
  }

 private:
  /**
   * 计算并输出每个partition的度中心性统计 
   * 综合考虑入度和出度，以及与medoids的距离
   * @param filename 输出文件的基础名称
   */
  void calculate_and_save_partition_centrality(const char* filename) {
    std::cout << "开始计算partition度中心性统计..." << std::endl;
    auto calc_start = omp_get_wtime();
    
    // 计算每个partition的入度、出度和度中心性
    std::vector<unsigned> partition_indegree(_partition_number, 0);
    std::vector<unsigned> partition_outdegree(_partition_number, 0);
    std::vector<float> partition_centrality(_partition_number, 0.0f);
    
#pragma omp parallel for
    for (unsigned pid = 0; pid < _partition_number; pid++) {
      // 遍历当前partition中的每个节点
      for (unsigned node : _partition[pid]) {
        // 计算入度：指向该节点的跨partition边
        for (unsigned neighbor : reverse_graph[node]) {
          unsigned neighbor_pid = id2pid[neighbor];
          if (neighbor_pid != pid && neighbor_pid != INF) {
#pragma omp atomic
            partition_indegree[pid]++;
          }
        }
        
        // 计算出度：该节点指向其他partition的边
        for (unsigned neighbor : direct_graph[node]) {
          unsigned neighbor_pid = id2pid[neighbor];
          if (neighbor_pid != pid && neighbor_pid != INF) {
#pragma omp atomic
            partition_outdegree[pid]++;
          }
        }
      }
    }
    
    // 计算度中心性分数（综合指标）
    std::cout << "计算度中心性分数..." << std::endl;
    for (unsigned pid = 0; pid < _partition_number; pid++) {
      if (_partition[pid].empty()) continue;
      // 加权：入度权重更高（搜索更容易到达）
      float weighted_score = partition_indegree[pid] * 2.0f + partition_outdegree[pid] * 1.0f;
      // 考虑partition大小的影响（大的partition可能更重要）
      float size_bonus = std::log(1.0f + _partition[pid].size());
      partition_centrality[pid] = weighted_score * size_bonus;
    }
    
    // 创建<中心性分数, partition_id>对用于排序
    std::vector<std::pair<float, unsigned>> centrality_pairs;
    for (unsigned i = 0; i < _partition_number; i++) {
      centrality_pairs.push_back({partition_centrality[i], i});
    }
    
    // 按中心性分数降序排序
    std::sort(centrality_pairs.begin(), centrality_pairs.end(), 
              [](const std::pair<float, unsigned>& a, const std::pair<float, unsigned>& b) {
                return a.first > b.first;  // 降序排序
              });
    
    // 输出统计信息
    int stat_size = (int)(_partition_number/5);
    for (int i = 0; i < std::min(10, stat_size); i++) { // 只显示前10个
      unsigned pid = centrality_pairs[i].second;
      std::cout << "Partition " << pid 
                << ": 中心性=" << centrality_pairs[i].first
                << " (入度=" << partition_indegree[pid] 
                << ", 出度=" << partition_outdegree[pid] 
                << ", 大小=" << _partition[pid].size() << ")" << std::endl;
    }
    
    auto calc_end = omp_get_wtime();
    std::cout << "Partition中心性计算耗时: " << (calc_end - calc_start) << " 秒" << std::endl;
    
    // 保存到文件
    std::string output_filename = std::string(filename) + "_top_centrality_partitions.txt";
    std::ofstream output_file(output_filename);
    if (output_file.is_open()) {
      output_file << "# Top " << stat_size << " partitions with highest centrality\n";
      output_file << "# Format: partition_id centrality_score indegree outdegree size\n";
      for (int i = 0; i < stat_size; i++) {
        unsigned pid = centrality_pairs[i].second;
        output_file << pid << " " << centrality_pairs[i].first 
                   << " " << partition_indegree[pid] << " " << partition_outdegree[pid] 
                   << " " << _partition[pid].size() << std::endl;
      }
      output_file.close();
      std::cout << "度中心性最高的partition ID已保存到文件: " << output_filename << std::endl;
    } else {
      std::cout << "无法创建输出文件: " << output_filename << std::endl;
    }
  }



 private:
  size_t _dim;  // vector dimension
  _u64 _nd;     // vector number
  _u64 _max_node_len;
  unsigned _width;                                  // max out-degree
  unsigned _ep;                                     // seed vertex id
  std::vector<std::vector<unsigned>> direct_graph;  // neighbor list
  std::vector<std::vector<unsigned>> full_graph;
  unsigned select_free;
  _u64 C;                                                  // partition size threshold
  _u64 _partition_number = 0;                              // the number of partitions
  std::vector<std::vector<unsigned>> _partition{1000000};  // each partition set
  std::vector<std::unique_ptr<std::mutex>> pmutex;
  int cur = 0;
  std::vector<std::vector<unsigned>> reverse_graph;
  std::vector<std::vector<unsigned>> undirect_graph;
  std::unordered_map<unsigned, unsigned> id2pid;
  std::unordered_map<unsigned, unsigned> id2ratio;
  int round = 0;
  double ivf_time = 0.0;
  bool _visual = false;
  unsigned cursize = 10000;
  uint64_t select_nums = 0;
  uint64_t getUnfilled_nums = 0;
  _u64 E;
  std::uniform_real_distribution<> *_dis;
  std::mt19937 *_gen;
  std::random_device *_rd;
  concurrent_queue free_q;

  std::vector<puu> _freq_list;
  vpu _freq_nei_list;
  std::vector<bool> _lock_nodes;
  std::vector<bool> _lock_pids;
  
  // KNN graph building parameters
  bool _build_knn_graph = false;
  std::vector<T> _data;  // Store original data for distance calculations
  std::unique_ptr<diskann::Distance<T>> _dist_cmp;
  _u64 _n_chunks = 0;  // Number of PQ chunks
  // Disk PQ resources
  bool _use_disk_pq = false;
  std::vector<_u8> _pq_codes;  // size: _nd * _n_chunks
  diskann::FixedChunkPQTable _pq_table;
};
}  // namespace GP