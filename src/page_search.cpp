#include <immintrin.h>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include "logger.h"
#include "percentile_stats.h"
#include "pq_flash_index.h"
#include "timer.h"

#define DYN_BEAM_WIDTH
#define DYN_PAGE_RATIO
#define PAGE_BUF_SIZE 50000
#define ENABLE_PAGE_POOL_REUSE_OPTIMIZATION

namespace diskann {
  template<typename T>
  void PQFlashIndex<T>::load_partition_data(const std::string &index_prefix,
      const _u64 nnodes_per_sector, const _u64 num_points) {
    std::string partition_file = index_prefix + "_partition.bin";
    std::ifstream part(partition_file);
    _u64          C, partition_nums, nd;
    part.read((char *) &C, sizeof(_u64));
    part.read((char *) &partition_nums, sizeof(_u64));
    part.read((char *) &nd, sizeof(_u64));
    if (nnodes_per_sector && num_points &&
        (C != nnodes_per_sector || nd != num_points)) {
      diskann::cerr << "partition information not correct." << std::endl;
      exit(-1);
    }
    diskann::cout << "Partition meta: C: " << C << " partition_nums: " << partition_nums
              << " nd: " << nd << std::endl;
    this->gp_layout_.resize(partition_nums);
    for (unsigned i = 0; i < partition_nums; i++) {
      unsigned s;
      part.read((char *) &s, sizeof(unsigned));
      this->gp_layout_[i].resize(s);
      part.read((char *) gp_layout_[i].data(), sizeof(unsigned) * s);
    }
    this->id2page_.resize(nd);
    part.read((char *) id2page_.data(), sizeof(unsigned) * nd);
    diskann::cout << "Load partition data done." << std::endl;
  }

  template<typename T>
  void PQFlashIndex<T>::load_and_cache_high_priority_partitions(const std::string &index_prefix) {
    std::vector<std::pair<std::string, std::string>> strategies = {
        {"_top_pagerank_pages.txt", "PageRank"},           // 1st choice: Page级PageRank
        {"_top_centrality_partitions.txt", "Centrality"},     // 2nd choice: 度中心性
    };
    
    std::ifstream priority_stream;
    std::string strategy_name = "";
    std::string priority_file = "";
    
    // 尝试找到可用的策略文件
    for (const auto& strategy : strategies) {
      priority_file = index_prefix + strategy.first;
      priority_stream.open(priority_file);
      if (priority_stream.is_open()) {
        strategy_name = strategy.second;
        break;
      }
    }
    
    if (!priority_stream.is_open()) {
      diskann::cout << "Warning: No partition priority file found, skipping pre-caching." << std::endl;
      return;
    }
    
    diskann::cout << "Using cache strategy: " << strategy_name << ", file: " << priority_file << std::endl;
    
    std::string line;
    partition_priority_.clear();
    high_priority_partitions_.clear();
    
    // 跳过注释行并解析数据
    while (std::getline(priority_stream, line)) {
      if (line.empty() || line[0] == '#') continue;
      std::istringstream iss(line);
      unsigned partition_id;
      float priority_score;
      if (iss >> partition_id >> priority_score) {
        partition_priority_.emplace_back(partition_id, priority_score);
        high_priority_partitions_.push_back(partition_id);
      }
    }
    priority_stream.close();
    
    if (partition_priority_.empty()) {
      diskann::cout << "Warning: No valid partition priority information found." << std::endl;
      return;
    }
    
    diskann::cout << "Loaded " << partition_priority_.size() << " partition priority entries using " << strategy_name << " strategy." << std::endl;
    
    // Limit the number of partitions to cache
    unsigned num_to_cache = high_priority_partitions_.size();
    
    diskann::cout << "Pre-caching " << num_to_cache << " high priority partitions..." << std::endl;
    
    // Initialize PagePool for pre-caching
    diskann::cout << "Initializing PagePool for pre-caching..." << std::endl;
    page_pool_.init((uint64_t)(num_to_cache), (uint64_t) SECTOR_LEN);
    
    // Pre-cache high priority partition pages with OpenMP parallelization
    std::atomic<unsigned> cached_pages{0};
    std::atomic<bool> pool_exhausted{false};
    
    #pragma omp parallel for schedule(dynamic, 4) num_threads(std::min(8u, num_to_cache))
    for (unsigned i = 0; i < num_to_cache; i++) {
      // Stop if pool is exhausted
      if (pool_exhausted.load(std::memory_order_acquire)) {
        continue;
      }
      
      unsigned partition_id = high_priority_partitions_[i];
      
      // Check if partition_id is valid
      if (partition_id >= gp_layout_.size()) {
        #pragma omp critical
        {
          diskann::cout << "Warning: Invalid partition_id " << partition_id 
                       << " (max: " << gp_layout_.size() - 1 << ")" << std::endl;
        }
        continue;
      }
      
      // Skip empty partition
      if (gp_layout_[partition_id].empty()) {
        continue;
      }
      
      // partition_id is the page_id!
      unsigned page_id = partition_id;
      
      // The page is already in page pool, skip
      if (page_pool_.contains(page_id)) {
        continue;
      }
      
      // Try to pre-read and cache the page corresponding to this partition
      try {
        char* buf = page_pool_.acquire();
        if (buf == nullptr) {
          pool_exhausted.store(true, std::memory_order_release);
          #pragma omp critical
          {
            diskann::cout << "Warning: Unable to get cache space, stopping pre-caching. Number of cached pages: " 
                         << cached_pages.load() << std::endl;
          }
          continue;
        }
        
        // Use the already initialized thread context (thread-safe pop)
        ThreadData<T> data = this->thread_data.pop();
        while (data.scratch.sector_scratch == nullptr) {
          this->thread_data.wait_for_push_notify();
          data = this->thread_data.pop();
        }
        IOContext &ctx = data.ctx;
        
        // Read page content (using AlignedFileReader)
        std::vector<AlignedRead> read_reqs;
        AlignedRead read_req;
        read_req.buf = buf;
        read_req.len = SECTOR_LEN;
        read_req.offset = SECTOR_LEN * (1 + page_id); // +1 skip header sector, use correct page_id
        read_reqs.push_back(read_req);
        
        reader->read(read_reqs, ctx, false); // Synchronous read (blocking call)
        
        // Return thread context (thread-safe push)
        this->thread_data.push(data);
        this->thread_data.push_notify_all();
        
        // Add page to page pool (using correct page_id)
        char* canonical_buf = page_pool_.add_page(page_id, buf);
        if (canonical_buf != buf) {
          // Page already exists (concurrent case), release our allocated buffer
          page_pool_.release(buf);
        }
        cached_pages.fetch_add(1, std::memory_order_relaxed);
      } catch (const std::exception& e) {
        #pragma omp critical
        {
          diskann::cout << "Warning: Pre-cache partition " << partition_id 
                       << " failed: " << e.what() << std::endl;
        }
        continue;
      }
    }
    diskann::cout << "Successfully pre-cached " << cached_pages.load()
                   << " high priority partition pages using " << strategy_name << " strategy." << std::endl;
  }

  template<typename T>
  void PQFlashIndex<T>::page_search(
      const T *query1, const _u64 k_search, const _u32 mem_L, const _u64 l_search, _u64 *indices,
      float *distances, const _u64 beam_width, const _u32 io_limit,
      const bool use_reorder_data, const float use_ratio, QueryStats *stats) {
    ThreadData<T> data = this->thread_data.pop();
    while (data.scratch.sector_scratch == nullptr) {
      this->thread_data.wait_for_push_notify();
      data = this->thread_data.pop();
    }

    if (beam_width > MAX_N_SECTOR_READS)
      throw ANNException("Beamwidth can not be higher than MAX_N_SECTOR_READS",
                         -1, __FUNCSIG__, __FILE__, __LINE__);

    // copy query to thread specific aligned and allocated memory (for distance
    // calculations we need aligned data)
    float        query_norm = 0;
    const T *    query = data.scratch.aligned_query_T;
    const float *query_float = data.scratch.aligned_query_float;

    uint32_t query_dim = metric == diskann::Metric::INNER_PRODUCT ? this-> data_dim - 1: this-> data_dim;

    for (uint32_t i = 0; i < query_dim; i++) {
      data.scratch.aligned_query_float[i] = query1[i];
      data.scratch.aligned_query_T[i] = query1[i];
      query_norm += query1[i] * query1[i];
    }

    // if inner product, we also normalize the query and set the last coordinate
    // to 0 (this is the extra coordindate used to convert MIPS to L2 search)
    if (metric == diskann::Metric::INNER_PRODUCT) {
      query_norm = std::sqrt(query_norm);
      data.scratch.aligned_query_T[this->data_dim - 1] = 0;
      data.scratch.aligned_query_float[this->data_dim - 1] = 0;
      for (uint32_t i = 0; i < this->data_dim - 1; i++) {
        data.scratch.aligned_query_T[i] /= query_norm;
        data.scratch.aligned_query_float[i] /= query_norm;
      }
    }

    IOContext &ctx = data.ctx;
    auto query_scratch = &(data.scratch);

    // reset query
    query_scratch->reset();

    // pointers to buffers for data
    T *   data_buf = query_scratch->coord_scratch;
    _mm_prefetch((char *) data_buf, _MM_HINT_T1);

    // sector scratch
    char *sector_scratch = query_scratch->sector_scratch;
    _u64 &sector_scratch_idx = query_scratch->sector_idx;

    // query <-> PQ chunk centers distances
    float *pq_dists = query_scratch->aligned_pqtable_dist_scratch;
    pq_table.populate_chunk_distances(query_float, pq_dists);

    // query <-> neighbor list
    float *dist_scratch = query_scratch->aligned_dist_scratch;
    _u8 *  pq_coord_scratch = query_scratch->aligned_pq_coord_scratch;

    Timer                 query_timer, io_timer, cpu_timer;
    std::vector<Neighbor> retset(l_search + 1);
    tsl::robin_set<_u64> &visited = *(query_scratch->visited);
    tsl::robin_set<unsigned> &page_visited = *(query_scratch->page_visited);
    unsigned cur_list_size = 0;

    std::vector<Neighbor> full_retset;
    full_retset.reserve(4096);

    // Dynamic beam width - using static policy
#ifdef DYN_BEAM_WIDTH
    _u32 cur_beam_width = 4;  // start with small beam width
#else
    _u32 cur_beam_width = beam_width;  // use fixed beam width
#endif
    _u32 max_marker = 0;  // track search progress

    // Dynamic page ratio - using static policy
#ifdef DYN_PAGE_RATIO
    float cur_use_ratio = 1;  // start with small page ratio
#else
    float cur_use_ratio = use_ratio;  // use fixed page ratio
#endif

    _u32                        best_medoid = 0;
    float                       best_dist = (std::numeric_limits<float>::max)();
    std::vector<SimpleNeighbor> medoid_dists;
    for (_u64 cur_m = 0; cur_m < num_medoids; cur_m++) {
      float cur_expanded_dist = dist_cmp_float->compare(
          query_float, centroid_data + aligned_dim * cur_m,
          (unsigned) aligned_dim);
      if (cur_expanded_dist < best_dist) {
        best_medoid = medoids[cur_m];
        best_dist = cur_expanded_dist;
      }
    }

    // lambda to batch compute query<-> node distances in PQ space
    auto compute_pq_dists = [this, pq_coord_scratch, pq_dists](const unsigned *ids,
                                                            const _u64 n_ids,
                                                            float *dists_out) {
      pq_flash_index_utils::aggregate_coords(ids, n_ids, this->data, this->n_chunks,
                         pq_coord_scratch);
      pq_flash_index_utils::pq_dist_lookup(pq_coord_scratch, n_ids, this->n_chunks, pq_dists,
                       dists_out);
    };

    auto compute_extact_dists_and_push = [&](const char* node_buf, const unsigned id) -> float {
      T *node_fp_coords_copy = data_buf;
      memcpy(node_fp_coords_copy, node_buf, disk_bytes_per_point);
      float cur_expanded_dist = dist_cmp->compare(query, node_fp_coords_copy,
                                            (unsigned) aligned_dim);
      full_retset.push_back(Neighbor(id, cur_expanded_dist, true));
      return cur_expanded_dist;
    };

    auto compute_and_push_nbrs = [&](const char *node_buf, unsigned& nk) {
      unsigned *node_nbrs = OFFSET_TO_NODE_NHOOD(node_buf);
      unsigned nnbrs = *(node_nbrs++);
      unsigned nbors_cand_size = 0;
      for (unsigned m = 0; m < nnbrs; ++m) {
        if (visited.find(node_nbrs[m]) == visited.end()) {
          node_nbrs[nbors_cand_size++] = node_nbrs[m];
          visited.insert(node_nbrs[m]);
        }
      }
      if (nbors_cand_size) {
        compute_pq_dists(node_nbrs, nbors_cand_size, dist_scratch);
        for (unsigned m = 0; m < nbors_cand_size; ++m) {
          const int nbor_id = node_nbrs[m];
          const float nbor_dist = dist_scratch[m];
          if (stats != nullptr) {
            stats->n_cmps++;
          }
          if (nbor_dist >= retset[cur_list_size - 1].distance &&
              (cur_list_size == l_search))
            continue;
          Neighbor nn(nbor_id, nbor_dist, true);
          // Return position in sorted list where nn inserted
          auto     r = InsertIntoPool(retset.data(), cur_list_size, nn);
          if (cur_list_size < l_search) ++cur_list_size;
          // nk logs the best position in the retset that was updated due to neighbors of n.
          if (r < nk) nk = r;
        }
      }
    };

    auto compute_and_add_to_retset = [&](const unsigned *node_ids, const _u64 n_ids) {
      compute_pq_dists(node_ids, n_ids, dist_scratch);
      for (_u64 i = 0; i < n_ids; ++i) {
        retset[cur_list_size].id = node_ids[i];
        retset[cur_list_size].distance = dist_scratch[i];
        retset[cur_list_size++].flag = true;
        visited.insert(node_ids[i]);
      }
    };

    if (mem_L) {
      std::vector<unsigned> mem_tags(mem_L);
      std::vector<float> mem_dists(mem_L);
      std::vector<T*> res = std::vector<T*>();
      mem_index_->search_with_tags(query, mem_L, mem_L, mem_tags.data(), mem_dists.data(), nullptr, res);
      compute_and_add_to_retset(mem_tags.data(), std::min((unsigned)mem_L,(unsigned)l_search));
    } else {
      compute_and_add_to_retset(&best_medoid, 1);
    }

    std::sort(retset.begin(), retset.begin() + cur_list_size);

    unsigned num_ios = 0;
    unsigned k = 0;

    // cleared every iteration
    std::vector<unsigned> frontier;
    frontier.reserve(2 * beam_width);
    std::vector<std::pair<unsigned, char *>> frontier_nhoods;
    frontier_nhoods.reserve(2 * beam_width);
    std::vector<AlignedRead> frontier_read_reqs;
    frontier_read_reqs.reserve(2 * beam_width);
    std::vector<std::pair<unsigned, std::pair<unsigned, unsigned *>>>
        cached_nhoods;
    cached_nhoods.reserve(2 * beam_width);

    std::vector<std::pair<unsigned, char *>> page_cached_nhoods;
    page_cached_nhoods.reserve(2 * beam_width);
    std::vector<unsigned> page_cache_release_pids;
  
    // 优化：直接存储页面指针，避免memcpy
    struct PageInfo {
      unsigned id;
      char* sector_buf;
    };
    std::vector<PageInfo> last_pages;
    last_pages.reserve(2 * beam_width);
    int n_ops = 0;

    while (k < cur_list_size && num_ios < io_limit) {
      unsigned nk = cur_list_size;
      // clear iteration state
      frontier.clear();
      frontier_nhoods.clear();
      frontier_read_reqs.clear();
      cached_nhoods.clear();
      sector_scratch_idx = 0;

#ifdef DYN_BEAM_WIDTH
      // Update beam width using static policy based on search progress
      constexpr _u32 kBeamWidths[] = {4, 4, 8, 8, 8, 8, 16, 16, 16};
      cur_beam_width = kBeamWidths[std::min(max_marker / 3, 8u)];
      // Ensure we don't exceed the maximum beam width
      cur_beam_width = std::min(cur_beam_width, (_u32)beam_width);
#endif

#ifdef DYN_PAGE_RATIO
      // Update page ratio using static policy based on search progress
      constexpr float kPageRatios[] = {1.0, 1.0, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2};
      cur_use_ratio = kPageRatios[std::min(max_marker / 5, 8u)];
      // Ensure we don't exceed the maximum page ratio
      cur_use_ratio = std::min(cur_use_ratio, use_ratio);
#endif

      // find new beam
      _u32 marker = k;
      _u32 num_seen = 0;

      // distribute cache and disk-read nodes
      while (marker < cur_list_size && frontier.size() < cur_beam_width &&
             num_seen < cur_beam_width) {
        const unsigned pid = id2page_[retset[marker].id];
        if (retset[marker].flag && page_visited.find(pid) == page_visited.end()) {
          num_seen++;
          char* cached_page_buf = page_pool_.enter_page(pid);
          if (cached_page_buf != nullptr) {
            page_cached_nhoods.emplace_back(pid, cached_page_buf);
            page_cache_release_pids.push_back(pid);
            page_visited.insert(pid);
            retset[marker].flag = false;
            if (stats != nullptr) {
              stats->n_cache_hits++;
            }
            marker++;
            continue;
          } 

          auto iter = nhood_cache.find(retset[marker].id);
          if (iter != nhood_cache.end()) {
            cached_nhoods.push_back(
                std::make_pair(retset[marker].id, iter->second));
            if (stats != nullptr) {
              stats->n_cache_hits++;
            }
          } else {
            frontier.push_back(retset[marker].id);
            page_visited.insert(pid);
          }
          retset[marker].flag = false;
        }
        marker++;
      }

      // Update max_marker for progress tracking
      max_marker = std::max(max_marker, marker);

      // read nhoods of frontier ids
      if (!frontier.empty()) {
        if (stats != nullptr)
          stats->n_hops++;
        for (_u64 i = 0; i < frontier.size(); i++) {
          auto id = frontier[i];
          std::pair<_u32, char *> fnhood;
          fnhood.first = id;
          fnhood.second = sector_scratch + sector_scratch_idx * SECTOR_LEN;
          sector_scratch_idx++;
          frontier_nhoods.push_back(fnhood);
          frontier_read_reqs.emplace_back(
              (static_cast<_u64>(id2page_[id]+1)) * SECTOR_LEN, SECTOR_LEN,
              fnhood.second);
          if (stats != nullptr) {
            stats->n_4k++;
            stats->n_ios++;
          }
          num_ios++;
        }
        n_ops = reader->submit_reqs(frontier_read_reqs, ctx);
        if (this->count_visited_nodes) {
#pragma omp critical
          {
            auto &cnt = this->node_visit_counter[retset[marker].id].second;
            ++cnt;
          }
        }
      }

      // compute remaining nodes in the pages that are fetched in the previous round
      for (size_t i = 0; i < last_pages.size(); ++i) {
        const unsigned last_io_id = last_pages[i].id;
        char    *sector_buf = last_pages[i].sector_buf;
        const unsigned pid = id2page_[last_io_id];
        const unsigned p_size = gp_layout_[pid].size();
        // minus one for the vector that is computed previously
        unsigned vis_size = cur_use_ratio * (p_size - 1);
        std::vector<std::pair<float, const char*>> vis_cand;
        vis_cand.reserve(p_size);
        for (unsigned j = 0; j < p_size; ++j) {
          const unsigned id = gp_layout_[pid][j];
          if (id == last_io_id) continue;
          
          const char* node_buf = sector_buf + j * max_node_len;
          const T* node_coords = (const T*)node_buf;  // 直接转换，避免memcpy
          
          float cur_expanded_dist = dist_cmp->compare(query, node_coords, (unsigned) aligned_dim);
          full_retset.push_back(Neighbor(id, cur_expanded_dist, true));
          vis_cand.emplace_back(cur_expanded_dist, node_buf);
        }
        if (vis_size && vis_size != p_size) {
          std::sort(vis_cand.begin(), vis_cand.end());
        }

        // compute PQ distances for neighbours of the vectors in the page
        for (unsigned j = 0; j < vis_size; ++j) {
          compute_and_push_nbrs(vis_cand[j].second, nk);
        }
      }
      last_pages.clear();

      // process page buffer cache hits: compute both target id and page neighbors
      if (!page_cached_nhoods.empty()) {
        for (auto &pc : page_cached_nhoods) {
          const unsigned pid = pc.first;  // now directly stores page_id
          char *sector_buf = pc.second;
          const unsigned page_size = gp_layout_[pid].size();
          unsigned vis_size = cur_use_ratio * page_size;
          std::vector<std::pair<float, const char*>> vis_cand;
          vis_cand.reserve(page_size);
          for (unsigned j = 0; j < page_size; ++j) {
            const unsigned node_id = gp_layout_[pid][j];
            char *node_buf = sector_buf + j * max_node_len;

            const T* node_coords = (const T*)node_buf;
            float cur_expanded_dist = dist_cmp->compare(query, node_coords, (unsigned) aligned_dim);
            full_retset.push_back(Neighbor(node_id, cur_expanded_dist, true));
            vis_cand.emplace_back(cur_expanded_dist, node_buf);
          }
          if (vis_size && vis_size != page_size) {
            std::sort(vis_cand.begin(), vis_cand.end());
          }

          // compute PQ distances for neighbours of the vectors in the page
          for (unsigned j = 0; j < vis_size; ++j) {
            compute_and_push_nbrs(vis_cand[j].second, nk);
          }
        }

        for (unsigned rpid : page_cache_release_pids) {
          page_pool_.leave_page(rpid);
        }
        page_cached_nhoods.clear();
        page_cache_release_pids.clear();
      }

      // process cached nhoods
      for (auto &cached_nhood : cached_nhoods) {
        auto id = cached_nhood.first;
        auto  global_cache_iter = coord_cache.find(cached_nhood.first);
        T *   node_fp_coords_copy = global_cache_iter->second;
        unsigned nnr = cached_nhood.second.first;
        unsigned* cnhood = cached_nhood.second.second;
        char node_buf[max_node_len];
        memcpy(node_buf, node_fp_coords_copy, disk_bytes_per_point);
        memcpy((node_buf + disk_bytes_per_point), &nnr, sizeof(unsigned));
        memcpy((node_buf + disk_bytes_per_point + sizeof(unsigned)), cnhood, sizeof(unsigned)*nnr);
        compute_extact_dists_and_push(node_buf, id);
        compute_and_push_nbrs(node_buf, nk);
      }

      if (marker == k) {
        break;
      }

      // get last submitted io results, blocking
      if (!frontier.empty()) {
        reader->get_events(ctx, n_ops);
      }

      for (auto &frontier_nhood : frontier_nhoods) {
        char *sector_buf = frontier_nhood.second;
        unsigned pid = id2page_[frontier_nhood.first];
        last_pages.push_back({frontier_nhood.first, sector_buf});
        for (unsigned j = 0; j < gp_layout_[pid].size(); ++j) {
          unsigned id = gp_layout_[pid][j];
          if (id == frontier_nhood.first) {
            char *node_buf = sector_buf + j * max_node_len;
            const T* node_coords = (const T*)node_buf;
            float cur_expanded_dist = dist_cmp->compare(query, node_coords, (unsigned) aligned_dim);
            full_retset.push_back(Neighbor(id, cur_expanded_dist, true));
            compute_and_push_nbrs(node_buf, nk);
          }
        }
      }

      // update best inserted position
      if (nk <= k)
        k = nk;  // k is the best position in retset updated in this round.
      else {
        while (++k < cur_list_size  && !retset[k].flag) {}
      }
    }

    // re-sort by distance
    std::sort(full_retset.begin(), full_retset.end(),
              [](const Neighbor &left, const Neighbor &right) {
                return left.distance < right.distance;
              });

    // copy k_search values
    _u64 t = 0;
    for (_u64 i = 0; i < full_retset.size() && t < k_search; i++) {
      if(i > 0 && full_retset[i].id == full_retset[i-1].id){
        continue;
      }
      indices[t] = full_retset[i].id;
      if (distances != nullptr) {
        distances[t] = full_retset[i].distance;
        if (metric == diskann::Metric::INNER_PRODUCT) {
          // flip the sign to convert min to max
          distances[t] = (-distances[t]);
          // rescale to revert back to original norms (cancelling the effect of
          // base and query pre-processing)
          if (max_base_norm != 0)
            distances[t] *= (max_base_norm * query_norm);
        }
      }
      t++;
    }

    if (t < k_search) {
      diskann::cerr << "The number of unique ids is less than topk, t = " << t << ", k_search = " << k_search << std::endl;
      // exit(1);
    }



    this->thread_data.push(data);
    this->thread_data.push_notify_all();

    if (stats != nullptr) {
      stats->total_us = (double) query_timer.elapsed();
    }
  }

  template<typename T>
  void PQFlashIndex<T>::page_search_interim(
      const _u64 k_search, const _u32 mem_L, const _u64 l_search, _u64 *indices,
      float *distances, const _u64 beam_width, const _u32 io_limit,
      const bool use_reorder_data, const float use_ratio, QueryStats *stats, PageSearchPersistData<T>* persist_data) {

    if (persist_data == nullptr) {
      std::cerr << "this function needs persistent data" << std::endl;
      exit(1);
    }

    std::vector<Neighbor>& retset = persist_data->ret_set;
    NeighborVec& kicked = persist_data->kicked;
    std::vector<Neighbor>& full_retset= persist_data->full_ret_set;
    ThreadData<T>& data = persist_data->thread_data;
    unsigned cur_list_size = persist_data->cur_list_size;
    auto query_scratch = &(data.scratch);
    float *pq_dists = query_scratch->aligned_pqtable_dist_scratch;
    T *data_buf = query_scratch->coord_scratch;
    _mm_prefetch((char *) data_buf, _MM_HINT_T1);
    const T *query = data.scratch.aligned_query_T;
    IOContext &ctx = data.ctx;

    char *sector_scratch = query_scratch->sector_scratch;
    _u64 &sector_scratch_idx = query_scratch->sector_idx;
    float *dist_scratch = query_scratch->aligned_dist_scratch;
    _u8 *  pq_coord_scratch = query_scratch->aligned_pq_coord_scratch;
    tsl::robin_set<_u64> &visited = *(query_scratch->visited);
    tsl::robin_set<unsigned> &page_visited = *(query_scratch->page_visited);

    if (beam_width > MAX_N_SECTOR_READS)
      throw ANNException("Beamwidth can not be higher than MAX_N_SECTOR_READS",
                         -1, __FUNCSIG__, __FILE__, __LINE__);

    // Dynamic beam width - using static policy
#ifdef DYN_BEAM_WIDTH
    _u32 cur_beam_width = 4;  // start with small beam width
#else
    _u32 cur_beam_width = beam_width;  // use fixed beam width
#endif
    _u32 max_marker = 0;  // track search progress

    // lambda to batch compute query<-> node distances in PQ space
    auto compute_pq_dists = [this, pq_coord_scratch, pq_dists](const unsigned *ids,
                                                            const _u64 n_ids,
                                                            float *dists_out) {
      pq_flash_index_utils::aggregate_coords(ids, n_ids, this->data, this->n_chunks,
                         pq_coord_scratch);
      pq_flash_index_utils::pq_dist_lookup(pq_coord_scratch, n_ids, this->n_chunks, pq_dists,
                       dists_out);
    };

    auto compute_extact_dists_and_push = [&](const char* node_buf, const unsigned id) -> float {
      T *node_fp_coords_copy = data_buf;
      memcpy(node_fp_coords_copy, node_buf, disk_bytes_per_point);
      float cur_expanded_dist = dist_cmp->compare(query, node_fp_coords_copy,
                                            (unsigned) aligned_dim);
      full_retset.push_back(Neighbor(id, cur_expanded_dist, true));
      return cur_expanded_dist;
    };

    auto compute_and_push_nbrs = [&](const char *node_buf, unsigned& nk) {
      unsigned *node_nbrs = OFFSET_TO_NODE_NHOOD(node_buf);
      unsigned nnbrs = *(node_nbrs++);
      unsigned nbors_cand_size = 0;
      for (unsigned m = 0; m < nnbrs; ++m) {
        if (visited.find(node_nbrs[m]) == visited.end()) {
          node_nbrs[nbors_cand_size++] = node_nbrs[m];
          visited.insert(node_nbrs[m]);
        }
      }
      if (nbors_cand_size) {
        compute_pq_dists(node_nbrs, nbors_cand_size, dist_scratch);
        for (unsigned m = 0; m < nbors_cand_size; ++m) {
          const int nbor_id = node_nbrs[m];
          const float nbor_dist = dist_scratch[m];
          if (stats != nullptr) {
            stats->n_cmps++;
          }
          if (nbor_dist >= retset[cur_list_size - 1].distance &&
              (cur_list_size == l_search)) {
            kicked.insert(Neighbor(nbor_id, nbor_dist, true));
            continue;
          }
          Neighbor nn(nbor_id, nbor_dist, true);
          // Return position in sorted list where nn inserted
          auto     r = InsertIntoPool(retset.data(), cur_list_size, nn, kicked, l_search);
          if (cur_list_size < l_search) ++cur_list_size;
          // nk logs the best position in the retset that was updated due to neighbors of n.
          if (r < nk) nk = r;
        }
      }
    };

    unsigned num_ios = 0;
    unsigned k = 0;

    // cleared every iteration
    std::vector<unsigned> frontier;
    frontier.reserve(2 * beam_width);
    std::vector<std::pair<unsigned, char *>> frontier_nhoods;
    frontier_nhoods.reserve(2 * beam_width);
    std::vector<AlignedRead> frontier_read_reqs;
    frontier_read_reqs.reserve(2 * beam_width);
    std::vector<std::pair<unsigned, std::pair<unsigned, unsigned *>>>
        cached_nhoods;
    cached_nhoods.reserve(2 * beam_width);

    // 优化：直接存储页面指针，避免memcpy
    struct PageInfo {
      unsigned id;
      char* sector_buf;
    };
    std::vector<PageInfo> last_pages;
    last_pages.reserve(2 * beam_width);
    int n_ops = 0;

    while (k < cur_list_size && num_ios < io_limit) {
      unsigned nk = cur_list_size;
      // clear iteration state
      frontier.clear();
      frontier_nhoods.clear();
      frontier_read_reqs.clear();
      cached_nhoods.clear();
      sector_scratch_idx = 0;

#ifdef DYN_BEAM_WIDTH
      // Update beam width using static policy based on search progress
      constexpr _u32 kBeamWidths[] = {4, 4, 8, 8, 16, 16, 24, 24, 32};
      cur_beam_width = kBeamWidths[std::min(max_marker / 5, 8u)];
      // Ensure we don't exceed the maximum beam width
      cur_beam_width = std::min(cur_beam_width, (_u32)beam_width);
#endif

      // find new beam
      _u32 marker = k;
      _u32 num_seen = 0;

      // distribute cache and disk-read nodes
      while (marker < cur_list_size && frontier.size() < cur_beam_width &&
             num_seen < cur_beam_width) {
        const unsigned pid = id2page_[retset[marker].id];
        if (page_visited.find(pid) == page_visited.end() && retset[marker].flag) {
          num_seen++;
          auto iter = nhood_cache.find(retset[marker].id);
          if (iter != nhood_cache.end()) {
            cached_nhoods.push_back(
                std::make_pair(retset[marker].id, iter->second));
            if (stats != nullptr) {
              stats->n_cache_hits++;
            }
          } else {
            frontier.push_back(retset[marker].id);
            page_visited.insert(pid);
          }
          retset[marker].flag = false;
        }
        marker++;
      }

      // Update max_marker for progress tracking
      max_marker = std::max(max_marker, marker);

      // read nhoods of frontier ids
      if (!frontier.empty()) {
        if (stats != nullptr)
          stats->n_hops++;
        for (_u64 i = 0; i < frontier.size(); i++) {
          auto                    id = frontier[i];
          std::pair<_u32, char *> fnhood;
          fnhood.first = id;
          fnhood.second = sector_scratch + sector_scratch_idx * SECTOR_LEN;
          sector_scratch_idx++;
          frontier_nhoods.push_back(fnhood);
          frontier_read_reqs.emplace_back(
              (static_cast<_u64>(id2page_[id]+1)) * SECTOR_LEN, SECTOR_LEN,
              fnhood.second);
          if (stats != nullptr) {
            stats->n_4k++;
            stats->n_ios++;
          }
          num_ios++;
        }
        n_ops = reader->submit_reqs(frontier_read_reqs, ctx);
        if (this->count_visited_nodes) {
#pragma omp critical
          {
            auto &cnt = this->node_visit_counter[retset[marker].id].second;
            ++cnt;
          }
        }
      }

      // compute remaining nodes in the pages that are fetched in the previous round
      for (size_t i = 0; i < last_pages.size(); ++i) {
        const unsigned last_io_id = last_pages[i].id;
        char    *sector_buf = last_pages[i].sector_buf;
        const unsigned pid = id2page_[last_io_id];
        const unsigned p_size = gp_layout_[pid].size();
        // minus one for the vector that is computed previously
        unsigned vis_size = use_ratio * (p_size - 1);
        std::vector<std::pair<float, const char*>> vis_cand;
        vis_cand.reserve(p_size);

        // compute exact distances of the vectors within the page
        for (unsigned j = 0; j < p_size; ++j) {
          const unsigned id = gp_layout_[pid][j];
          if (id == last_io_id) continue;
          const char* node_buf = sector_buf + j * max_node_len;
          float dist = compute_extact_dists_and_push(node_buf, id);
          vis_cand.emplace_back(dist, node_buf);
        }
        if (vis_size && vis_size != p_size) {
          std::sort(vis_cand.begin(), vis_cand.end());
        }

        // compute PQ distances for neighbours of the vectors in the page
        for (unsigned j = 0; j < vis_size; ++j) {
          compute_and_push_nbrs(vis_cand[j].second, nk);
        }
      }
      last_pages.clear();

      // process cached nhoods
      for (auto &cached_nhood : cached_nhoods) {
        auto id = cached_nhood.first;
        auto  global_cache_iter = coord_cache.find(cached_nhood.first);
        T *   node_fp_coords_copy = global_cache_iter->second;
        unsigned nnr = cached_nhood.second.first;
        unsigned* cnhood = cached_nhood.second.second;
        char node_buf[max_node_len];
        memcpy(node_buf, node_fp_coords_copy, disk_bytes_per_point);
        memcpy((node_buf + disk_bytes_per_point), &nnr, sizeof(unsigned));
        memcpy((node_buf + disk_bytes_per_point + sizeof(unsigned)), cnhood, sizeof(unsigned)*nnr);
        compute_extact_dists_and_push(node_buf, id);
        compute_and_push_nbrs(node_buf, nk);
      }

      // get last submitted io results, blocking
      if (!frontier.empty()) {
        reader->get_events(ctx, n_ops);
      }

      // compute only the desired vectors in the pages - one for each page
      // postpone remaining vectors to the next round
      for (auto &frontier_nhood : frontier_nhoods) {
        char *sector_buf = frontier_nhood.second;
        unsigned pid = id2page_[frontier_nhood.first];
        // 优化：直接存储页面指针，避免memcpy
        last_pages.push_back({frontier_nhood.first, sector_buf});

        for (unsigned j = 0; j < gp_layout_[pid].size(); ++j) {
          unsigned id = gp_layout_[pid][j];
          if (id == frontier_nhood.first) {
            char *node_buf = sector_buf + j * max_node_len;
            compute_extact_dists_and_push(node_buf, id);
            compute_and_push_nbrs(node_buf, nk);
          }
        }
      }

      // update best inserted position
      if (nk <= k)
        k = nk;  // k is the best position in retset updated in this round.
      else
        ++k;
    }

    // re-sort by distance
    std::sort(full_retset.begin(), full_retset.end(),
              [](const Neighbor &left, const Neighbor &right) {
                return left.distance < right.distance;
              });

    // copy k_search values
    _u64 t = 0;
    for (_u64 i = 0; i < full_retset.size() && t < k_search; i++) {
      if(i > 0 && full_retset[i].id == full_retset[i-1].id){
        std::cout << "has replica" << std::endl;
        continue;
      }
      indices[t] = full_retset[i].id;
      if (distances != nullptr) {
        distances[t] = full_retset[i].distance;
        if (metric == diskann::Metric::INNER_PRODUCT) {
          // flip the sign to convert min to max
          distances[t] = (-distances[t]);
          // rescale to revert back to original norms (cancelling the effect of
          // base and query pre-processing)
          if (max_base_norm != 0)
            distances[t] *= (max_base_norm * persist_data->query_norm);
        }
      }
      t++;
    }

    if (t < k_search) {
      diskann::cerr << "The number of unique ids is less than topk" << std::endl;
      exit(1);
    }

    persist_data->cur_list_size = cur_list_size;
  }
  template<typename T>
  void PQFlashIndex<T>::page_search_sq(
      const T *query1, const _u64 k_search, const _u32 mem_L, const _u64 l_search, _u64 *indices,
      float *distances, const _u64 beam_width, const _u32 io_limit,
      const bool use_reorder_data, const float use_ratio, QueryStats *stats) {
    ThreadData<T> data = this->thread_data.pop();
    while (data.scratch.sector_scratch == nullptr) {
      this->thread_data.wait_for_push_notify();
      data = this->thread_data.pop();
    }

    if (beam_width > MAX_N_SECTOR_READS)
      throw ANNException("Beamwidth can not be higher than MAX_N_SECTOR_READS",
                         -1, __FUNCSIG__, __FILE__, __LINE__);

    // copy query to thread specific aligned and allocated memory (for distance
    // calculations we need aligned data)
    float        query_norm = 0;
    const float *    query = data.scratch.aligned_query_float;
    const float *query_float = data.scratch.aligned_query_float;

    uint32_t query_dim = metric == diskann::Metric::INNER_PRODUCT ? this-> data_dim - 1: this-> data_dim;

    for (uint32_t i = 0; i < query_dim; i++) {
      data.scratch.aligned_query_float[i] = query1[i];
      query_norm += query1[i] * query1[i];
    }

    // if inner product, we also normalize the query and set the last coordinate
    // to 0 (this is the extra coordindate used to convert MIPS to L2 search)
    if (metric == diskann::Metric::INNER_PRODUCT) {
      query_norm = std::sqrt(query_norm);
      data.scratch.aligned_query_T[this->data_dim - 1] = 0;
      data.scratch.aligned_query_float[this->data_dim - 1] = 0;
      for (uint32_t i = 0; i < this->data_dim - 1; i++) {
        data.scratch.aligned_query_T[i] /= query_norm;
        data.scratch.aligned_query_float[i] /= query_norm;
      }
    }

    // if sq table not find, return
    if(! this-> frac){
      std::cout << "need first load max min table." << std::endl;
      exit(-1);
    }
    IOContext &ctx = data.ctx;
    auto       query_scratch = &(data.scratch);

    // reset query
    query_scratch->reset();

    // pointers to buffers for data
    T *   data_buf = query_scratch->coord_scratch;
    _mm_prefetch((char *) data_buf, _MM_HINT_T1);

    // sector scratch
    char *sector_scratch = query_scratch->sector_scratch;
    _u64 &sector_scratch_idx = query_scratch->sector_idx;

    // query <-> PQ chunk centers distances
    float *pq_dists = query_scratch->aligned_pqtable_dist_scratch;
    pq_table.populate_chunk_distances(query_float, pq_dists);

    // query <-> neighbor list
    float *dist_scratch = query_scratch->aligned_dist_scratch;
    _u8 *  pq_coord_scratch = query_scratch->aligned_pq_coord_scratch;

    Timer                 query_timer, io_timer, cpu_timer;
    std::vector<Neighbor> retset(l_search + 1);
    tsl::robin_set<_u64> &visited = *(query_scratch->visited);
    tsl::robin_set<unsigned> &page_visited = *(query_scratch->page_visited);
    unsigned cur_list_size = 0;

    std::vector<Neighbor> full_retset;
    full_retset.reserve(4096);

    // Dynamic beam width - using static policy
#ifdef DYN_BEAM_WIDTH
    _u32 cur_beam_width = 4;  // start with small beam width
#else
    _u32 cur_beam_width = beam_width;  // use fixed beam width
#endif
    _u32 max_marker = 0;  // track search progress

    _u32                        best_medoid = 0;
    float                       best_dist = (std::numeric_limits<float>::max)();
    std::vector<SimpleNeighbor> medoid_dists;
    for (_u64 cur_m = 0; cur_m < num_medoids; cur_m++) {
      float cur_expanded_dist = dist_cmp_float->compare(
          query_float, centroid_data + aligned_dim * cur_m,
          (unsigned) aligned_dim);
      if (cur_expanded_dist < best_dist) {
        best_medoid = medoids[cur_m];
        best_dist = cur_expanded_dist;
      }
    }

    // lambda to batch compute query<-> node distances in PQ space
    auto compute_pq_dists = [this, pq_coord_scratch, pq_dists](const unsigned *ids,
                                                            const _u64 n_ids,
                                                            float *dists_out) {
      pq_flash_index_utils::aggregate_coords(ids, n_ids, this->data, this->n_chunks,
                         pq_coord_scratch);
      pq_flash_index_utils::pq_dist_lookup(pq_coord_scratch, n_ids, this->n_chunks, pq_dists,
                       dists_out);
    };

    auto compute_extact_dists_and_push = [&](const char* node_buf, const unsigned id) -> float {
      float *node_fp_coords_copy = (float*) data_buf;
      uint8_t* node_sq_data = (uint8_t*)node_buf;
      /* memcpy(node_fp_coords_copy, node_buf, disk_bytes_per_point); */
      for(uint32_t i = 0; i < aligned_dim; i+=8){
        __m128i sq_vec = _mm_loadl_epi64((__m128i*) (node_sq_data + i));
        __m256 frac_vec = _mm256_load_ps(this->frac + i);
        __m256 min_vec = _mm256_load_ps(this->mins + i);
        __m256i sq_vec_i = _mm256_cvtepu8_epi32(sq_vec);
        __m256 sq_vec_f = _mm256_cvtepi32_ps(sq_vec_i);
        sq_vec_f = _mm256_mul_ps(sq_vec_f, frac_vec);
        sq_vec_f = _mm256_add_ps(sq_vec_f, min_vec);
        _mm256_storeu_ps(node_fp_coords_copy + i, sq_vec_f);
      }
      float cur_expanded_dist = dist_cmp_float->compare(query, node_fp_coords_copy,
                                            (unsigned) aligned_dim);
      full_retset.push_back(Neighbor(id, cur_expanded_dist, true));
      return cur_expanded_dist;
    };

    auto compute_and_push_nbrs = [&](const char *node_buf, unsigned& nk) {
      unsigned *node_nbrs = OFFSET_TO_NODE_NHOOD(node_buf);
      unsigned nnbrs = *(node_nbrs++);
      unsigned nbors_cand_size = 0;
      for (unsigned m = 0; m < nnbrs; ++m) {
        if (visited.find(node_nbrs[m]) == visited.end()) {
          node_nbrs[nbors_cand_size++] = node_nbrs[m];
          visited.insert(node_nbrs[m]);
        }
      }
      if (nbors_cand_size) {
        compute_pq_dists(node_nbrs, nbors_cand_size, dist_scratch);
        for (unsigned m = 0; m < nbors_cand_size; ++m) {
          const int nbor_id = node_nbrs[m];
          const float nbor_dist = dist_scratch[m];
          if (stats != nullptr) {
            stats->n_cmps++;
          }
          if (nbor_dist >= retset[cur_list_size - 1].distance &&
              (cur_list_size == l_search))
            continue;
          Neighbor nn(nbor_id, nbor_dist, true);
          // Return position in sorted list where nn inserted
          auto     r = InsertIntoPool(retset.data(), cur_list_size, nn);
          if (cur_list_size < l_search) ++cur_list_size;
          // nk logs the best position in the retset that was updated due to neighbors of n.
          if (r < nk) nk = r;
        }
      }
    };

    auto compute_and_add_to_retset = [&](const unsigned *node_ids, const _u64 n_ids) {
      compute_pq_dists(node_ids, n_ids, dist_scratch);
      for (_u64 i = 0; i < n_ids; ++i) {
        retset[cur_list_size].id = node_ids[i];
        retset[cur_list_size].distance = dist_scratch[i];
        retset[cur_list_size++].flag = true;
        visited.insert(node_ids[i]);
      }
    };

    if (mem_L) {
      std::vector<unsigned> mem_tags(mem_L);
      std::vector<float> mem_dists(mem_L);
      std::vector<T*> res = std::vector<T*>();
      mem_index_->search_with_tags((T*)query, mem_L, mem_L, mem_tags.data(), mem_dists.data(), nullptr, res);
      compute_and_add_to_retset(mem_tags.data(), std::min((unsigned)mem_L,(unsigned)l_search));
    } else {
      compute_and_add_to_retset(&best_medoid, 1);
    }

    std::sort(retset.begin(), retset.begin() + cur_list_size);

    unsigned num_ios = 0;
    unsigned k = 0;

    // cleared every iteration
    std::vector<unsigned> frontier;
    frontier.reserve(2 * beam_width);
    std::vector<std::pair<unsigned, char *>> frontier_nhoods;
    frontier_nhoods.reserve(2 * beam_width);
    std::vector<AlignedRead> frontier_read_reqs;
    frontier_read_reqs.reserve(2 * beam_width);
    std::vector<std::pair<unsigned, std::pair<unsigned, unsigned *>>>
        cached_nhoods;
    cached_nhoods.reserve(2 * beam_width);

    // 优化：直接存储页面指针，避免memcpy
    struct PageInfo {
      unsigned id;
      char* sector_buf;
    };
    std::vector<PageInfo> last_pages;
    last_pages.reserve(2 * beam_width);
    int n_ops = 0;

    while (k < cur_list_size && num_ios < io_limit) {
      unsigned nk = cur_list_size;
      // clear iteration state
      frontier.clear();
      frontier_nhoods.clear();
      frontier_read_reqs.clear();
      cached_nhoods.clear();
      sector_scratch_idx = 0;

#ifdef DYN_BEAM_WIDTH
      // Update beam width using static policy based on search progress
      constexpr _u32 kBeamWidths[] = {4, 4, 8, 8, 16, 16, 24, 24, 32};
      cur_beam_width = kBeamWidths[std::min(max_marker / 5, 8u)];
      // Ensure we don't exceed the maximum beam width
      cur_beam_width = std::min(cur_beam_width, (_u32)beam_width);
#endif

      // find new beam
      _u32 marker = k;
      _u32 num_seen = 0;

      // distribute cache and disk-read nodes
      while (marker < cur_list_size && frontier.size() < cur_beam_width &&
             num_seen < cur_beam_width) {
        const unsigned pid = id2page_[retset[marker].id];
        if (page_visited.find(pid) == page_visited.end() && retset[marker].flag) {
          num_seen++;
          auto iter = nhood_cache.find(retset[marker].id);
          if (iter != nhood_cache.end()) {
            cached_nhoods.push_back(
                std::make_pair(retset[marker].id, iter->second));
            if (stats != nullptr) {
              stats->n_cache_hits++;
            }
          } else {
            frontier.push_back(retset[marker].id);
            page_visited.insert(pid);
          }
          retset[marker].flag = false;
        }
        marker++;
      }

      // Update max_marker for progress tracking
      max_marker = std::max(max_marker, marker);

      // read nhoods of frontier ids
      if (!frontier.empty()) {
        if (stats != nullptr)
          stats->n_hops++;
        for (_u64 i = 0; i < frontier.size(); i++) {
          auto                    id = frontier[i];
          std::pair<_u32, char *> fnhood;
          fnhood.first = id;
          fnhood.second = sector_scratch + sector_scratch_idx * SECTOR_LEN;
          sector_scratch_idx++;
          frontier_nhoods.push_back(fnhood);
          frontier_read_reqs.emplace_back(
              (static_cast<_u64>(id2page_[id]+1)) * SECTOR_LEN, SECTOR_LEN,
              fnhood.second);
          if (stats != nullptr) {
            stats->n_4k++;
            stats->n_ios++;
          }
          num_ios++;
        }
        n_ops = reader->submit_reqs(frontier_read_reqs, ctx);
        if (this->count_visited_nodes) {
#pragma omp critical
          {
            auto &cnt = this->node_visit_counter[retset[marker].id].second;
            ++cnt;
          }
        }
      }

      // compute remaining nodes in the pages that are fetched in the previous round
      for (size_t i = 0; i < last_pages.size(); ++i) {
        const unsigned last_io_id = last_pages[i].id;
        char    *sector_buf = last_pages[i].sector_buf;
        const unsigned pid = id2page_[last_io_id];
        const unsigned p_size = gp_layout_[pid].size();
        // minus one for the vector that is computed previously
        unsigned vis_size = use_ratio * (p_size - 1);
        std::vector<std::pair<float, const char*>> vis_cand;
        vis_cand.reserve(p_size);

        // compute exact distances of the vectors within the page
        for (unsigned j = 0; j < p_size; ++j) {
          const unsigned id = gp_layout_[pid][j];
          if (id == last_io_id) continue;
          const char* node_buf = sector_buf + j * max_node_len;
          float dist = compute_extact_dists_and_push(node_buf, id);
          vis_cand.emplace_back(dist, node_buf);
        }
        if (vis_size && vis_size != p_size) {
          std::sort(vis_cand.begin(), vis_cand.end());
        }

        // compute PQ distances for neighbours of the vectors in the page
        for (unsigned j = 0; j < vis_size; ++j) {
          compute_and_push_nbrs(vis_cand[j].second, nk);
        }
      }
      last_pages.clear();

      // process cached nhoods
      for (auto &cached_nhood : cached_nhoods) {
        auto id = cached_nhood.first;
        auto  global_cache_iter = coord_cache.find(cached_nhood.first);
        T *   node_fp_coords_copy = global_cache_iter->second;
        unsigned nnr = cached_nhood.second.first;
        unsigned* cnhood = cached_nhood.second.second;
        char node_buf[max_node_len];
        memcpy(node_buf, node_fp_coords_copy, disk_bytes_per_point);
        memcpy((node_buf + disk_bytes_per_point), &nnr, sizeof(unsigned));
        memcpy((node_buf + disk_bytes_per_point + sizeof(unsigned)), cnhood, sizeof(unsigned)*nnr);
        compute_extact_dists_and_push(node_buf, id);
        compute_and_push_nbrs(node_buf, nk);
      }

      // get last submitted io results, blocking
      if (!frontier.empty()) {
        reader->get_events(ctx, n_ops);
      }

      // compute only the desired vectors in the pages - one for each page
      // postpone remaining vectors to the next round
      for (auto &frontier_nhood : frontier_nhoods) {
        char *sector_buf = frontier_nhood.second;
        unsigned pid = id2page_[frontier_nhood.first];
        // 优化：直接存储页面指针，避免memcpy
        last_pages.push_back({frontier_nhood.first, sector_buf});

        for (unsigned j = 0; j < gp_layout_[pid].size(); ++j) {
          unsigned id = gp_layout_[pid][j];
          if (id == frontier_nhood.first) {
            char *node_buf = sector_buf + j * max_node_len;
            compute_extact_dists_and_push(node_buf, id);
            compute_and_push_nbrs(node_buf, nk);
          }
        }
      }

      // update best inserted position
      if (nk <= k)
        k = nk;  // k is the best position in retset updated in this round.
      else
        ++k;
    }

    // re-sort by distance
    std::sort(full_retset.begin(), full_retset.end(),
              [](const Neighbor &left, const Neighbor &right) {
                return left.distance < right.distance;
              });

    // copy k_search values
    _u64 t = 0;
    for (_u64 i = 0; i < full_retset.size() && t < k_search; i++) {
      if(i > 0 && full_retset[i].id == full_retset[i-1].id){
        continue;
      }
      indices[t] = full_retset[i].id;
      if (distances != nullptr) {
        distances[t] = full_retset[i].distance;
        if (metric == diskann::Metric::INNER_PRODUCT) {
          // flip the sign to convert min to max
          distances[t] = (-distances[t]);
          // rescale to revert back to original norms (cancelling the effect of
          // base and query pre-processing)
          if (max_base_norm != 0)
            distances[t] *= (max_base_norm * query_norm);
        }
      }
      t++;
    }

    if (t < k_search) {
      diskann::cerr << "The number of unique ids is less than topk" << std::endl;
      exit(1);
    }

    this->thread_data.push(data);
    this->thread_data.push_notify_all();

    if (stats != nullptr) {
      stats->total_us = (double) query_timer.elapsed();
    }
  }
  template class PQFlashIndex<_u8>;
  template class PQFlashIndex<_s8>;
  template class PQFlashIndex<float>;

} // namespace diskann
