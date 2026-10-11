// Hybrid MPI + OpenMP traffic control analytics engine, with two selectable inter-rank scheduling strategies:

//   STATIC  mode: the file is split into world_size contiguous ranges once, up front (the same approach used in the pure-MPI version
//                 in the previous assignment). Every rank processes exactly one fixed range and there is no further communication until
//                 the final reduction.

//   DYNAMIC mode: rank 0 acts as a dedicated DISPATCHER and never processes data itself. The file is divided into many small, equal
//                 line-count CHUNKS (far smaller than a static per-rank share). Worker ranks (1...world_size-1) request one chunk
//                 at a time from the dispatcher, process it, and immediately ask for another -- a classic master-worker /
//                 self-scheduling pattern. A rank that happens to run faster (less local contention for physical CPU time,
//                 nothing else competing on its VM at that moment, etc.) simply completes more chunks over the run; a slower rank
//                 completes fewer. This is NEW relative to both earlier assignments in this unit (threaded producer-consumer, and
//                 pure-MPI static block partition): the amount of work each process does is decided at RUN TIME by observed progress,
//                 not fixed in advance by a formula.

// Within EVERY rank, in both modes, the actual parsing and aggregation of whatever lines that rank is responsible for is parallelized across
// OpenMP threads: lines are first read sequentially into memory (file I/O is inherently sequential), then parsed and folded into THREAD-LOCAL hash
// maps in parallel (#pragma omp parallel for), and finally the thread-local maps are merged into the rank's own cumulative map by a single thread.
// OpenMP's built-in `reduction` clause only supports scalar/arithmetic types, not arbitrary STL containers, so this manual
// "thread-local-array-then-serial-merge" pattern is the standard technique for reducing into a hash map without per-insert locking (a per-insert
// `#pragma omp critical` would serialize every single map update and defeat the purpose of threading at all).

// Build:
//   mpic++ -O3 -std=c++17 -fopenmp -o traffic_hybrid traffic_hybrid.cpp

// Run (static mode, example):
//   OMP_NUM_THREADS=4 mpirun -f hostfile -n 4 ./traffic_hybrid large.txt 5 static

// Run (dynamic mode, example, 20000-line chunks):
//   OMP_NUM_THREADS=4 mpirun -f hostfile -n 4 ./traffic_hybrid large.txt 5 dynamic 20000

#include <mpi.h>
#include <omp.h>
#include <algorithm>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

// ---------------------------------------------------------------------------
// Shared types and parsing
// ---------------------------------------------------------------------------
struct TrafficRecord {
    long long timestamp_epoch = 0;
    int light_id = 0;
    int cars = 0;
};

using HourMap = std::unordered_map<long long, std::unordered_map<int, long long>>;

bool parse_line(const std::string& line, TrafficRecord& out) {
    if (line.empty()) return false;
    std::size_t p1 = line.find(',');
    if (p1 == std::string::npos) return false;
    std::size_t p2 = line.find(',', p1 + 1);
    if (p2 == std::string::npos) return false;
    try {
        out.timestamp_epoch = std::stoll(line.substr(0, p1));
        out.light_id = std::stoi(line.substr(p1 + 1, p2 - p1 - 1));
        out.cars = std::stoi(line.substr(p2 + 1));
    } catch (const std::exception&) {
        return false;
    }
    return true;
}

long long count_lines(const std::string& filepath, int rank) {
    std::ifstream in(filepath);
    if (!in.is_open()) {
        std::cerr << "[rank " << rank << "] Cannot open input file: " << filepath << "\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    long long count = 0;
    std::string line;
    while (std::getline(in, line)) ++count;
    return count;
}

// Reads lines [start_line, end_line) from the file sequentially (I/O is
// inherently sequential) into an in-memory buffer, so the CPU-bound parse
// step that follows can be split across OpenMP threads by index.
std::vector<std::string> read_line_range(const std::string& filepath, long long start_line, long long end_line) {
    std::vector<std::string> buf;
    buf.reserve(static_cast<std::size_t>(end_line - start_line));
    std::ifstream in(filepath);
    std::string line;
    long long cur = 0;
    while (cur < start_line && std::getline(in, line)) ++cur;
    while (cur < end_line && std::getline(in, line)) {
        buf.push_back(line);
        ++cur;
    }
    return buf;
}

// Parses an in-memory batch of lines in parallel across OpenMP threads.
// Each thread accumulates into its OWN hash map (thread_maps[tid]) with no
// locking, then a single thread merges all thread-local maps into
// out_map/out_malformed after the parallel region ends. This is the
// manual-reduction pattern described at the top of this file.
void process_lines_openmp(const std::vector<std::string>& lines, HourMap& out_map, long long& out_malformed) {
    int max_threads = omp_get_max_threads();
    std::vector<HourMap> thread_maps(static_cast<std::size_t>(max_threads));
    std::vector<long long> thread_malformed(static_cast<std::size_t>(max_threads), 0);

    #pragma omp parallel for schedule(dynamic, 2048)
    for (long long i = 0; i < static_cast<long long>(lines.size()); ++i) {
        int tid = omp_get_thread_num();
        TrafficRecord rec;
        if (parse_line(lines[static_cast<std::size_t>(i)], rec)) {
            long long hour = rec.timestamp_epoch / 3600;
            thread_maps[static_cast<std::size_t>(tid)][hour][rec.light_id] += rec.cars;
        } else {
            ++thread_malformed[static_cast<std::size_t>(tid)];
        }
    }

    // Serial merge -- cheap relative to the parallel parse step above, since
    // it touches only distinct keys, not every input line.
    for (int t = 0; t < max_threads; ++t) {
        out_malformed += thread_malformed[static_cast<std::size_t>(t)];
        for (const auto& [hour, lights] : thread_maps[static_cast<std::size_t>(t)]) {
            for (const auto& [light_id, cars] : lights) {
                out_map[hour][light_id] += cars;
            }
        }
    }
}

std::vector<long long> flatten(const HourMap& m) {
    std::vector<long long> flat;
    for (const auto& [hour, lights] : m) {
        for (const auto& [light_id, cars] : lights) {
            flat.push_back(hour);
            flat.push_back(static_cast<long long>(light_id));
            flat.push_back(cars);
        }
    }
    return flat;
}

void merge_flat_into(const std::vector<long long>& flat, HourMap& into) {
    for (std::size_t i = 0; i + 2 < flat.size(); i += 3) {
        into[flat[i]][static_cast<int>(flat[i + 1])] += flat[i + 2];
    }
}

void print_top_n(const HourMap& global_map, int top_n) {
    std::vector<long long> hours;
    hours.reserve(global_map.size());
    for (const auto& [hour, _] : global_map) hours.push_back(hour);
    std::sort(hours.begin(), hours.end());

    std::cout << "=== Top " << top_n << " congested traffic lights per hour ===\n";
    for (long long hour : hours) {
        std::vector<std::pair<int, long long>> lights(global_map.at(hour).begin(), global_map.at(hour).end());
        std::sort(lights.begin(), lights.end(),
                  [](const auto& a, const auto& b) { return a.second > b.second; });
        std::cout << "Hour Block: " << hour << " (Timestamp window: ["
                  << (hour * 3600) << " to " << ((hour + 1) * 3600 - 1) << "])\n";
        int shown = 0;
        for (const auto& [light_id, cars] : lights) {
            if (shown >= top_n) break;
            std::cout << "  Position " << (shown + 1) << " -> Traffic Light ID: "
                      << light_id << " | Total Cars: " << cars << "\n";
            ++shown;
        }
        std::cout << "\n";
    }
}

// ---------------------------------------------------------------------------
// STATIC mode: fixed, even up-front partition (same scheme as the pure-MPI
// assignment), with OpenMP parallelizing each rank's own parse step.
// ---------------------------------------------------------------------------
void run_static(const std::string& input_file, int top_n, int rank, int world_size, char proc_name[]) {
    MPI_Barrier(MPI_COMM_WORLD);
    double t_start = MPI_Wtime();

    long long total_lines = count_lines(input_file, rank);
    long long base = total_lines / world_size;
    long long remainder = total_lines % world_size;
    long long start_line = static_cast<long long>(rank) * base + std::min<long long>(rank, remainder);
    long long lines_assigned = base + (rank < remainder ? 1 : 0);
    long long end_line = start_line + lines_assigned;

    std::vector<std::string> lines = read_line_range(input_file, start_line, end_line);
    HourMap local_map;
    long long malformed = 0;
    process_lines_openmp(lines, local_map, malformed);

    std::vector<long long> local_flat = flatten(local_map);
    int local_count = static_cast<int>(local_flat.size());
    long long distinct_keys = local_count / 3;

    std::vector<int> all_lines(world_size), all_keys(world_size), all_malformed(world_size);
    std::vector<char> all_names(static_cast<std::size_t>(world_size) * MPI_MAX_PROCESSOR_NAME);
    int li = static_cast<int>(lines_assigned), ki = static_cast<int>(distinct_keys), mi = static_cast<int>(malformed);
    MPI_Gather(&li, 1, MPI_INT, all_lines.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Gather(&ki, 1, MPI_INT, all_keys.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Gather(&mi, 1, MPI_INT, all_malformed.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Gather(proc_name, MPI_MAX_PROCESSOR_NAME, MPI_CHAR, all_names.data(), MPI_MAX_PROCESSOR_NAME, MPI_CHAR, 0, MPI_COMM_WORLD);

    std::vector<int> recv_counts, displs;
    if (rank == 0) recv_counts.resize(world_size);
    MPI_Gather(&local_count, 1, MPI_INT, recv_counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

    std::vector<long long> all_flat;
    if (rank == 0) {
        displs.resize(world_size);
        long long total = 0;
        for (int i = 0; i < world_size; ++i) { displs[i] = static_cast<int>(total); total += recv_counts[i]; }
        all_flat.resize(static_cast<std::size_t>(total));
    }
    MPI_Gatherv(local_flat.data(), local_count, MPI_LONG_LONG,
                rank == 0 ? all_flat.data() : nullptr,
                rank == 0 ? recv_counts.data() : nullptr,
                rank == 0 ? displs.data() : nullptr,
                MPI_LONG_LONG, 0, MPI_COMM_WORLD);

    double elapsed = MPI_Wtime() - t_start, max_elapsed = 0.0;
    MPI_Reduce(&elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        std::cout << "Mode: STATIC | world_size=" << world_size
                  << " | OMP threads/rank=" << omp_get_max_threads()
                  << " | total_lines=" << total_lines << "\n\n";
        std::cout << "=== Partition balance across ranks (static) ===\n";
        std::cout << "Rank | Host                 | Lines assigned | Distinct keys | Malformed\n";
        for (int r = 0; r < world_size; ++r) {
            std::string host(all_names.data() + static_cast<std::size_t>(r) * MPI_MAX_PROCESSOR_NAME);
            std::cout << "  " << r << "  | " << host;
            for (std::size_t pad = host.size(); pad < 20; ++pad) std::cout << ' ';
            std::cout << " | " << all_lines[r];
            for (int pad = std::to_string(all_lines[r]).size(); pad < 15; ++pad) std::cout << ' ';
            std::cout << " | " << all_keys[r];
            for (int pad = std::to_string(all_keys[r]).size(); pad < 14; ++pad) std::cout << ' ';
            std::cout << " | " << all_malformed[r] << "\n";
        }
        std::cout << "\n";

        HourMap global_map;
        merge_flat_into(all_flat, global_map);
        print_top_n(global_map, top_n);

        long long total_malformed = 0;
        for (int m : all_malformed) total_malformed += m;
        std::cout << "--------------------------------------------\n";
        std::cout << "Total malformed/skipped lines: " << total_malformed << "\n";
        std::cout << "Simulation execution completed in: " << (max_elapsed * 1000.0) << " ms\n";
        std::cout << "--------------------------------------------\n";
    }
}

// ---------------------------------------------------------------------------
// DYNAMIC mode: master-worker self-scheduling across MPI ranks.
// ---------------------------------------------------------------------------
constexpr int TAG_WORK_REQUEST = 1;
constexpr int TAG_ASSIGNMENT   = 2; // payload: [start_line, end_line]; [-1,-1] = terminate
constexpr int TAG_RESULT_META  = 3; // payload: [lines_processed, distinct_keys, malformed, chunks_completed]
constexpr int TAG_RESULT_DATA  = 4; // payload: flattened (hour, light_id, cars) triples
constexpr int TAG_RESULT_HOST  = 5; // payload: processor name

void dispatcher_loop(long long total_lines, long long chunk_size, int world_size) {
    long long num_chunks = (total_lines + chunk_size - 1) / chunk_size;
    long long next_chunk = 0;
    int active_workers = world_size - 1;

    while (active_workers > 0) {
        MPI_Status status;
        int dummy;
        MPI_Recv(&dummy, 1, MPI_INT, MPI_ANY_SOURCE, TAG_WORK_REQUEST, MPI_COMM_WORLD, &status);
        int dest = status.MPI_SOURCE;

        long long range[2];
        if (next_chunk < num_chunks) {
            range[0] = next_chunk * chunk_size;
            range[1] = std::min<long long>(range[0] + chunk_size, total_lines);
            ++next_chunk;
            MPI_Send(range, 2, MPI_LONG_LONG, dest, TAG_ASSIGNMENT, MPI_COMM_WORLD);
        } else {
            range[0] = -1; range[1] = -1;
            MPI_Send(range, 2, MPI_LONG_LONG, dest, TAG_ASSIGNMENT, MPI_COMM_WORLD);
            --active_workers;
        }
    }
}

void worker_loop(const std::string& input_file, long long /*total_lines*/, HourMap& local_map,
                  long long& lines_processed, long long& malformed, long long& chunks_completed) {
    while (true) {
        int dummy = 0;
        MPI_Send(&dummy, 1, MPI_INT, 0, TAG_WORK_REQUEST, MPI_COMM_WORLD);

        long long range[2];
        MPI_Recv(range, 2, MPI_LONG_LONG, 0, TAG_ASSIGNMENT, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        if (range[0] == -1) break; // no more chunks -- dispatcher told us to stop

        std::vector<std::string> lines = read_line_range(input_file, range[0], range[1]);
        process_lines_openmp(lines, local_map, malformed);
        lines_processed += static_cast<long long>(lines.size());
        ++chunks_completed;
    }
}

void run_dynamic(const std::string& input_file, int top_n, long long chunk_size, int rank, int world_size, char proc_name[]) {
    if (world_size < 2) {
        // No separate worker ranks exist. Rank 0 degenerates to processing
        // every chunk itself -- still exercises OpenMP and chunking logic,
        // and gives a valid (if non-distributed) single-process baseline.
        MPI_Barrier(MPI_COMM_WORLD);
        double t_start = MPI_Wtime();

        long long total_lines = count_lines(input_file, rank);
        long long num_chunks = (total_lines + chunk_size - 1) / chunk_size;
        HourMap local_map;
        long long malformed = 0, lines_processed = 0;
        for (long long c = 0; c < num_chunks; ++c) {
            long long s = c * chunk_size, e = std::min<long long>(s + chunk_size, total_lines);
            std::vector<std::string> lines = read_line_range(input_file, s, e);
            process_lines_openmp(lines, local_map, malformed);
            lines_processed += static_cast<long long>(lines.size());
        }

        double elapsed = MPI_Wtime() - t_start;
        std::cout << "Mode: DYNAMIC (single-rank fallback, no separate workers) | OMP threads="
                  << omp_get_max_threads() << " | total_lines=" << total_lines
                  << " | chunks=" << num_chunks << "\n\n";
        print_top_n(local_map, top_n);
        std::cout << "--------------------------------------------\n";
        std::cout << "Total malformed/skipped lines: " << malformed << "\n";
        std::cout << "Simulation execution completed in: " << (elapsed * 1000.0) << " ms\n";
        std::cout << "--------------------------------------------\n";
        return;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t_start = MPI_Wtime();

    long long total_lines = count_lines(input_file, rank);

    if (rank == 0) {
        dispatcher_loop(total_lines, chunk_size, world_size);

        HourMap global_map;
        long long total_malformed = 0;
        std::vector<long long> all_lines(world_size, 0), all_keys(world_size, 0), all_malformed_v(world_size, 0), all_chunks(world_size, 0);
        std::vector<std::string> all_hosts(world_size, "(dispatcher)");

        for (int i = 0; i < world_size - 1; ++i) {
            MPI_Status status;
            long long meta[4];
            MPI_Recv(meta, 4, MPI_LONG_LONG, MPI_ANY_SOURCE, TAG_RESULT_META, MPI_COMM_WORLD, &status);
            int src = status.MPI_SOURCE;

            long long data_len = meta[1] * 3;
            std::vector<long long> flat(static_cast<std::size_t>(data_len));
            MPI_Recv(flat.data(), static_cast<int>(data_len), MPI_LONG_LONG, src, TAG_RESULT_DATA, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            char host[MPI_MAX_PROCESSOR_NAME];
            MPI_Recv(host, MPI_MAX_PROCESSOR_NAME, MPI_CHAR, src, TAG_RESULT_HOST, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            all_lines[src] = meta[0];
            all_keys[src] = meta[1];
            all_malformed_v[src] = meta[2];
            all_chunks[src] = meta[3];
            all_hosts[src] = std::string(host);
            total_malformed += meta[2];

            merge_flat_into(flat, global_map);
        }

        double elapsed = MPI_Wtime() - t_start;

        long long num_chunks = (total_lines + chunk_size - 1) / chunk_size;
        std::cout << "Mode: DYNAMIC | world_size=" << world_size
                  << " | OMP threads/rank=" << omp_get_max_threads()
                  << " | total_lines=" << total_lines
                  << " | chunk_size=" << chunk_size
                  << " | num_chunks=" << num_chunks << "\n\n";
        std::cout << "=== Chunks completed per worker rank (dynamic self-scheduling) ===\n";
        std::cout << "Rank | Host                 | Chunks completed | Lines processed | Distinct keys\n";
        for (int r = 1; r < world_size; ++r) {
            std::cout << "  " << r << "  | " << all_hosts[r];
            for (std::size_t pad = all_hosts[r].size(); pad < 20; ++pad) std::cout << ' ';
            std::cout << " | " << all_chunks[r];
            for (int pad = std::to_string(all_chunks[r]).size(); pad < 17; ++pad) std::cout << ' ';
            std::cout << " | " << all_lines[r];
            for (int pad = std::to_string(all_lines[r]).size(); pad < 16; ++pad) std::cout << ' ';
            std::cout << " | " << all_keys[r] << "\n";
        }
        std::cout << "(Rank 0 was a dedicated dispatcher and processed no chunks itself.)\n\n";

        print_top_n(global_map, top_n);
        std::cout << "--------------------------------------------\n";
        std::cout << "Total malformed/skipped lines: " << total_malformed << "\n";
        std::cout << "Simulation execution completed in: " << (elapsed * 1000.0) << " ms\n";
        std::cout << "--------------------------------------------\n";

    } else {
        HourMap local_map;
        long long lines_processed = 0, malformed = 0, chunks_completed = 0;
        worker_loop(input_file, total_lines, local_map, lines_processed, malformed, chunks_completed);

        std::vector<long long> flat = flatten(local_map);
        long long meta[4] = { lines_processed, static_cast<long long>(flat.size() / 3), malformed, chunks_completed };
        MPI_Send(meta, 4, MPI_LONG_LONG, 0, TAG_RESULT_META, MPI_COMM_WORLD);
        MPI_Send(flat.data(), static_cast<int>(flat.size()), MPI_LONG_LONG, 0, TAG_RESULT_DATA, MPI_COMM_WORLD);
        MPI_Send(proc_name, MPI_MAX_PROCESSOR_NAME, MPI_CHAR, 0, TAG_RESULT_HOST, MPI_COMM_WORLD);
        // Worker's job is done once its final result has been sent; it can
        // proceed straight to MPI_Finalize without waiting on rank 0, since
        // nothing further depends on this worker's participation.
    }
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    int rank = 0, world_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    if (argc < 4) {
        if (rank == 0) {
            std::cerr << "Usage: OMP_NUM_THREADS=<n> mpirun -f hostfile -n <procs> " << argv[0]
                      << " <input_file> <top_N> <static|dynamic> [chunk_size]\n";
        }
        MPI_Finalize();
        return 1;
    }

    const std::string input_file = argv[1];
    const int top_n = std::max(1, std::atoi(argv[2]));
    const std::string mode = argv[3];
    const long long chunk_size = (argc >= 5) ? std::max(1LL, std::atoll(argv[4])) : 20000LL;

    char proc_name[MPI_MAX_PROCESSOR_NAME];
    int proc_name_len = 0;
    MPI_Get_processor_name(proc_name, &proc_name_len);

    if (mode == "static") {
        run_static(input_file, top_n, rank, world_size, proc_name);
    } else if (mode == "dynamic") {
        run_dynamic(input_file, top_n, chunk_size, rank, world_size, proc_name);
    } else {
        if (rank == 0) std::cerr << "Unknown mode '" << mode << "': expected 'static' or 'dynamic'\n";
        MPI_Finalize();
        return 1;
    }

    MPI_Finalize();
    return 0;
}