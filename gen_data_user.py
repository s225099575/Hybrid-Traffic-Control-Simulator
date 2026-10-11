import random

# Fixed seed so regenerating a file for a re-run produces identical data.
random.seed(42)

def generate_data(filename, num_records, num_lights=50):
    start_time = 1718000000  # Arbitrary base unix baseline epoch
    with open(filename, 'w') as f:
        for _ in range(num_records):
            # 12 intervals per hour, spread across a fixed 6-hour window.
            # NOTE: all files (small/medium/large/stress) intentionally share
            # this same 6-hour window regardless of record count -- larger
            # files pack more records into the same hourly buckets rather
            # than spreading across more hours. This is a deliberate stress
            # choice: it concentrates contention on the same small set of
            # TrafficAggregator map entries as record count grows.
            timestamp = start_time + random.randint(0, 5) * 3600 + random.choice([i * 300 for i in range(12)])
            light_id = random.randint(1, num_lights)
            cars = random.randint(5, 120)
            f.write(f"{timestamp},{light_id},{cars}\n")
    print(f"File successfully created: {filename} with {num_records} items.")

if __name__ == "__main__":
    # Generating the formal test workload files
    generate_data("small.txt", 10000, num_lights=20)
    generate_data("medium.txt", 200000, num_lights=50)
    generate_data("large.txt", 3000000, num_lights=100)

    # A large file deliberately constructed for stress testing.
    generate_data("stress.txt", 15000000, num_lights=1000)
