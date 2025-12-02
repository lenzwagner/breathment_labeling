from Utils.text_saver import *
from Utils.result_comparator import compare_result_files
import time

# --- 1. Input Data ---

pi = {(1, 1): 0.0, (1, 2): 0.0, (1, 3): 0.0, (1, 4): 0.0, (1, 5): 0.0, (1, 6): -2.0, (1, 7): -2.0, (1, 8): 0.0, (1, 9): 0.0, (1, 10): 0.0, (1, 11): 0.0, (1, 12): 0.0, (1, 13): 0.0, (1, 14): 0.0, (1, 15): 0.0, (1, 16): -27.0, (1, 17): 0.0, (1, 18): 0.0, (1, 19): 0.0, (1, 20): 0.0, (1, 21): -27.0, (1, 22): 0.0, (1, 23): 0.0, (1, 24): 0.0, (1, 25): 0.0, (1, 26): 0.0, (1, 27): 0.0, (1, 28): 0.0, (1, 29): 0.0, (1, 30): 0.0, (1, 31): 0.0, (1, 32): 0.0, (1, 33): 0.0, (1, 34): 0.0, (1, 35): 0.0, (1, 36): 0.0, (1, 37): 0.0, (1, 38): 0.0, (1, 39): 0.0, (1, 40): 0.0, (1, 41): 0.0, (1, 42): 0.0, (2, 1): 0.0, (2, 2): -2.0, (2, 3): 0.0, (2, 4): 0.0, (2, 5): 0.0, (2, 6): 0.0, (2, 7): -1.0, (2, 8): -1.0, (2, 9): -1.0, (2, 10): 0.0, (2, 11): 0.0, (2, 12): 0.0, (2, 13): 0.0, (2, 14): 0.0, (2, 15): 0.0, (2, 16): 0.0, (2, 17): 0.0, (2, 18): 0.0, (2, 19): -16.0, (2, 20): 0.0, (2, 21): 0.0, (2, 22): -27.0, (2, 23): 0.0, (2, 24): 0.0, (2, 25): 0.0, (2, 26): 0.0, (2, 27): 0.0, (2, 28): 0.0, (2, 29): 0.0, (2, 30): 0.0, (2, 31): 0.0, (2, 32): 0.0, (2, 33): 0.0, (2, 34): 0.0, (2, 35): 0.0, (2, 36): 0.0, (2, 37): 0.0, (2, 38): 0.0, (2, 39): 0.0, (2, 40): 0.0, (2, 41): 0.0, (2, 42): 0.0, (3, 1): 0.0, (3, 2): -2.0, (3, 3): 0.0, (3, 4): 0.0, (3, 5): 0.0, (3, 6): -1.0, (3, 7): 0.0, (3, 8): 0.0, (3, 9): -2.0, (3, 10): -1.0, (3, 11): 0.0, (3, 12): 0.0, (3, 13): 0.0, (3, 14): 0.0, (3, 15): -16.0, (3, 16): -27.0, (3, 17): 0.0, (3, 18): 0.0, (3, 19): 0.0, (3, 20): 0.0, (3, 21): 0.0, (3, 22): 0.0, (3, 23): -16.0, (3, 24): 0.0, (3, 25): 0.0, (3, 26): 0.0, (3, 27): 0.0, (3, 28): 0.0, (3, 29): 0.0, (3, 30): 0.0, (3, 31): 0.0, (3, 32): 0.0, (3, 33): 0.0, (3, 34): 0.0, (3, 35): 0.0, (3, 36): 0.0, (3, 37): 0.0, (3, 38): 0.0, (3, 39): 0.0, (3, 40): 0.0, (3, 41): 0.0, (3, 42): 0.0}
gamma = {2: 12.0, 6: 27.0, 8: 16.0, 13: 0.0, 15: 0.0, 16: 0.0, 18: 0.0, 19: 0.0, 20: 27.0, 23: 16.0, 28: 0.0, 31: 27.0, 32: 0.0, 35: 0.0, 36: 0.0, 37: 0.0, 38: 8.0, 44: 0.0, 45: 0.0, 47: 0.0, 48: 9.0, 49: 0.0, 50: 10.0, 51: 0.0, 53: 0.0, 54: 10.0, 60: 2.0, 61: 16.0, 64: 9.0, 65: 0.0, 66: 0.0, 67: 0.0, 68: 0.0, 73: 16.0, 74: 0.0, 76: 0.0, 77: 0.0, 78: 0.0, 80: 16.0, 84: 10.0}

s_i = {1: 17, 2: 9, 3: 13, 4: 17, 5: 4, 6: 6, 7: 5, 8: 9, 9: 5, 10: 8, 11: 5, 12: 9, 13: 5, 14: 5, 15: 6, 16: 7, 17: 4, 18: 5, 19: 7, 20: 7, 21: 8, 22: 14, 23: 10, 24: 10, 25: 8, 26: 6, 27: 8, 28: 5, 29: 5, 30: 5, 31: 6, 32: 4, 33: 4, 34: 4, 35: 3, 36: 8, 37: 4, 38: 5, 39: 10, 40: 8, 41: 9, 42: 8, 43: 9, 44: 7, 45: 4, 46: 10, 47: 2, 48: 6, 49: 7, 50: 7, 51: 6, 52: 10, 53: 3, 54: 7, 55: 6, 56: 6, 57: 7, 58: 9, 59: 7, 60: 5, 61: 4, 62: 4, 63: 2, 64: 6, 65: 10, 66: 8, 67: 3, 68: 7, 69: 5, 70: 8, 71: 4, 72: 7, 73: 6, 74: 6, 75: 3, 76: 5, 77: 7, 78: 7, 79: 3, 80: 6, 81: 4, 82: 6, 83: 11, 84: 7}
r_i = {1: -8, 2: 2, 3: -4, 4: -2, 5: -4, 6: 16, 7: -16, 8: 14, 9: -35, 10: -12, 11: -10, 12: -25, 13: 22, 14: -12, 15: 25, 16: 39, 17: -17, 18: 33, 19: 27, 20: 16, 21: -34, 22: 0, 23: 12, 24: -13, 25: -18, 26: -12, 27: -16, 28: 25, 29: -9, 30: -3, 31: 15, 32: 38, 33: -13, 34: -33, 35: 40, 36: 29, 37: 25, 38: 5, 39: -18, 40: -24, 41: -12, 42: 0, 43: -31, 44: 18, 45: 21, 46: -35, 47: 30, 48: 2, 49: 36, 50: 1, 51: 26, 52: -1, 53: 12, 54: 3, 55: -34, 56: -25, 57: -18, 58: -4, 59: -7, 60: 6, 61: 16, 62: -5, 63: -11, 64: 4, 65: 38, 66: 25, 67: 15, 68: 28, 69: -18, 70: -28, 71: -29, 72: -26, 73: 12, 74: 36, 75: -27, 76: 18, 77: 38, 78: 24, 79: -24, 80: 14, 81: -16, 82: -23, 83: -35, 84: 5}
obj_mode = {2: 1, 6: 0, 8: 0, 13: 0, 15: 0, 16: 0, 18: 0, 19: 0, 20: 0, 23: 0, 28: 0, 31: 0, 32: 0, 35: 0, 36: 0, 37: 0, 38: 1, 44: 0, 45: 0, 47: 0, 48: 1, 49: 0, 50: 1, 51: 0, 53: 0, 54: 1, 60: 0, 61: 0, 64: 1, 65: 0, 66: 0, 67: 0, 68: 0, 73: 0, 74: 0, 76: 0, 77: 0, 78: 0, 80: 0, 84: 1}


# Parameter
MS = 5
MIN_MS = 2
MAX_TIME = 42
WORKERS = [1, 2, 3]

# Lookup Table
theta_lookup = [0.2 + 0.01 * k for k in range(50)]
theta_lookup = [min(x, 1.0) for x in theta_lookup]


# --- 2. Helper Functions ---

def check_strict_feasibility(history, next_val, MS, MIN_MS):
    potential_sequence = history + (next_val,)
    seq_len = len(potential_sequence)

    if seq_len < MS:
        current_sum = sum(potential_sequence)
        remaining_slots = MS - seq_len
        max_possible_sum = current_sum + remaining_slots
        if max_possible_sum < MIN_MS:
            return False
        return True
    else:
        current_window = potential_sequence[-MS:]
        if sum(current_window) < MIN_MS:
            return False
        return True


def add_state_to_buckets(buckets, cost, prog, ai_count, hist, path, recipient_id, pruning_stats, dominance_mode='bucket', zeta=None, epsilon=1e-9):
    """
    Adds a state to buckets, applying dominance rules.

    buckets structure:
      - Key: (ai_count, hist) or (ai_count, hist, zeta)
      - Value: list_of_states [(cost, prog, path), ...]

    dominance_mode:
      - 'bucket': Compares only within same (ai_count, hist, zeta)
      - 'global': Compares across all buckets (ai >= ai', hist >= hist', zeta >= zeta')

    zeta: Tuple of binary deviation indicators for branch constraints (None if no constraints)
    epsilon: Tolerance for float comparisons (default 1e-9)

    TODO: Performance Optimization - Hash-based Deduplication
          Current complexity is O(bucket_size) for dominance checks.
          Consider implementing hash-based deduplication before bucket insertion:
          - state_hash = hash((ai_count, hist, round(cost, 6), round(prog, 6)))
          - Check if state_hash in seen_states -> O(1) lookup
          - Benefits: Faster duplicate detection, reduces bucket iterations
    """
    # Bucket key includes zeta if branch constraints are active
    if zeta is not None:
        bucket_key = (ai_count, hist, zeta)
    else:
        bucket_key = (ai_count, hist)

    # --- 1. GLOBAL PRUNING CHECK (Only in Global Mode) ---
    # Check if the NEW state is dominated by ANYONE
    if dominance_mode == 'global':
        is_dominated_globally = False
        dominator_global = None

        for (ai_other, hist_other), other_list in buckets.items():
            # Another bucket can only dominate if it is "better/equal" in AI & Hist
            # We need: ai_other >= ai_count AND hist_other >= hist

            # AI Check
            if ai_other < ai_count:
                continue

            # Hist Check (component-wise >=)
            if len(hist_other) != len(hist):
                continue

            hist_better = True
            for h1, h2 in zip(hist_other, hist):
                if h1 < h2: # h1 muss >= h2 sein
                    hist_better = False
                    break
            if not hist_better:
                continue

            # If we are here, the other bucket is "structurally" better or equal.
            # Now we check Cost & Prog in this bucket
            for c_old, p_old, _ in other_list:
                # Use epsilon for float comparison: c_old <= cost + eps AND p_old >= prog - eps
                if c_old <= cost + epsilon and p_old >= prog - epsilon:
                    is_dominated_globally = True
                    dominator_global = (c_old, p_old, ai_other, hist_other)
                    break

            if is_dominated_globally:
                break

        if is_dominated_globally:
            pruning_stats['dominance'] += 1
            if not pruning_stats['printed_dominance'].get(recipient_id, False):
                print(f"    [DOMINANCE GLOBAL] Recipient {recipient_id}: Pruned new state (C={cost:.2f}, P={prog:.2f}, AI={ai_count})")
                print(f"                       by (C={dominator_global[0]:.2f}, P={dominator_global[1]:.2f}, AI={dominator_global[2]})")
                pruning_stats['printed_dominance'][recipient_id] = True
            return

    # --- 2. BUCKET MANAGEMENT ---
    if bucket_key not in buckets:
        buckets[bucket_key] = []

    bucket_list = buckets[bucket_key]

    # --- 3. LOCAL DOMINANCE (Within the Bucket) ---
    # Even in Global Mode we do this to keep our own bucket clean

    # Check if new state is dominated by existing state in the SAME bucket
    is_dominated = False
    dominator = None

    for c_old, p_old, _ in bucket_list:
        # Use epsilon for float comparison: c_old <= cost + eps AND p_old >= prog - eps
        if c_old <= cost + epsilon and p_old >= prog - epsilon:
            is_dominated = True
            dominator = (c_old, p_old)
            break

    if is_dominated:
        pruning_stats['dominance'] += 1
        if not pruning_stats['printed_dominance'].get(recipient_id, False):
            print(f"    [DOMINANCE BUCKET] Recipient {recipient_id}: Pruned new state (C={cost:.2f}, P={prog:.2f})")
            print(f"                       by same bucket (C={dominator[0]:.2f}, P={dominator[1]:.2f})")
            pruning_stats['printed_dominance'][recipient_id] = True
        return

    # --- 4. CLEANUP (Only important in Global Mode, but also good locally) ---
    # Remove existing states that are dominated by the new one

    new_bucket_list = []

    for c_old, p_old, path_old in bucket_list:
        # Use epsilon for float comparison: cost <= c_old + eps AND prog >= p_old - eps
        if cost <= c_old + epsilon and prog >= p_old - epsilon:
            pruning_stats['dominance'] += 1
            # We skip print here for clarity, or only once
            continue
        new_bucket_list.append((c_old, p_old, path_old))

    new_bucket_list.append((cost, prog, path))
    buckets[bucket_key] = new_bucket_list



def generate_full_column_vector(worker_id, path_assignments, start_time, end_time, max_time, num_workers):
    vector_length = num_workers * max_time
    full_vector = [0.0] * vector_length
    worker_offset = (worker_id - 1) * max_time
    for t_idx, val in enumerate(path_assignments):
        current_time = start_time + t_idx
        global_idx = worker_offset + (current_time - 1)
        if 0 <= global_idx < vector_length:
            full_vector[global_idx] = float(val)
    return full_vector


def validate_final_column(col_data, s_req, MS, MIN_MS, theta_table):
    """
    Validates a finished column strictly on all constraints.
    Returns a list of errors (empty if everything is OK).
    """
    errors = []
    path = col_data['path_pattern']

    # 1. Check Start & End Constraint
    if path[0] != 1:
        errors.append(f"Start constraint violation: First day must be Machine (1), found {path[0]}")

    # MODIFIED: Check End Constraint
    # If it is NOT a timeout, it MUST be a 1.
    # If it is a timeout, it can also be a 0.
    is_timeout = (col_data['end'] == MAX_TIME)

    if path[-1] != 1:
        if not is_timeout:
            errors.append(f"End constraint violation: Last day must be Machine (1), found {path[-1]}")
        # In the timeout case (is_timeout == True) we implicitly allow 0 here.

    # 2. Check Service Target (s_i)
    progress = 0.0
    ai_usage = 0
    for x in path:
        if x == 1:
            progress += 1.0
        else:
            eff = theta_table[ai_usage] if ai_usage < len(theta_table) else 1.0
            progress += eff
            ai_usage += 1

    # Tolerance 1e-9
    if progress < s_req - 1e-9:
        # Special case: Timeout (end of horizon) may miss target if the model allows it.
        # We mark it here as info/error for control (as desired).
        if is_timeout:
            errors.append(f"Target NOT met (TIMEOUT case): {progress:.2f} < {s_req}")
        else:
            errors.append(f"Target NOT met: {progress:.2f} < {s_req}")

    # 3. Check Rolling Window (Strict Window-by-Window)
    if len(path) >= MS:
        for i in range(len(path) - MS + 1):
            window = path[i: i + MS]
            if sum(window) < MIN_MS:
                errors.append(
                    f"Window violation at index {i} (Days {col_data['start'] + i}-{col_data['start'] + i + MS - 1}): {window} sum={sum(window)}")

    else:
        current_sum = sum(path)
        remaining = MS - len(path)
        if current_sum + remaining < MIN_MS:
            errors.append(f"Short path violation: {path} (sum {current_sum}) cannot satisfy MIN_MS={MIN_MS}")

    return errors


# --- 3. Labeling Algorithm ---

def compute_lower_bound(current_cost, start_time, end_time, gamma_k, obj_mode):
    """
    Calculates Lower Bound for Bound Pruning.

    Assumption: Maximum productivity (only therapists with efficiency = 1.0)
    This guarantees that we don't miss any optimal solutions.
    compute_lower_bound(cost, r_k, tau, gamma_k, obj_mode)
    Returns:
        float: Minimum achievable final Reduced Cost (optimistic)
    """
    import math

    # Time Cost is fixed for the specific column length (end_time - start_time + 1)
    duration = end_time - start_time + 1
    time_cost = duration * obj_mode

    # Current cost contains the accumulated -pi values so far.
    # We assume future -pi values are 0 (optimistic, since -pi >= 0).

    lower_bound = current_cost + time_cost - gamma_k

    return lower_bound


def compute_candidate_workers(workers, r_k, tau_max, pi_dict):
    """
    Worker Dominance Pre-Elimination:
    Worker j1 dominates j2 if π_{j1,t} >= π_{j2,t} for all t in [r_k, tau_max]
    AND π_{j1,t} > π_{j2,t} for at least one t (strict dominance).
    Since π values are <= 0 (implicit costs), higher π means lower cost.
    Returns the set of non-dominated workers.
    """
    candidate_workers = []

    for j1 in workers:
        is_dominated = False

        for j2 in workers:
            if j1 == j2:
                continue

            # Check if j2 dominates j1
            # j2 dominates j1 if:
            # 1. π_{j2,t} >= π_{j1,t} for ALL t in [r_k, tau_max]
            # 2. π_{j2,t} > π_{j1,t} for at least ONE t (strict improvement)

            all_better_or_equal = True
            at_least_one_strictly_better = False

            for t in range(r_k, tau_max + 1):
                pi_j1 = pi_dict.get((j1, t), 0.0)
                pi_j2 = pi_dict.get((j2, t), 0.0)

                if pi_j2 < pi_j1:  # j2 is worse in this period
                    all_better_or_equal = False
                    break
                elif pi_j2 > pi_j1:  # j2 is strictly better in this period
                    at_least_one_strictly_better = True

            # j2 dominates j1 if it's at least as good everywhere and strictly better somewhere
            if all_better_or_equal and at_least_one_strictly_better:
                is_dominated = True
                break

        if not is_dominated:
            candidate_workers.append(j1)

    return candidate_workers


def solve_pricing_for_recipient(recipient_id, r_k, s_k, gamma_k, obj_mode, pi_dict, workers, max_time, ms, min_ms, theta_lookup,
                                use_bound_pruning=True, dominance_mode='bucket', branch_constraints=None, branching_variant='mp'):
    best_reduced_cost = float('inf')
    best_columns = []
    epsilon = 1e-9

    pruning_stats = {
        'lb': 0,
        'dominance': 0,
        'printed_dominance': {}
    }

    time_until_end = MAX_TIME - r_k + 1

    # --- DEBUG: Print Duals for Recipient 6 ---
    if recipient_id == 6:
        print(f"\n[DEBUG] Dual Values (pi) for Recipient {recipient_id} (r_k={r_k} to {MAX_TIME}):")
        for t in range(r_k, MAX_TIME + 1):
            row_vals = []
            for w in workers:
                val = pi_dict.get((w, t), 0.0)
                row_vals.append(f"W{w}:{val:.2f}")
            print(f"  t={t}: " + ", ".join(row_vals))
        print("-" * 60 + "\n")


    # Worker Dominance Pre-Elimination
    candidate_workers = compute_candidate_workers(WORKERS, r_k, MAX_TIME, pi)
    eliminated_workers = [w for w in WORKERS if w not in candidate_workers]

    # Print for each Recipient
    if eliminated_workers:
        print(f"Recipient {recipient_id:2d}: Candidate workers = {candidate_workers} (eliminated {eliminated_workers})")
    else:
        print(f"Recipient {recipient_id:2d}: Candidate workers = {candidate_workers} (no dominance)")

    # --- Parse Branch Constraints (MP Branching) ---
    # For MP branching: branch_constraints contains "original_schedule" that must be EXCLUDED
    # We need to track deviation from forbidden schedules via ζ_t vector
    # Only applies when direction="left" (left branch = forbid)
    forbidden_schedules = []
    use_branch_constraints = False

    if branch_constraints is not None:
        # branch_constraints can be a dict with multiple keys (constraint IDs)
        # Each constraint has: profile, direction, original_schedule
        for constraint_key, constraint_data in branch_constraints.items():
            # Only process if this constraint is for our recipient
            if constraint_data.get("profile") != recipient_id:
                continue

            # Only process if direction is "left" (forbid this schedule)
            if constraint_data.get("direction") != "left":
                continue

            # --- MP Branching Logic ---
            if branching_variant == 'mp':
                # Check if this is MP branching (has "original_schedule" key)
                if "original_schedule" in constraint_data:
                    use_branch_constraints = True
                    # Extract the forbidden schedule
                    # original_schedule format: {(profile, worker, interval, shift): value}
                    forbidden_schedule = {}
                    for key, val in constraint_data["original_schedule"].items():
                        # key = (profile, worker, interval, shift)
                        # We only care about (worker, interval) -> value mapping
                        profile, worker, interval, shift = key
                        forbidden_schedule[(worker, interval)] = val

                    forbidden_schedules.append({
                        "schedule": forbidden_schedule,
                        "constraint_key": constraint_key, # Keep for debugging/tracking
                        "direction": constraint_data.get("direction", "left") # Keep for debugging/tracking
                        # "bound": constraint_data.get("bound") # Not strictly needed for equality check
                    })

            # --- SP Branching Logic (Placeholder) ---
            elif branching_variant == 'sp':
                pass # Initialize empty function / logic for SP branching here

        if use_branch_constraints:
            print(f"  [MP BRANCHING] {len(forbidden_schedules)} no-good cut(s) active for recipient {recipient_id}")



    for j in candidate_workers:
        effective_min_duration = min(int(s_k), time_until_end)
        start_tau = r_k + effective_min_duration - 1

        for tau in range(start_tau, MAX_TIME + 1):
            is_timeout_scenario = (tau == MAX_TIME)

            start_cost = -pi.get((j, r_k), 0)

            # Debug status for Dominance (once per Recipient/Worker loop instance? No, per Recipient global!)
            # But we are here in the Worker loop.
            # If we want it per Recipient, we need a Dict that persists across the Worker loop.

            # We define it at the beginning of the function solve_pricing_for_recipient

            # Initialization (Bucket structure)
            # Key: (ai_count, hist, zeta), Value: List of (cost, prog, path)
            # Initialize deviation vector ζ_t for branch constraints
            num_cuts = len(forbidden_schedules)
            initial_zeta = tuple([0] * num_cuts) if use_branch_constraints else None

            current_states = {}
            # Initialize with start state
            initial_history = (1,) # First action is always 1 (Therapist)
            add_state_to_buckets(current_states, start_cost, 1.0, 0, initial_history, [1], recipient_id, pruning_stats, dominance_mode, initial_zeta, epsilon)

            # DP Loop until just before Tau
            pruned_count_total = 0  # Counter for pruned states

            for t in range(r_k + 1, tau):
                next_states = {}
                pruned_count_this_period = 0

                # Iterate over all buckets
                # Note: bucket key may include zeta if branch constraints are active
                for bucket_key, bucket_list in current_states.items():
                    # Extract components from bucket key
                    if use_branch_constraints:
                        ai_count, hist, zeta = bucket_key
                    else:
                        ai_count, hist = bucket_key
                        zeta = None

                    # Iterate over all states in the bucket
                    for cost, prog, path in bucket_list:

                        # BOUND PRUNING: Check if state is promising
                        if use_bound_pruning:
                            lb = compute_lower_bound(cost, r_k, tau, gamma_k, obj_mode)
                            if lb >= 0:
                                pruned_count_this_period += 1
                                pruned_count_total += 1
                                pruning_stats['lb'] += 1
                                continue  # State is pruned!

                        # Feasibility Check (remains)
                        remaining_steps = tau - t + 1
                        if not is_timeout_scenario:
                            if prog + remaining_steps * 1.0 < s_k - epsilon:
                                continue

                        # A: Therapist
                        if check_strict_feasibility(hist, 1, MS, MIN_MS):
                            cost_ther = cost - pi.get((j, t), 0)
                            prog_ther = prog + 1.0
                            # History Shift: Append new action + take last MS-1 elements
                            # Note: This is conceptually equivalent to LaTeX spec (prepend + take first),
                            # just with reversed indexing. Both represent the same rolling window of
                            # the most recent MS-1 actions, regardless of left-to-right or right-to-left order.
                            new_hist_ther = (hist + (1,))
                            if len(new_hist_ther) > MS - 1: new_hist_ther = new_hist_ther[-(MS - 1):]

                            # Update deviation vector ζ_t if branch constraints are active
                            new_zeta_ther = zeta
                            if use_branch_constraints:
                                new_zeta_ther = list(zeta)
                                for cut_idx, cut in enumerate(forbidden_schedules):
                                    if new_zeta_ther[cut_idx] == 0:  # Not yet deviated
                                        # Check if assignment at (j, t) differs from forbidden schedule
                                        forbidden_val = cut["schedule"].get((j, t), None)
                                        if forbidden_val is not None and forbidden_val != 1:
                                            new_zeta_ther[cut_idx] = 1  # Deviated!
                                new_zeta_ther = tuple(new_zeta_ther)

                            add_state_to_buckets(next_states, cost_ther, prog_ther, ai_count, new_hist_ther, path + [1], recipient_id, pruning_stats, dominance_mode, new_zeta_ther, epsilon)

                        # B: AI
                        if check_strict_feasibility(hist, 0, MS, MIN_MS):
                            cost_ai = cost
                            efficiency = theta_lookup[ai_count] if ai_count < len(theta_lookup) else 1.0
                            prog_ai = prog + efficiency
                            ai_count_new = ai_count + 1
                            # History Shift: Same logic as Therapist (see comment above)
                            new_hist_ai = (hist + (0,))
                            if len(new_hist_ai) > MS - 1: new_hist_ai = new_hist_ai[-(MS - 1):]

                            # Update deviation vector ζ_t if branch constraints are active
                            new_zeta_ai = zeta
                            if use_branch_constraints:
                                new_zeta_ai = list(zeta)
                                for cut_idx, cut in enumerate(forbidden_schedules):
                                    if new_zeta_ai[cut_idx] == 0:  # Not yet deviated
                                        # Check if assignment at (j, t) differs from forbidden schedule
                                        forbidden_val = cut["schedule"].get((j, t), None)
                                        if forbidden_val is not None and forbidden_val != 0:
                                            new_zeta_ai[cut_idx] = 1  # Deviated!
                                new_zeta_ai = tuple(new_zeta_ai)

                            add_state_to_buckets(next_states, cost_ai, prog_ai, ai_count_new, new_hist_ai, path + [0], recipient_id, pruning_stats, dominance_mode, new_zeta_ai, epsilon)

                current_states = next_states
                if not current_states: break

            # Final Step (Transition to Tau)
            # This logic implements Equation (3) for the terminal cost term Ψ_{jξ}:
            # - If ξ < |T| (is_timeout_scenario=False): We enforce a worker assignment (move=1).
            #   This corresponds to Ψ = V_{ξ-1} - π_{jξ}.
            # - If ξ = |T| (is_timeout_scenario=True): We allow any feasible state (move=0 or 1).
            #   This corresponds to Ψ = V_ξ.
            for bucket_key, bucket_list in current_states.items():
                # Extract components from bucket key
                if use_branch_constraints:
                    ai_count, hist, zeta = bucket_key
                else:
                    ai_count, hist = bucket_key
                    zeta = None

                for cost, prog, path in bucket_list:

                    # We collect possible end steps for this state
                    possible_moves = []

                    # Option 1: End with Therapist (1) - Standard
                    if check_strict_feasibility(hist, 1, MS, MIN_MS):
                        possible_moves.append(1)

                    # Option 2: End with App (0) - ONLY if Timeout
                    if is_timeout_scenario:
                        if check_strict_feasibility(hist, 0, MS, MIN_MS):
                            possible_moves.append(0)

                    for move in possible_moves:
                        # Calculate values based on Move type
                        if move == 1:
                            final_cost_accum = cost - pi.get((j, tau), 0)
                            final_prog = prog + 1.0
                            # Here we use the old count since it hasn't increased
                            final_ai_count = ai_count
                        else:  # move == 0
                            final_cost_accum = cost
                            efficiency = theta_lookup[ai_count] if ai_count < len(theta_lookup) else 1.0
                            final_prog = prog + efficiency
                            final_ai_count = ai_count + 1

                        final_path = path + [move]
                        condition_met = (final_prog >= s_k - epsilon)

                        # Update final zeta for this move
                        final_zeta = zeta
                        if use_branch_constraints:
                            final_zeta = list(zeta)
                            for cut_idx, cut in enumerate(forbidden_schedules):
                                if final_zeta[cut_idx] == 0:  # Not yet deviated
                                    # Check if final assignment at (j, tau) differs from forbidden schedule
                                    forbidden_val = cut["schedule"].get((j, tau), None)
                                    if forbidden_val is not None and forbidden_val != move:
                                        final_zeta[cut_idx] = 1  # Deviated!
                            final_zeta = tuple(final_zeta)

                        # Calculate reduced cost early for debug printing
                        temp_reduced_cost = None
                        if condition_met or is_timeout_scenario:
                            duration = tau - r_k + 1
                            temp_reduced_cost = (obj_mode * duration) + final_cost_accum - gamma_k

                        # TERMINAL FEASIBILITY CHECK: All deviation vector entries must equal 1
                        if use_branch_constraints:
                            if temp_reduced_cost is not None:
                                print(f"    [CHECK] Zeta {final_zeta} | RC: {temp_reduced_cost:.6f}")

                            if not all(z == 1 for z in final_zeta):
                                # This path hasn't deviated from all forbidden schedules -> REJECT
                                print(f"    [PRUNED] Schedule matches forbidden branch constraint. "
                                      f"Deviation vector ζ = {final_zeta}")
                                continue

                        if condition_met or is_timeout_scenario:
                            # duration and reduced_cost already calculated above if needed
                            if temp_reduced_cost is None: # Should match logic above
                                duration = tau - r_k + 1
                                reduced_cost = (obj_mode * duration) + final_cost_accum - gamma_k
                            else:
                                reduced_cost = temp_reduced_cost

                            col_candidate = {
                                'k': recipient_id,
                                'worker': j,
                                'start': r_k,
                                'end': tau,
                                'duration': duration,
                                'reduced_cost': reduced_cost,
                                'final_progress': final_prog,
                                'x_vector': generate_full_column_vector(j, final_path, r_k, tau, MAX_TIME, len(WORKERS)),
                                'path_pattern': final_path
                            }

                            if reduced_cost < best_reduced_cost - epsilon:
                                best_reduced_cost = reduced_cost
                                best_columns = [col_candidate]
                            elif abs(reduced_cost - best_reduced_cost) < epsilon:
                                best_columns.append(col_candidate)

            # Debug Output: Bound Pruning Statistics
            if pruned_count_total > 0:
                print(f"    Worker {j}, tau={tau}: Pruned {pruned_count_total} states by Lower Bound")

    return best_columns


# --- 4. Testing & Validation Functions ---

def create_reference_solution(results):
    """
    Creates a sorted reference solution from the results.

    Format: List of (recipient_id, reduced_cost) sorted by recipient_id
    Reduced costs rounded to 2 decimal places.

    Returns:
        list: [(recipient_id, reduced_cost), ...] sorted by recipient_id
    """
    solution = []
    for res in results:
        recipient_id = res['k']
        reduced_cost = round(res['reduced_cost'], 2)
        solution.append((recipient_id, reduced_cost))

    # Sort by recipient_id
    solution.sort(key=lambda x: x[0])

    return solution


def compare_solutions(current_results, reference_solution):
    """
    Compares current results with reference solution.

    Returns:
        tuple: (is_identical, differences)
            - is_identical: bool, True if identical
            - differences: list of dict with deviations
    """
    current_solution = create_reference_solution(current_results)

    # Create dictionaries for easy comparison
    current_dict = {k: rc for k, rc in current_solution}
    reference_dict = {k: rc for k, rc in reference_solution}

    differences = []
    all_recipients = set(current_dict.keys()) | set(reference_dict.keys())

    for recipient_id in sorted(all_recipients):
        current_rc = current_dict.get(recipient_id, None)
        reference_rc = reference_dict.get(recipient_id, None)

        # MODIFIED: Only compare if reference expects a NEGATIVE reduced cost.
        # In Column Generation, we don't care about non-negative solutions (0.00 or positive).
        if reference_rc is not None and reference_rc >= -1e-9:
            continue

        if current_rc is None:
            differences.append({
                'recipient': recipient_id,
                'status': 'MISSING',
                'current': None,
                'reference': reference_rc
            })
        elif reference_rc is None:
            # MODIFIED: Ignore EXTRA solutions if they are non-negative (0.00 or positive).
            # These are irrelevant for Column Generation.
            if current_rc >= -1e-9:
                continue

            differences.append({
                'recipient': recipient_id,
                'status': 'EXTRA',
                'current': current_rc,
                'reference': None
            })
        elif abs(current_rc - reference_rc) > 1e-6:
            differences.append({
                'recipient': recipient_id,
                'status': 'DIFFERENT',
                'current': current_rc,
                'reference': reference_rc,
                'delta': current_rc - reference_rc
            })

    is_identical = len(differences) == 0

    return is_identical, differences


def print_solution_comparison(current_results, reference_solution):
    """
    Prints comparison between current solution and reference.
    """
    is_identical, differences = compare_solutions(current_results, reference_solution)

    print("\n" + "="*70)
    print("SOLUTION VALIDATION")
    print("="*70)

    if is_identical:
        print("✅ IDENTICAL - All Reduced Costs match the reference!")
        current_solution = create_reference_solution(current_results)
        print(f"   Anzahl Recipients: {len(current_solution)}")
    else:
        print(f"❌ DIFFERENCES FOUND - {len(differences)} deviation(s):")
        print()
        for diff in differences:
            recipient_id = diff['recipient']
            status = diff['status']

            if status == 'MISSING':
                print(f"  Recipient {recipient_id:2d}: MISSING in current solution")
                print(f"    Reference: {diff['reference']:.2f}")
            elif status == 'EXTRA':
                print(f"  Recipient {recipient_id:2d}: EXTRA in current solution")
                print(f"    Current: {diff['current']:.2f}")
            elif status == 'DIFFERENT':
                print(f"  Recipient {recipient_id:2d}: DIFFERENT")
                print(f"    Current:   {diff['current']:8.2f}")
                print(f"    Reference: {diff['reference']:8.2f}")
                print(f"    Delta:     {diff['delta']:8.2f}")
            print()

    print("="*70)


# --- 5. Global Labeling Algorithm Function ---

def run_labeling_algorithm(recipients_r, recipients_s, gamma_dict, obj_mode_dict,
                           pi_dict, workers, max_time, ms, min_ms, theta_lookup,
                           print_worker_selection=True, validate_columns=True, print_results=True,
                           use_bound_pruning=True, dominance_mode='bucket', max_columns_per_recipient=None,
                           branch_constraints=None, branching_variant='mp', n_workers=None):
    """
    Global Labeling Algorithm Function.

    Labeling Algorithm for Column Generation (Pricing Problem Solver)

    ================================================================================

    Parameters:
    -----------
    recipients_r : dict
        Release times {recipient_id: r_k}
    recipients_s : dict
        Service targets {recipient_id: s_k}
    gamma_dict : dict
        Dual values gamma {recipient_id: gamma_k}
    obj_mode_dict : dict
        Objective multipliers {recipient_id: multiplier}
    pi_dict : dict
        Dual values pi {(worker_id, time): pi_jt}
    workers : list
        List of worker IDs
    max_time : int
        Planning horizon
    ms : int
        Rolling window size
    min_ms : int
        Minimum human services in window
    theta_lookup : list
        AI efficiency lookup table
    print_worker_selection : bool
        Print worker dominance info per recipient
    validate_columns : bool
        Validate final columns against constraints
    print_results : bool
        Print final results summary
    use_bound_pruning : bool
        Enable/Disable lower bound pruning (Default: True)
    dominance_mode : str
        'bucket' (default) or 'global' dominance strategy
    max_columns_per_recipient : int or None
        Maximum number of alternative optimal columns to store per recipient.
        None (default) = unlimited, stores all optimal columns.
        Example: 10 = store at most 10 alternative optimal schedules per recipient.
    n_workers : int or None
        Number of parallel workers for recipient processing.
        None (default) = sequential processing (single core).
        Example: 4 = use 4 parallel processes (recommended: number of CPU cores).
        Note: Uses multiprocessing.Pool with Map-Reduce pattern (no race conditions).

    Returns:
    --------
    list of dict
        List of best columns (can be multiple per recipient if alternatives exist)
    """
    import time

    t0 = time.time()
    results = []

    # Set global variables (for helper functions)
    global MAX_TIME, MS, MIN_MS, WORKERS, pi
    MAX_TIME = max_time
    MS = ms
    MIN_MS = min_ms
    WORKERS = workers
    pi = pi_dict

    # Pruning Statistics
    pruning_stats = {
        'lb': 0,
        'dominance': 0,
        'printed_dominance': {}  # {recipient_id: bool}
    }

    # === PARALLEL OR SEQUENTIAL PROCESSING ===
    # Use Map-Reduce pattern to avoid race conditions on shared results list

    if n_workers is not None and n_workers > 1:
        # --- PARALLEL PROCESSING (Map-Reduce Pattern) ---
        from multiprocessing import Pool

        print(f"\n[PARALLEL MODE] Using {n_workers} workers for {len(recipients_r)} recipients")

        # MAP: Prepare arguments for each recipient
        recipient_args = []
        for k in recipients_r:
            gamma_val = gamma_dict.get(k, 0.0)
            multiplier = obj_mode_dict.get(k, 1)
            recipient_args.append((
                k, recipients_r[k], recipients_s[k],
                gamma_val, multiplier, pi_dict, workers,
                max_time, ms, min_ms, theta_lookup,
                use_bound_pruning, dominance_mode,
                branch_constraints, branching_variant
            ))

        # Execute in parallel (each process returns its own results)
        with Pool(processes=n_workers) as pool:
            all_cols = pool.starmap(solve_pricing_for_recipient, recipient_args)

        # REDUCE: Merge results sequentially (no race condition)
        recipient_keys = list(recipients_r.keys())
        for k, cols in zip(recipient_keys, all_cols):
            if cols:
                # Validation, deduplication, and storage (identical to sequential)
                valid_cols = []
                for col in cols:
                    validation_errors = validate_final_column(col, recipients_s[k], ms, min_ms, theta_lookup)
                    if not validation_errors:
                        valid_cols.append(col)

                unique_cols = []
                seen_x_vectors = set()
                for col in valid_cols:
                    x_vec_tuple = tuple(col['x_vector'])
                    if x_vec_tuple not in seen_x_vectors:
                        seen_x_vectors.add(x_vec_tuple)
                        unique_cols.append(col)

                if unique_cols:
                    if max_columns_per_recipient is not None and len(unique_cols) > max_columns_per_recipient:
                        cols_to_store = unique_cols[:max_columns_per_recipient]
                    else:
                        cols_to_store = unique_cols

                    for col in cols_to_store:
                        col['num_alternative_optimal'] = len(valid_cols)
                        col['num_unique'] = len(unique_cols)
                        col['num_stored'] = len(cols_to_store)
                    results.extend(cols_to_store)

    else:
        # --- SEQUENTIAL PROCESSING (Original Logic) ---
        for k in recipients_r:
            gamma_val = gamma_dict.get(k, 0.0)
            multiplier = obj_mode_dict.get(k, 1)

            # Pass entire branch_constraints dict - the solver will filter by recipient
            cols = solve_pricing_for_recipient(k, recipients_r[k], recipients_s[k],
                                               gamma_val, multiplier, pi_dict, workers, max_time, ms, min_ms, theta_lookup,
                                               use_bound_pruning=use_bound_pruning, dominance_mode=dominance_mode, branch_constraints=branch_constraints, branching_variant=branching_variant)

            if cols:
                # Validate each column BEFORE storing - only keep valid ones
                valid_cols = []
                for col in cols:
                    validation_errors = validate_final_column(col, recipients_s[k], ms, min_ms, theta_lookup)
                    if not validation_errors:  # Only if NO errors
                        valid_cols.append(col)

                # Remove duplicates based on x_vector (post-filter for bucket dominance edge cases)
                unique_cols = []
                seen_x_vectors = set()
                for col in valid_cols:
                    x_vec_tuple = tuple(col['x_vector'])
                    if x_vec_tuple not in seen_x_vectors:
                        seen_x_vectors.add(x_vec_tuple)
                        unique_cols.append(col)

                # Store UNIQUE VALID optimal columns for this recipient (with optional limit)
                if unique_cols:
                    # Apply max limit if specified
                    if max_columns_per_recipient is not None and len(unique_cols) > max_columns_per_recipient:
                        cols_to_store = unique_cols[:max_columns_per_recipient]
                    else:
                        cols_to_store = unique_cols

                    # Track: total valid found, unique, and actually stored
                    for col in cols_to_store:
                        col['num_alternative_optimal'] = len(valid_cols)  # Total valid alternatives found
                        col['num_unique'] = len(unique_cols)  # Unique alternatives (after dedup)
                        col['num_stored'] = len(cols_to_store)  # How many are actually stored
                    results.extend(cols_to_store)  # Add the (limited) UNIQUE VALID optimal columns

    runtime = time.time() - t0

    if print_results:
        print(f"\nRuntime: {runtime:.4f}s")
        print(f"Pruning Stats: Lower Bound = {pruning_stats['lb']}, State Dominance = {pruning_stats['dominance']}")

        # Group results by recipient to show alternatives
        results_by_recipient = {}
        for res in results:
            k = res['k']
            if k not in results_by_recipient:
                results_by_recipient[k] = []
            results_by_recipient[k].append(res)

        print(f"\n--- Final Results ({len(results)} optimal schedules for {len(results_by_recipient)} recipients) ---")

        for k in sorted(results_by_recipient.keys()):
            recipient_schedules = results_by_recipient[k]
            num_alternatives = len(recipient_schedules)

            # Count active cuts for this recipient
            num_active_cuts = 0
            if branch_constraints:
                for constraint_data in branch_constraints.values():
                    if constraint_data.get("profile") == k and constraint_data.get("direction") == "left":
                        num_active_cuts += 1

            # Show only the FIRST schedule, but display the count
            res = recipient_schedules[0]

            print(f"\nRecipient {k}:")
            print(f"  Reduced Cost: {res['reduced_cost']:.6f}")
            print(f"  Alternative optimal schedules: {num_alternatives}")
            if num_active_cuts > 0:
                print(f"  Active branch cuts: {num_active_cuts}")
            print(f"  Worker: {res['worker']}, Interval: {res['start']}-{res['end']}")

            vec = res['x_vector']
            time_indices = [(i % max_time) + 1 for i, x in enumerate(vec) if x > 0.5]
            print(f"  Active Time Steps (Day 1-{max_time}): {time_indices}")

            last_day_val = res['path_pattern'][-1]
            last_day_type = "Therapist" if last_day_val == 1 else "App"
            print(f"  Last Session Type: {last_day_type} (Val: {last_day_val})")

            if validate_columns:
                validation_errors = validate_final_column(res, recipients_s[res['k']],
                                                         ms, min_ms, theta_lookup)
                if validation_errors:
                    print("  [!] NOT CHECKED CONSTRAINTS / VIOLATIONS:")
                    for err in validation_errors:
                        print(f"      - {err}")
                else:
                    print("  [OK] All constraints satisfied.")

    # Final comprehensive validation of ALL found schedules
    print("\n" + "="*70)
    print("FINAL VALIDATION SUMMARY")
    print("="*70)

    total_violations = 0
    recipients_with_violations = []

    # Group by recipient for uniqueness check
    results_by_recipient = {}
    for res in results:
        k = res['k']
        if k not in results_by_recipient:
            results_by_recipient[k] = []
        results_by_recipient[k].append(res)

    for res in results:
        validation_errors = validate_final_column(res, recipients_s[res['k']],
                                                  ms, min_ms, theta_lookup)
        if validation_errors:
            total_violations += len(validation_errors)
            recipients_with_violations.append(res['k'])
            if print_results:  # Only print details if verbose
                print(f"\n❌ Recipient {res['k']} has {len(validation_errors)} violation(s):")
                for err in validation_errors:
                    print(f"   - {err}")

    # X-Vector Uniqueness Check
    print("\n" + "-"*70)
    print("X-VECTOR UNIQUENESS CHECK (per Recipient)")
    print("-"*70)

    total_x_vector_duplicates = 0
    recipients_with_x_duplicates = []

    for k in sorted(results_by_recipient.keys()):
        recipient_cols = results_by_recipient[k]

        # Convert x_vectors to tuples for set comparison
        x_vectors = [tuple(col['x_vector']) for col in recipient_cols]
        unique_x_vectors = set(x_vectors)

        num_duplicates = len(x_vectors) - len(unique_x_vectors)

        if num_duplicates > 0:
            total_x_vector_duplicates += num_duplicates
            recipients_with_x_duplicates.append(k)
            if print_results:
                print(f"❌ Recipient {k}: {len(recipient_cols)} columns, {len(unique_x_vectors)} unique x_vectors ({num_duplicates} duplicates)")

    if total_x_vector_duplicates == 0:
        print(f"✅ ALL x_vectors are UNIQUE per recipient!")
        print(f"   Total columns checked: {len(results)}")
        print(f"   Recipients: {len(results_by_recipient)}")
    else:
        print(f"❌ FOUND {total_x_vector_duplicates} x_vector DUPLICATES!")
        print(f"   Affected recipients: {recipients_with_x_duplicates}")

    print("\n" + "-"*70)

    if total_violations == 0:
        print(f"✅ ALL {len(results)} schedules are VALID!")
        print(f"   - All rolling window constraints satisfied (MS={ms}, MIN_MS={min_ms})")
        print(f"   - All start/end constraints satisfied")
        print(f"   - All service targets met (or timeout)")
    else:
        print(f"❌ FOUND {total_violations} VIOLATIONS in {len(recipients_with_violations)} recipients!")
        print(f"   Recipients with violations: {recipients_with_violations}")
        print(f"   WARNING: These schedules may be INFEASIBLE!")

    print("="*70)

    return results


# --- 6. Main Execution ---

if __name__ == "__main__":
    # Optional: Reference solution for comparison (None = create for the first time)
    # After the first run: Copy the output here
    REFERENCE_SOLUTION = [
        ( 2,   -4.00),
        ( 6,   21.00),
        ( 8,  -42.00),
        (13,   18.00),
        (15,  -18.00),
        (18,   18.00),
        (19,  -11.00),
        (20,   29.00),
        (23,  -29.00),
        (28,   -6.00),
        (31,    7.00),
        (36,  -10.00),
        (37,   18.00),
        (38,    0.00),
        (44,   -9.00),
        (45,  -21.00),
        (47,    0.00),
        (48,   -5.00),
        (50,  -34.00),
        (51,   -7.00),
        (53,   13.00),
        (54,   -3.00),
        (60,  -12.00),
        (64,   -9.00),
        (66,  -15.00),
        (68,   -8.00),
        (76,   -7.00),
        (78,   -8.00),
        (80,  -29.00),
        (84,   -3.00),
    ]

    # ========================================================================
    # BRANCH CONSTRAINTS DEFINITION
    # ========================================================================

    class MPVariableBranching:
        def __init__(self, profile_n, column_a, bound, direction, original_schedule=None):
            self.profile = profile_n
            self.column = column_a
            self.bound = bound
            self.direction = direction
            self.original_schedule = original_schedule

    # Define the constraints list
    node_branching_constraints = [
        MPVariableBranching(
            profile_n=6,
            column_a=7,
            bound=0.0,
            direction='left',
            original_schedule={
                (6, 1, 1, 0): 0, (6, 1, 2, 0): 0, (6, 1, 3, 0): 0, (6, 1, 4, 0): 0, (6, 1, 5, 0): 0,
                (6, 1, 6, 0): 0, (6, 1, 7, 0): 0, (6, 1, 8, 0): 0, (6, 1, 9, 0): 0, (6, 1, 10, 0): 0,
                (6, 1, 11, 0): 0, (6, 1, 12, 0): 0, (6, 1, 13, 0): 0, (6, 1, 14, 0): 0, (6, 1, 15, 0): 0,
                (6, 1, 16, 0): 1, (6, 1, 17, 0): 1, (6, 1, 18, 0): 0, (6, 1, 19, 0): 1, (6, 1, 20, 0): 1,
                (6, 1, 21, 0): 0, (6, 1, 22, 0): 1, (6, 1, 23, 0): 1, (6, 1, 24, 0): 1, (6, 1, 25, 0): 1,
                (6, 1, 26, 0): 0, (6, 1, 27, 0): 0, (6, 1, 28, 0): 0, (6, 1, 29, 0): 1, (6, 1, 30, 0): 0,
                (6, 1, 31, 0): 0, (6, 1, 32, 0): 0, (6, 1, 33, 0): 0, (6, 1, 34, 0): 0, (6, 1, 35, 0): 0,
                (6, 1, 36, 0): 0, (6, 1, 37, 0): 0, (6, 1, 38, 0): 0, (6, 1, 39, 0): 0, (6, 1, 40, 0): 0,
                (6, 1, 41, 0): 0, (6, 1, 42, 0): 0, (6, 2, 1, 0): 0, (6, 2, 2, 0): 0, (6, 2, 3, 0): 0,
                (6, 2, 4, 0): 0, (6, 2, 5, 0): 0, (6, 2, 6, 0): 0, (6, 2, 7, 0): 0, (6, 2, 8, 0): 0,
                (6, 2, 9, 0): 0, (6, 2, 10, 0): 0, (6, 2, 11, 0): 0, (6, 2, 12, 0): 0, (6, 2, 13, 0): 0,
                (6, 2, 14, 0): 0, (6, 2, 15, 0): 0, (6, 2, 16, 0): 0, (6, 2, 17, 0): 0, (6, 2, 18, 0): 0,
                (6, 2, 19, 0): 0, (6, 2, 20, 0): 0, (6, 2, 21, 0): 0, (6, 2, 22, 0): 0, (6, 2, 23, 0): 0,
                (6, 2, 24, 0): 0, (6, 2, 25, 0): 0, (6, 2, 26, 0): 0, (6, 2, 27, 0): 0, (6, 2, 28, 0): 0,
                (6, 2, 29, 0): 0, (6, 2, 30, 0): 0, (6, 2, 31, 0): 0, (6, 2, 32, 0): 0, (6, 2, 33, 0): 0,
                (6, 2, 34, 0): 0, (6, 2, 35, 0): 0, (6, 2, 36, 0): 0, (6, 2, 37, 0): 0, (6, 2, 38, 0): 0,
                (6, 2, 39, 0): 0, (6, 2, 40, 0): 0, (6, 2, 41, 0): 0, (6, 2, 42, 0): 0, (6, 3, 1, 0): 0,
                (6, 3, 2, 0): 0, (6, 3, 3, 0): 0, (6, 3, 4, 0): 0, (6, 3, 5, 0): 0, (6, 3, 6, 0): 0,
                (6, 3, 7, 0): 0, (6, 3, 8, 0): 0, (6, 3, 9, 0): 0, (6, 3, 10, 0): 0, (6, 3, 11, 0): 0,
                (6, 3, 12, 0): 0, (6, 3, 13, 0): 0, (6, 3, 14, 0): 0, (6, 3, 15, 0): 0, (6, 3, 16, 0): 0,
                (6, 3, 17, 0): 0, (6, 3, 18, 0): 0, (6, 3, 19, 0): 0, (6, 3, 20, 0): 0, (6, 3, 21, 0): 0,
                (6, 3, 22, 0): 0, (6, 3, 23, 0): 0, (6, 3, 24, 0): 0, (6, 3, 25, 0): 0, (6, 3, 26, 0): 0,
                (6, 3, 27, 0): 0, (6, 3, 28, 0): 0, (6, 3, 29, 0): 0, (6, 3, 30, 0): 0, (6, 3, 31, 0): 0,
                (6, 3, 32, 0): 0, (6, 3, 33, 0): 0, (6, 3, 34, 0): 0, (6, 3, 35, 0): 0, (6, 3, 36, 0): 0,
                (6, 3, 37, 0): 0, (6, 3, 38, 0): 0, (6, 3, 39, 0): 0, (6, 3, 40, 0): 0, (6, 3, 41, 0): 0,
                (6, 3, 42, 0): 0
            }
        ),
        MPVariableBranching(
            profile_n=6,
            column_a=7,
            bound=0.0,
            direction='left',
            original_schedule={
                (6, 1, 1, 0): 0, (6, 1, 2, 0): 0, (6, 1, 3, 0): 0, (6, 1, 4, 0): 0, (6, 1, 5, 0): 0,
                (6, 1, 6, 0): 0, (6, 1, 7, 0): 0, (6, 1, 8, 0): 0, (6, 1, 9, 0): 0, (6, 1, 10, 0): 0,
                (6, 1, 11, 0): 0, (6, 1, 12, 0): 0, (6, 1, 13, 0): 0, (6, 1, 14, 0): 0, (6, 1, 15, 0): 0,
                (6, 1, 16, 0): 1, (6, 1, 17, 0): 1, (6, 1, 18, 0): 0, (6, 1, 19, 0): 1, (6, 1, 20, 0): 1,
                (6, 1, 21, 0): 1, (6, 1, 22, 0): 0, (6, 1, 23, 0): 0, (6, 1, 24, 0): 1, (6, 1, 25, 0): 1,
                (6, 1, 26, 0): 0, (6, 1, 27, 0): 0, (6, 1, 28, 0): 0, (6, 1, 29, 0): 1, (6, 1, 30, 0): 0,
                (6, 1, 31, 0): 0, (6, 1, 32, 0): 0, (6, 1, 33, 0): 0, (6, 1, 34, 0): 0, (6, 1, 35, 0): 0,
                (6, 1, 36, 0): 0, (6, 1, 37, 0): 0, (6, 1, 38, 0): 0, (6, 1, 39, 0): 0, (6, 1, 40, 0): 0,
                (6, 1, 41, 0): 0, (6, 1, 42, 0): 0, (6, 2, 1, 0): 0, (6, 2, 2, 0): 0, (6, 2, 3, 0): 0,
                (6, 2, 4, 0): 0, (6, 2, 5, 0): 0, (6, 2, 6, 0): 0, (6, 2, 7, 0): 0, (6, 2, 8, 0): 0,
                (6, 2, 9, 0): 0, (6, 2, 10, 0): 0, (6, 2, 11, 0): 0, (6, 2, 12, 0): 0, (6, 2, 13, 0): 0,
                (6, 2, 14, 0): 0, (6, 2, 15, 0): 0, (6, 2, 16, 0): 0, (6, 2, 17, 0): 0, (6, 2, 18, 0): 0,
                (6, 2, 19, 0): 0, (6, 2, 20, 0): 0, (6, 2, 21, 0): 0, (6, 2, 22, 0): 0, (6, 2, 23, 0): 0,
                (6, 2, 24, 0): 0, (6, 2, 25, 0): 0, (6, 2, 26, 0): 0, (6, 2, 27, 0): 0, (6, 2, 28, 0): 0,
                (6, 2, 29, 0): 0, (6, 2, 30, 0): 0, (6, 2, 31, 0): 0, (6, 2, 32, 0): 0, (6, 2, 33, 0): 0,
                (6, 2, 34, 0): 0, (6, 2, 35, 0): 0, (6, 2, 36, 0): 0, (6, 2, 37, 0): 0, (6, 2, 38, 0): 0,
                (6, 2, 39, 0): 0, (6, 2, 40, 0): 0, (6, 2, 41, 0): 0, (6, 2, 42, 0): 0, (6, 3, 1, 0): 0,
                (6, 3, 2, 0): 0, (6, 3, 3, 0): 0, (6, 3, 4, 0): 0, (6, 3, 5, 0): 0, (6, 3, 6, 0): 0,
                (6, 3, 7, 0): 0, (6, 3, 8, 0): 0, (6, 3, 9, 0): 0, (6, 3, 10, 0): 0, (6, 3, 11, 0): 0,
                (6, 3, 12, 0): 0, (6, 3, 13, 0): 0, (6, 3, 14, 0): 0, (6, 3, 15, 0): 0, (6, 3, 16, 0): 0,
                (6, 3, 17, 0): 0, (6, 3, 18, 0): 0, (6, 3, 19, 0): 0, (6, 3, 20, 0): 0, (6, 3, 21, 0): 0,
                (6, 3, 22, 0): 0, (6, 3, 23, 0): 0, (6, 3, 24, 0): 0, (6, 3, 25, 0): 0, (6, 3, 26, 0): 0,
                (6, 3, 27, 0): 0, (6, 3, 28, 0): 0, (6, 3, 29, 0): 0, (6, 3, 30, 0): 0, (6, 3, 31, 0): 0,
                (6, 3, 32, 0): 0, (6, 3, 33, 0): 0, (6, 3, 34, 0): 0, (6, 3, 35, 0): 0, (6, 3, 36, 0): 0,
                (6, 3, 37, 0): 0, (6, 3, 38, 0): 0, (6, 3, 39, 0): 0, (6, 3, 40, 0): 0, (6, 3, 41, 0): 0,
                (6, 3, 42, 0): 0
            }
        ),
        MPVariableBranching(
            profile_n=6,
            column_a=7,
            bound=0.0,
            direction='right',
            original_schedule={
                (6, 1, 1, 0): 0, (6, 1, 2, 0): 0, (6, 1, 3, 0): 0, (6, 1, 4, 0): 0, (6, 1, 5, 0): 0,
                (6, 1, 6, 0): 0, (6, 1, 7, 0): 0, (6, 1, 8, 0): 0, (6, 1, 9, 0): 0, (6, 1, 10, 0): 0,
                (6, 1, 11, 0): 0, (6, 1, 12, 0): 0, (6, 1, 13, 0): 0, (6, 1, 14, 0): 0, (6, 1, 15, 0): 0,
                (6, 1, 16, 0): 1, (6, 1, 17, 0): 1, (6, 1, 18, 0): 1, (6, 1, 19, 0): 1, (6, 1, 20, 0): 0,
                (6, 1, 21, 0): 0, (6, 1, 22, 0): 1, (6, 1, 23, 0): 0, (6, 1, 24, 0): 1, (6, 1, 25, 0): 1,
                (6, 1, 26, 0): 0, (6, 1, 27, 0): 0, (6, 1, 28, 0): 0, (6, 1, 29, 0): 1, (6, 1, 30, 0): 0,
                (6, 1, 31, 0): 0, (6, 1, 32, 0): 0, (6, 1, 33, 0): 0, (6, 1, 34, 0): 0, (6, 1, 35, 0): 0,
                (6, 1, 36, 0): 0, (6, 1, 37, 0): 0, (6, 1, 38, 0): 0, (6, 1, 39, 0): 0, (6, 1, 40, 0): 0,
                (6, 1, 41, 0): 0, (6, 1, 42, 0): 0, (6, 2, 1, 0): 0, (6, 2, 2, 0): 0, (6, 2, 3, 0): 0,
                (6, 2, 4, 0): 0, (6, 2, 5, 0): 0, (6, 2, 6, 0): 0, (6, 2, 7, 0): 0, (6, 2, 8, 0): 0,
                (6, 2, 9, 0): 0, (6, 2, 10, 0): 0, (6, 2, 11, 0): 0, (6, 2, 12, 0): 0, (6, 2, 13, 0): 0,
                (6, 2, 14, 0): 0, (6, 2, 15, 0): 0, (6, 2, 16, 0): 0, (6, 2, 17, 0): 0, (6, 2, 18, 0): 0,
                (6, 2, 19, 0): 0, (6, 2, 20, 0): 0, (6, 2, 21, 0): 0, (6, 2, 22, 0): 0, (6, 2, 23, 0): 0,
                (6, 2, 24, 0): 0, (6, 2, 25, 0): 0, (6, 2, 26, 0): 0, (6, 2, 27, 0): 0, (6, 2, 28, 0): 0,
                (6, 2, 29, 0): 0, (6, 2, 30, 0): 0, (6, 2, 31, 0): 0, (6, 2, 32, 0): 0, (6, 2, 33, 0): 0,
                (6, 2, 34, 0): 0, (6, 2, 35, 0): 0, (6, 2, 36, 0): 0, (6, 2, 37, 0): 0, (6, 2, 38, 0): 0,
                (6, 2, 39, 0): 0, (6, 2, 40, 0): 0, (6, 2, 41, 0): 0, (6, 2, 42, 0): 0, (6, 3, 1, 0): 0,
                (6, 3, 2, 0): 0, (6, 3, 3, 0): 0, (6, 3, 4, 0): 0, (6, 3, 5, 0): 0, (6, 3, 6, 0): 0,
                (6, 3, 7, 0): 0, (6, 3, 8, 0): 0, (6, 3, 9, 0): 0, (6, 3, 10, 0): 0, (6, 3, 11, 0): 0,
                (6, 3, 12, 0): 0, (6, 3, 13, 0): 0, (6, 3, 14, 0): 0, (6, 3, 15, 0): 0, (6, 3, 16, 0): 0,
                (6, 3, 17, 0): 0, (6, 3, 18, 0): 0, (6, 3, 19, 0): 0, (6, 3, 20, 0): 0, (6, 3, 21, 0): 0,
                (6, 3, 22, 0): 0, (6, 3, 23, 0): 0, (6, 3, 24, 0): 0, (6, 3, 25, 0): 0, (6, 3, 26, 0): 0,
                (6, 3, 27, 0): 0, (6, 3, 28, 0): 0, (6, 3, 29, 0): 0, (6, 3, 30, 0): 0, (6, 3, 31, 0): 0,
                (6, 3, 32, 0): 0, (6, 3, 33, 0): 0, (6, 3, 34, 0): 0, (6, 3, 35, 0): 0, (6, 3, 36, 0): 0,
                (6, 3, 37, 0): 0, (6, 3, 38, 0): 0, (6, 3, 39, 0): 0, (6, 3, 40, 0): 0, (6, 3, 41, 0): 0,
                (6, 3, 42, 0): 0
            }
        ),
    ]

    # Convert list to dictionary for labeling algorithm
    branch_constraints = {}
    for idx, constraint in enumerate(node_branching_constraints):
        branch_constraints[idx] = {
            "profile": constraint.profile,
            "column": constraint.column,
            "direction": constraint.direction,
            "bound": constraint.bound,
            "original_schedule": constraint.original_schedule
        }

    # To disable all constraints, just use:
    # branch_constraints = {}

    # ========================================================================
    # RUN LABELING ALGORITHM
    # ========================================================================

    print("="*70)
    print("LABELING ALGORITHM - Column Generation Pricing")
    start_time = time.time()
    print("="*70)
    if branch_constraints:
        print(f"Branch Constraints: {len(branch_constraints)} constraint(s) defined")
        for key, constraint in branch_constraints.items():
            print(f"  {key}: profile={constraint['profile']}, direction={constraint['direction']}")
    else:
        print("Branch Constraints: None (empty dict)")
    print("="*70)
    print()

    results = run_labeling_algorithm(
        recipients_r=r_i,
        recipients_s=s_i,
        gamma_dict=gamma,
        obj_mode_dict=obj_mode,
        pi_dict=pi,
        workers=WORKERS,
        max_time=MAX_TIME,
        ms=MS,
        min_ms=MIN_MS,
        theta_lookup=theta_lookup,
        print_worker_selection=True,
        validate_columns=True,
        print_results=True,
        use_bound_pruning=True,
        dominance_mode='bucket',
        max_columns_per_recipient=20, # No limit for now, to see all alternatives
        branch_constraints=branch_constraints,
        branching_variant='mp',
        n_workers=None # Set to e.g. 4 to enable parallel processing with 4 cores
    )

    # Comparison with fixed reference solution
    if REFERENCE_SOLUTION is not None:
        print_solution_comparison(results, REFERENCE_SOLUTION)
    else:
        print("\n" + "="*70)
        print("INFO: No reference solution defined. Set REFERENCE_SOLUTION to enable regression testing.")
        print("="*70)

    print(f"Runtime", time.time()-start_time)

    # Save results to text file
    save_results_to_txt(results, filename="results/results2.txt")

    # Compare result files
    compare_result_files("results/results.txt", "results/results2.txt")