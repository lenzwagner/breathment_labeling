import collections
import sys
import time

# --- 1. Input Data ---

gamma = {2: 15.0, 6: 0.0, 8: 42.0, 13: 0.0, 15: 18.0, 16: 0.0, 18: 0.0, 19: 11.0, 20: 0.0, 23: 29.0, 28: 6.0, 31: 0.0, 32: 0.0, 35: 0.0, 36: 10.0, 37: 0.0, 38: 6.0, 44: 9.0, 45: 21.0, 47: 0.0, 48: 12.0, 49: 0.0, 50: 42.0, 51: 7.0, 53: 0.0, 54: 11.0, 60: 12.0, 61: 0.0, 64: 16.0, 65: 0.0, 66: 15.0, 67: 0.0, 68: 8.0, 73: 0.0, 74: 0.0, 76: 7.0, 77: 0.0, 78: 8.0, 80: 29.0, 84: 11.0}
pi = {(1, 1): 0.0, (1, 2): 0.0, (1, 3): 0.0, (1, 4): 0.0, (1, 5): 0.0, (1, 6): 0.0, (1, 7): -35.0, (1, 8): 0.0, (1, 9): -3.0, (1, 10): 0.0, (1, 11): 0.0, (1, 12): 0.0, (1, 13): 0.0, (1, 14): -29.0, (1, 15): 0.0, (1, 16): 0.0, (1, 17): 0.0, (1, 18): 0.0, (1, 19): 0.0, (1, 20): 0.0, (1, 21): -21.0, (1, 22): 0.0, (1, 23): 0.0, (1, 24): 0.0, (1, 25): 0.0, (1, 26): 0.0, (1, 27): 0.0, (1, 28): -18.0, (1, 29): 0.0, (1, 30): 0.0, (1, 31): 0.0, (1, 32): 0.0, (1, 33): 0.0, (1, 34): 0.0, (1, 35): -18.0, (1, 36): 0.0, (1, 37): 0.0, (1, 38): 0.0, (1, 39): 0.0, (1, 40): 0.0, (1, 41): 0.0, (1, 42): 0.0, (2, 1): -29.0, (2, 2): -6.0, (2, 3): 0.0, (2, 4): 0.0, (2, 5): 0.0, (2, 6): 0.0, (2, 7): 0.0, (2, 8): -10.0, (2, 9): 0.0, (2, 10): 0.0, (2, 11): 0.0, (2, 12): 0.0, (2, 13): 0.0, (2, 14): 0.0, (2, 15): -13.0, (2, 16): 0.0, (2, 17): 0.0, (2, 18): 0.0, (2, 19): 0.0, (2, 20): 0.0, (2, 21): 0.0, (2, 22): -29.0, (2, 23): 0.0, (2, 24): 0.0, (2, 25): 0.0, (2, 26): 0.0, (2, 27): 0.0, (2, 28): 0.0, (2, 29): -21.0, (2, 30): 0.0, (2, 31): 0.0, (2, 32): 0.0, (2, 33): 0.0, (2, 34): 0.0, (2, 35): 0.0, (2, 36): -11.0, (2, 37): 0.0, (2, 38): 0.0, (2, 39): 0.0, (2, 40): 0.0, (2, 41): 0.0, (2, 42): 0.0, (3, 1): -29.0, (3, 2): -6.0, (3, 3): 0.0, (3, 4): 0.0, (3, 5): 0.0, (3, 6): 0.0, (3, 7): 0.0, (3, 8): 0.0, (3, 9): -12.0, (3, 10): 0.0, (3, 11): 0.0, (3, 12): 0.0, (3, 13): 0.0, (3, 14): 0.0, (3, 15): 0.0, (3, 16): -35.0, (3, 17): 0.0, (3, 18): 0.0, (3, 19): 0.0, (3, 20): 0.0, (3, 21): -7.0, (3, 22): 0.0, (3, 23): -14.0, (3, 24): 0.0, (3, 25): 0.0, (3, 26): -6.0, (3, 27): 0.0, (3, 28): 0.0, (3, 29): 0.0, (3, 30): -12.0, (3, 31): 0.0, (3, 32): 0.0, (3, 33): 0.0, (3, 34): 0.0, (3, 35): 0.0, (3, 36): 0.0, (3, 37): -18.0, (3, 38): 0.0, (3, 39): 0.0, (3, 40): 0.0, (3, 41): 0.0, (3, 42): 0.0}

r_i =  {2: 2, 6: 16, 8: 14, 13: 22, 15: 25, 16: 39, 18: 33, 19: 27, 20: 16, 23: 12, 28: 25, 31: 15, 32: 38, 35: 40, 36: 29, 37: 25, 38: 5, 44: 18, 45: 21, 47: 30, 48: 2, 49: 36, 50: 1, 51: 26, 53: 12, 54: 3, 60: 6, 61: 16, 64: 4, 65: 38, 66: 25, 67: 15, 68: 28, 73: 12, 74: 36, 76: 18, 77: 38, 78: 24, 80: 14, 84: 5}
s_i =  {1: 17, 2: 9, 3: 13, 4: 17, 5: 4, 6: 6, 7: 5, 8: 9, 9: 5, 10: 8, 11: 5, 12: 9, 13: 5, 14: 5, 15: 6, 16: 7, 17: 4, 18: 5, 19: 7, 20: 7, 21: 8, 23: 10, 24: 10, 25: 8, 26: 6, 27: 8, 28: 5, 29: 5, 30: 5, 31: 6, 32: 4, 33: 4, 34: 4, 35: 3, 36: 8, 37: 4, 38: 5, 39: 10, 40: 8, 41: 9, 43: 9, 44: 7, 45: 4, 46: 10, 47: 2, 48: 6, 49: 7, 50: 7, 51: 6, 52: 10, 53: 3, 54: 7, 55: 6, 56: 6, 57: 7, 58: 9, 59: 7, 60: 5, 61: 4, 62: 4, 63: 2, 64: 6, 65: 10, 66: 8, 67: 3, 68: 7, 69: 5, 70: 8, 71: 4, 72: 7, 73: 6, 74: 6, 75: 3, 76: 5, 77: 7, 78: 7, 79: 3, 80: 6, 81: 4, 82: 6, 83: 11, 84: 7}
obj_mode =  {2: 1, 6: 0, 8: 0, 13: 0, 15: 0, 16: 0, 18: 0, 19: 0, 20: 0, 23: 0, 28: 0, 31: 0, 32: 0, 35: 0, 36: 0, 37: 0, 38: 1, 44: 0, 45: 0, 47: 0, 48: 1, 49: 0, 50: 1, 51: 0, 53: 0, 54: 1, 60: 0, 61: 0, 64: 1, 65: 0, 66: 0, 67: 0, 68: 0, 73: 0, 74: 0, 76: 0, 77: 0, 78: 0, 80: 0, 84: 1}


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
    Validiert eine fertige Spalte strikt auf alle Constraints.
    Gibt eine Liste von Fehlern zurück (leer wenn alles OK).
    """
    errors = []
    path = col_data['path_pattern']

    # 1. Check Start & End Constraint
    if path[0] != 1:
        errors.append(f"Start constraint violation: First day must be Machine (1), found {path[0]}")

    # MODIFIZIERT: Check End Constraint
    # Wenn es KEIN Timeout ist, MUSS es eine 1 sein.
    # Wenn es ein Timeout ist, darf es auch eine 0 sein.
    is_timeout = (col_data['end'] == MAX_TIME)

    if path[-1] != 1:
        if not is_timeout:
            errors.append(f"End constraint violation: Last day must be Machine (1), found {path[-1]}")
        # Im Timeout-Fall (is_timeout == True) erlauben wir hier implizit die 0.

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

    # Toleranz 1e-9
    if progress < s_req - 1e-9:
        # Sonderfall: Timeout (Ende des Horizonts) darf Ziel verfehlen, wenn Modell das erlaubt.
        # Wir markieren es hier als Info/Fehler zur Kontrolle (wie gewünscht).
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


def solve_pricing_for_recipient(k, r_k, s_k, gamma_k, obj_multiplier):
    best_reduced_cost = float('inf')
    best_columns = []
    epsilon = 1e-9

    time_until_end = MAX_TIME - r_k + 1
    
    # Worker Dominance Pre-Elimination
    candidate_workers = compute_candidate_workers(WORKERS, r_k, MAX_TIME, pi)
    eliminated_workers = [w for w in WORKERS if w not in candidate_workers]
    
    # Print für jeden Recipient
    if eliminated_workers:
        print(f"Recipient {k:2d}: Candidate workers = {candidate_workers} (eliminated {eliminated_workers})")
    else:
        print(f"Recipient {k:2d}: Candidate workers = {candidate_workers} (no dominance)")

    for j in candidate_workers:
        effective_min_duration = min(int(s_k), time_until_end)
        start_tau = r_k + effective_min_duration - 1

        for tau in range(start_tau, MAX_TIME + 1):
            is_timeout_scenario = (tau == MAX_TIME)

            start_cost = -pi.get((j, r_k), 0)
            current_states = {
                (1.0, 0, (1,)): (start_cost, [1])
            }

            # DP Loop bis kurz vor Tau
            for t in range(r_k + 1, tau):
                next_states = {}
                for state, (cost, path) in current_states.items():
                    prog, ai_count, hist = state

                    remaining_steps = tau - t + 1
                    if not is_timeout_scenario:
                        if prog + remaining_steps * 1.0 < s_k - epsilon:
                            continue

                    # A: Therapist
                    if check_strict_feasibility(hist, 1, MS, MIN_MS):
                        cost_ther = cost - pi.get((j, t), 0)
                        prog_ther = prog + 1.0
                        new_hist_ther = (hist + (1,))
                        if len(new_hist_ther) > MS - 1: new_hist_ther = new_hist_ther[-(MS - 1):]

                        state_ther = (prog_ther, ai_count, new_hist_ther)
                        if state_ther not in next_states or cost_ther < next_states[state_ther][0]:
                            next_states[state_ther] = (cost_ther, path + [1])

                    # B: AI
                    if check_strict_feasibility(hist, 0, MS, MIN_MS):
                        cost_ai = cost
                        efficiency = theta_lookup[ai_count] if ai_count < len(theta_lookup) else 1.0
                        prog_ai = prog + efficiency
                        ai_count_new = ai_count + 1
                        new_hist_ai = (hist + (0,))
                        if len(new_hist_ai) > MS - 1: new_hist_ai = new_hist_ai[-(MS - 1):]

                        state_ai = (prog_ai, ai_count_new, new_hist_ai)
                        if state_ai not in next_states or cost_ai < next_states[state_ai][0]:
                            next_states[state_ai] = (cost_ai, path + [0])

                current_states = next_states
                if not current_states: break

            # Final Step (Transition to Tau)
            # Hier ist die Änderung für den Timeout-Fall
            for state, (cost, path) in current_states.items():
                prog, ai_count, hist = state

                # Wir sammeln mögliche End-Schritte für diesen State
                possible_moves = []

                # Option 1: Enden mit Therapeut (1) - Standard
                if check_strict_feasibility(hist, 1, MS, MIN_MS):
                    possible_moves.append(1)

                # Option 2: Enden mit App (0) - NUR wenn Timeout
                if is_timeout_scenario:
                    if check_strict_feasibility(hist, 0, MS, MIN_MS):
                        possible_moves.append(0)

                for move in possible_moves:
                    # Berechne Werte basierend auf Move-Typ
                    if move == 1:
                        final_cost_accum = cost - pi.get((j, tau), 0)
                        final_prog = prog + 1.0
                        # Hier nutzen wir den alten count, da er sich nicht erhöht hat
                        final_ai_count = ai_count
                    else:  # move == 0
                        final_cost_accum = cost
                        efficiency = theta_lookup[ai_count] if ai_count < len(theta_lookup) else 1.0
                        final_prog = prog + efficiency
                        final_ai_count = ai_count + 1

                    final_path = path + [move]
                    condition_met = (final_prog >= s_k - epsilon)

                    if condition_met or is_timeout_scenario:
                        duration = tau - r_k + 1
                        reduced_cost = (obj_multiplier * duration) + final_cost_accum - gamma_k

                        col_candidate = {
                            'k': k,
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

    return best_columns


# --- 4. Global Labeling Algorithm Function ---

def run_labeling_algorithm(recipients_r, recipients_s, gamma_dict, obj_mode_dict, 
                           pi_dict, workers, max_time, ms, min_ms, theta_lookup,
                           print_worker_selection=True, validate_columns=True, print_results=True):
    """
    Globale Labeling-Algorithmus Funktion.
    
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
        
    Returns:
    --------
    list of dict
        List of best columns (one per recipient)
    """
    import time
    
    t0 = time.time()
    results = []
    
    # Setze globale Variablen (für Helper-Funktionen)
    global MAX_TIME, MS, MIN_MS, WORKERS, pi
    MAX_TIME = max_time
    MS = ms
    MIN_MS = min_ms
    WORKERS = workers
    pi = pi_dict
    
    for k in recipients_r:
        gamma_val = gamma_dict.get(k, 0.0)
        multiplier = obj_mode_dict.get(k, 1)
        
        cols = solve_pricing_for_recipient(k, recipients_r[k], recipients_s[k], 
                                           gamma_val, multiplier)
        
        if cols:
            results.append(cols[0])
    
    runtime = time.time() - t0
    
    if print_results:
        print(f"\nRuntime: {runtime:.4f}s")
        print("\n--- Final Results (First found optimal per Recipient) ---")
        
        for res in results:
            print(f"\nRecipient {res['k']}:")
            print(f"  Reduced Cost: {res['reduced_cost']:.6f}")
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
    
    return results


# --- 5. Main Execution ---

if __name__ == "__main__":
    # Aufruf der globalen Labeling-Algorithmus Funktion
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
        print_results=True
    )
