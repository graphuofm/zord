// zord buffer-pool cache SIMULATION in C++ (the HOT PATH for the known snapshot access schedule).
//
// PURPOSE (O2): simulate Belady/MIN and reuse-distance (MRD) eviction over the KNOWN access
// sequence at scale (L = S * epochs * window can reach tens of millions, called per device). The
// Python `runtime.bufferpool` runs an explicit per-access for-loop with a dict (O(L) but Python-
// slow); this kernel does the simulation in C++ and Python keeps the byte accounting + reporting.
//
// RESULT CONTRACT: this kernel must reproduce, BYTE-FOR-BYTE, the miss set (is_miss[]) that the
// Python `BufferPool._staged_bytes_for_policy` produces, because the Python side multiplies is_miss
// by the per-access unit bytes to get staged_bytes. That means the EVICTION VICTIM choice (and its
// tie-break) must match Python's `max(resident, key=lambda k: resident[k])`, which returns the
// FIRST resident unit (in insertion order) attaining the maximum key. We replicate that exactly.
//
// Build:  g++ -O3 -std=c++17 -fopenmp src/zord/cpp/bufferpool.cpp -o build/bufferpool
//
// Input  (binary, little-endian):
//   int64  L                  -- access length
//   int32  access[L]          -- unit ids in access order
//   int64  capacity_units     -- resident slots (clamped to >= 1, matching Python)
//   int32  policy             -- 0 = belady (optimal offline), 1 = mrd (online reuse-distance)
// Output (binary, little-endian):
//   int64  L
//   int32  is_miss[L]         -- 1 if access i was a miss/stage, else 0
//   int64  admissions         -- total misses (== sum is_miss)
//   int64  evictions          -- total evictions (admissions minus the cold fill that found room)
//
// belady: precompute next_use[L] (one O(L) backward pass), on a miss with a full cache evict the
//         resident unit whose next_use is FARTHEST in the future (an unreferenced unit -> _NEVER ->
//         evicted first). Tie -> the earliest-inserted resident unit (Python `max` first-wins).
// mrd:    track each unit's recent reuse distance (gap between its last two accesses; a never-seen
//         unit gets BIG = L), evict the resident unit with the LARGEST estimate. Same tie rule.
//
// Eviction is an O(cap) scan over the resident set; total O(L*cap_effective). cap is the HBM slot
// budget (typically a handful to a few hundred), so this is effectively O(L) at scale.
#include <cstdio>
#include <cstdint>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <chrono>
#include <limits>
using namespace std;
using i64 = int64_t;
using i32 = int32_t;

static double now() {
    return chrono::duration<double>(chrono::steady_clock::now().time_since_epoch()).count();
}

// Sentinel "next use" for a unit never referenced again on the remaining schedule. Matches the
// Python _NEVER = np.iinfo(np.int64).max so the eviction comparison is identical.
static const i64 NEVER = numeric_limits<i64>::max();

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <in.bin> <out.bin>\n", argv[0]);
        return 1;
    }
    const char* inpath = argv[1];
    const char* outpath = argv[2];
    double t0 = now();

    FILE* f = fopen(inpath, "rb");
    if (!f) { fprintf(stderr, "open fail %s\n", inpath); return 1; }
    i64 L = 0;
    if (fread(&L, 8, 1, f) != 1) { fprintf(stderr, "hdr L fail\n"); return 1; }
    if (L < 0) { fprintf(stderr, "bad L=%lld\n", (long long)L); return 1; }
    vector<i32> access((size_t)L);
    if (L > 0 && fread(access.data(), 4, (size_t)L, f) != (size_t)L) {
        fprintf(stderr, "access read fail\n"); return 1;
    }
    i64 capacity_units = 0;
    i32 policy = 0;
    if (fread(&capacity_units, 8, 1, f) != 1) { fprintf(stderr, "cap read fail\n"); return 1; }
    if (fread(&policy, 4, 1, f) != 1) { fprintf(stderr, "policy read fail\n"); return 1; }
    fclose(f);

    // Match Python: cap = max(1, int(capacity_units)).
    i64 cap = capacity_units < 1 ? 1 : capacity_units;
    fprintf(stderr, "[cpp] bufferpool L=%lld cap=%lld policy=%d loaded in %.3fs\n",
            (long long)L, (long long)cap, (int)policy, now() - t0);

    vector<i32> is_miss((size_t)L, 0);
    i64 admissions = 0;
    i64 evictions = 0;

    // The resident set is kept as a PARALLEL ARRAY of (unit, key) in INSERTION ORDER. This lets the
    // eviction scan reproduce Python's `max(resident, key=...)` first-wins-on-tie semantics: we scan
    // the resident array front-to-back and keep the first slot whose key is strictly greater than the
    // current best (a tie does NOT replace, so the earliest-inserted maximum wins). On a hit we update
    // the unit's key IN PLACE (preserving its insertion position, exactly like a Python dict
    // reassignment). On eviction we erase the victim slot (shifting the tail down by one) so insertion
    // order is preserved for the survivors -- the same order a Python dict would have after a del.
    // `where` maps unit-id -> its index in the resident arrays (for O(1) hit/refresh lookup).
    vector<i32> res_unit;   res_unit.reserve((size_t)cap + 1);
    vector<i64> res_key;    res_key.reserve((size_t)cap + 1);
    unordered_map<i32, i32> where;   // unit -> position in res_unit/res_key, -1 if absent
    where.reserve((size_t)cap * 2 + 16);

    double tc = now();

    if (policy == 0) {
        // ---- BELADY / MIN ----------------------------------------------------------------
        // next_use[i] = next position j>i with access[j]==access[i], else NEVER (one backward pass,
        // identical to Python _next_use_array which walks right-to-left remembering last_seen).
        vector<i64> next_use((size_t)L, NEVER);
        {
            unordered_map<i32, i64> last_seen;
            last_seen.reserve((size_t)L);   // worst case all-distinct; fine for the simulation scale
            for (i64 i = L - 1; i >= 0; i--) {
                i32 u = access[(size_t)i];
                auto it = last_seen.find(u);
                if (it != last_seen.end()) next_use[(size_t)i] = it->second;
                last_seen[u] = i;
            }
        }

        for (i64 i = 0; i < L; i++) {
            i32 u = access[(size_t)i];
            auto it = where.find(u);
            if (it != where.end() && it->second >= 0) {
                // HIT: refresh this unit's next_use in place (no eviction, not a miss).
                res_key[(size_t)it->second] = next_use[(size_t)i];
            } else {
                // MISS: stage this unit.
                is_miss[(size_t)i] = 1;
                admissions++;
                if ((i64)res_unit.size() >= cap) {
                    // Evict the FARTHEST next_use; first-wins on tie (earliest insertion).
                    i64 bestpos = 0;
                    i64 bestkey = res_key[0];
                    for (size_t p = 1; p < res_key.size(); p++) {
                        if (res_key[p] > bestkey) { bestkey = res_key[p]; bestpos = (i64)p; }
                    }
                    i32 victim = res_unit[(size_t)bestpos];
                    where.erase(victim);
                    // erase slot bestpos, shifting the tail down to preserve insertion order.
                    res_unit.erase(res_unit.begin() + bestpos);
                    res_key.erase(res_key.begin() + bestpos);
                    for (size_t p = (size_t)bestpos; p < res_unit.size(); p++)
                        where[res_unit[p]] = (i32)p;
                    evictions++;
                }
                where[u] = (i32)res_unit.size();
                res_unit.push_back(u);
                res_key.push_back(next_use[(size_t)i]);
            }
        }

    } else {
        // ---- MRD -- online reuse-distance estimate ----------------------------------------
        // last_pos[u] = position of u's previous access (any time). reuse_est = i - last_pos[u], or
        // BIG = L for a never-before-seen unit. Identical to the Python mrd path. Larger estimate =
        // predicted used-least-soon = preferred eviction victim.
        const i64 BIG = L;
        unordered_map<i32, i64> last_pos;
        last_pos.reserve((size_t)L);

        for (i64 i = 0; i < L; i++) {
            i32 u = access[(size_t)i];
            i64 reuse_est;
            auto lp = last_pos.find(u);
            if (lp != last_pos.end()) reuse_est = i - lp->second;
            else                      reuse_est = BIG;
            last_pos[u] = i;

            auto it = where.find(u);
            if (it != where.end() && it->second >= 0) {
                res_key[(size_t)it->second] = reuse_est;   // hit -> refresh estimate in place
            } else {
                is_miss[(size_t)i] = 1;
                admissions++;
                if ((i64)res_unit.size() >= cap) {
                    i64 bestpos = 0;
                    i64 bestkey = res_key[0];
                    for (size_t p = 1; p < res_key.size(); p++) {
                        if (res_key[p] > bestkey) { bestkey = res_key[p]; bestpos = (i64)p; }
                    }
                    i32 victim = res_unit[(size_t)bestpos];
                    where.erase(victim);
                    res_unit.erase(res_unit.begin() + bestpos);
                    res_key.erase(res_key.begin() + bestpos);
                    for (size_t p = (size_t)bestpos; p < res_unit.size(); p++)
                        where[res_unit[p]] = (i32)p;
                    evictions++;
                }
                where[u] = (i32)res_unit.size();
                res_unit.push_back(u);
                res_key.push_back(reuse_est);
            }
        }
    }

    fprintf(stderr, "[cpp] bufferpool sim done in %.3fs (total %.3fs): admissions=%lld evictions=%lld\n",
            now() - tc, now() - t0, (long long)admissions, (long long)evictions);

    FILE* o = fopen(outpath, "wb");
    if (!o) { fprintf(stderr, "open out fail %s\n", outpath); return 1; }
    fwrite(&L, 8, 1, o);
    if (L > 0) fwrite(is_miss.data(), 4, (size_t)L, o);
    fwrite(&admissions, 8, 1, o);
    fwrite(&evictions, 8, 1, o);
    fclose(o);
    return 0;
}
