"""
VRP Engine — Vehicle Routing Problem with Time Windows & Capacity
Python port of cVRPTWreadwriteEverything.cs

Key design decisions preserved from original C#:
  - Dummy-warehouse pickup-delivery trick: each delivery node i gets a phantom
    depot copy at index (original_n + i - 1). Pickup-delivery pair
    (phantom_depot → real_stop) forces the truck to "load" at depot before
    delivering. This models all cargo originating at the depot.
  - Per-vehicle transit callbacks with driving/service efficiency multipliers.
  - Two solve modes: fresh (PATH_CHEAPEST_ARC) and warm-start from prior solution.
  - GuidedLocalSearch metaheuristic with same operator set as C# production code.
  - OSRM public API for travel times, batched ≤100×100, cached to disk by hash.
"""

import hashlib
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

# ── Cache ──────────────────────────────────────────────────────────────────────
CACHE_DIR = Path(__file__).parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)

def _matrix_cache_key(lats, lons):
    s = ",".join(f"{la:.6f},{lo:.6f}" for la, lo in zip(lats, lons))
    return hashlib.md5(s.encode()).hexdigest()

def _load_cached(key):
    p = CACHE_DIR / f"{key}.npy"
    return np.load(str(p)).tolist() if p.exists() else None

def _save_cached(key, matrix):
    np.save(str(CACHE_DIR / f"{key}.npy"), np.array(matrix))


# ── OSRM distance matrix ───────────────────────────────────────────────────────
OSRM_URL   = "http://router.project-osrm.org/table/v1/driving/"
OSRM_BATCH = 100

def build_time_matrix(lats, lons, minimum_drive_time=5):
    """N×N travel-time matrix in minutes via OSRM, batched and disk-cached."""
    n   = len(lats)
    key = _matrix_cache_key(lats, lons)
    hit = _load_cached(key)
    if hit is not None:
        return hit

    # OSRM expects lon,lat
    coords_str = ";".join(f"{lo:.6f},{la:.6f}" for la, lo in zip(lats, lons))
    matrix = [[0] * n for _ in range(n)]

    src_start = 0
    while src_start < n:
        src_end  = min(src_start + OSRM_BATCH, n)
        dst_start = 0
        while dst_start < n:
            dst_end = min(dst_start + OSRM_BATCH, n)
            src_param = ";".join(str(i) for i in range(src_start, src_end))
            dst_param = ";".join(str(i) for i in range(dst_start, dst_end))
            url = (f"{OSRM_URL}{coords_str}"
                   f"?sources={src_param}&destinations={dst_param}"
                   f"&annotations=duration")
            for attempt in range(3):
                try:
                    r = requests.get(url, timeout=60)
                    r.raise_for_status()
                    data = r.json()
                    break
                except Exception as e:
                    if attempt == 2:
                        raise RuntimeError(f"OSRM failed after 3 attempts: {e}")
                    time.sleep(2 ** attempt)
            durations = data["durations"]
            for i, si in enumerate(range(src_start, src_end)):
                for j, dj in enumerate(range(dst_start, dst_end)):
                    raw = durations[i][j]
                    if raw is None or raw == 0:
                        matrix[si][dj] = 0
                    else:
                        matrix[si][dj] = max(int(raw / 60), minimum_drive_time)
            dst_start = dst_end
        src_start = src_end

    _save_cached(key, matrix)
    return matrix


# ── Input loading ──────────────────────────────────────────────────────────────

def load_from_excel(file_obj):
    """Read workbook with sheets: locations, vehicles, parameters, solution."""
    xl = pd.ExcelFile(file_obj)
    out = {}
    for sheet in ["locations", "vehicles", "parameters", "solution"]:
        out[sheet] = (pd.read_excel(xl, sheet_name=sheet, header=None)
                      if sheet in xl.sheet_names else pd.DataFrame())
    return out

def load_from_csvs(locations, vehicles, parameters, solution=None):
    """Load from 4 CSV paths (for standalone testing)."""
    out = {
        "locations":  pd.read_csv(locations,  header=None),
        "vehicles":   pd.read_csv(vehicles,   header=None),
        "parameters": pd.read_csv(parameters, header=None),
        "solution":   (pd.read_csv(solution, comment="#", header=None)
                       if solution and Path(solution).exists()
                       else pd.DataFrame()),
    }
    return out


# ── Parameters ─────────────────────────────────────────────────────────────────

class Params:
    DroppedOrderCost      = 100
    MinuteCost            = 1
    RouteTimeLimit        = 14 * 60
    SearchTimeLimit       = 120
    MinimumDrivingTime    = 5
    TimeWindowPenalty1    = 1
    TimeWindowPenalty2    = 1
    TimeWindowPenalty3    = 1
    TimeWindowPenalty4    = 1
    StartFromLastSolution = False

def _parse_params(df):
    p = Params()
    for _, row in df.iterrows():
        name = str(row[0]).strip()
        val  = str(row[1]).strip()
        mapping = {
            "DroppedOrderCost":          ("DroppedOrderCost",      int),
            "MinuteCost":                ("MinuteCost",             int),
            "RouteTimeLimit":            ("RouteTimeLimit",         int),
            "SearchParametersTimeLimit": ("SearchTimeLimit",        int),
            "MinimumDriveTime":          ("MinimumDrivingTime",     int),
            "TimeWindowPenalty1":        ("TimeWindowPenalty1",     int),
            "TimeWindowPenalty2":        ("TimeWindowPenalty2",     int),
            "TimeWindowPenalty3":        ("TimeWindowPenalty3",     int),
            "TimeWindowPenalty4":        ("TimeWindowPenalty4",     int),
            "StartFromLastSolution":     ("StartFromLastSolution",  lambda v: v.lower() == "true"),
        }
        if name in mapping:
            attr, cast = mapping[name]
            setattr(p, attr, cast(val))
    return p


# ── Main solver ────────────────────────────────────────────────────────────────

def run_vrp(sheets, time_matrix_override=None):
    """
    Run the VRP.  sheets = dict with keys locations/vehicles/parameters/solution.
    time_matrix_override: optional pre-built N×N matrix (for testing without OSRM).
    Returns dict with status, routes, summary, csv.
    """
    params = _parse_params(sheets["parameters"])

    # ── locations ──────────────────────────────────────────────────────────────
    loc = sheets["locations"]
    orig_n  = len(loc)               # row 0 = depot
    total_n = orig_n * 2 - 1        # real nodes + dummy depot copies

    lats          = [0.0] * total_n
    lons          = [0.0] * total_n
    demands       = [0.0] * total_n
    svc_times     = [0]   * total_n
    tw            = [(0, 9999, 0, 0)] * total_n  # (open, close, blk_s, blk_e)

    depot_lat = float(loc.iloc[0, 0])
    depot_lon = float(loc.iloc[0, 1])
    depot_svc = int(loc.iloc[0, 3])
    depot_tw  = (int(loc.iloc[0, 4]), int(loc.iloc[0, 5]),
                 int(loc.iloc[0, 6]), int(loc.iloc[0, 7]))

    for i in range(orig_n):
        r = loc.iloc[i]
        lats[i]      = float(r[0])
        lons[i]      = float(r[1])
        demands[i]   = -float(r[2])   # negative = delivery (reduces load)
        svc_times[i] = int(r[3])
        tw[i]        = (int(r[4]), int(r[5]), int(r[6]), int(r[7]))
        if i > 0:                      # create dummy depot copy
            di = orig_n + i - 1
            lats[di]      = depot_lat
            lons[di]      = depot_lon
            demands[di]   = float(r[2])   # positive = pickup
            svc_times[di] = depot_svc
            tw[di]        = depot_tw

    # ── vehicles ───────────────────────────────────────────────────────────────
    veh = sheets["vehicles"]
    n_veh       = len(veh)
    capacities  = [int(veh.iloc[v, 0]) for v in range(n_veh)]
    fixed_costs = [int(veh.iloc[v, 1]) for v in range(n_veh)]
    drv_eff     = [float(veh.iloc[v, 2]) if veh.shape[1] > 2 else 1.0 for v in range(n_veh)]
    svc_eff     = [float(veh.iloc[v, 3]) if veh.shape[1] > 3 else 1.0 for v in range(n_veh)]

    # ── distance matrix ────────────────────────────────────────────────────────
    if time_matrix_override is not None:
        tmat = time_matrix_override
    else:
        tmat = build_time_matrix(lats, lons, params.MinimumDrivingTime)

    # ── pickup-delivery pairs  (dummy_depot_i → real_stop_i) ──────────────────
    pd_pairs = [(orig_n + i - 1, i) for i in range(1, orig_n)]

    # ── warm-start solution ────────────────────────────────────────────────────
    initial_routes = None
    if params.StartFromLastSolution and not sheets["solution"].empty:
        initial_routes = _parse_solution(sheets["solution"], n_veh)

    # ── OR-Tools model ─────────────────────────────────────────────────────────
    manager = pywrapcp.RoutingIndexManager(total_n, n_veh, 0)
    routing = pywrapcp.RoutingModel(manager)

    # Per-vehicle transit callbacks
    cb_indices = []
    for v in range(n_veh):
        de, se = drv_eff[v], svc_eff[v]
        def _make_cb(de=de, se=se):
            def cb(fi, ti):
                fn = manager.IndexToNode(fi)
                tn = manager.IndexToNode(ti)
                return int(tmat[fn][tn] * de) + int(svc_times[fn] * se)
            return cb
        ci = routing.RegisterTransitCallback(_make_cb())
        cb_indices.append(ci)
        routing.SetArcCostEvaluatorOfVehicle(ci, v)

    for v in range(n_veh):
        routing.SetFixedCostOfVehicle(fixed_costs[v], v)

    # Time dimensions
    routing.AddDimensionWithVehicleTransits(cb_indices, 0, 9999, False, "Time")
    routing.AddDimensionWithVehicleTransits(
        cb_indices, 0, params.RouteTimeLimit, True, "TransitTime")
    time_dim    = routing.GetDimensionOrDie("Time")
    transit_dim = routing.GetDimensionOrDie("TransitTime")
    time_dim.SetSpanCostCoefficientForAllVehicles(params.MinuteCost)

    # Time windows per node
    for i in range(1, total_n):
        idx = manager.NodeToIndex(i)
        open_, close_, blk_s, blk_e = tw[i]
        time_dim.CumulVar(idx).SetRange(open_, close_)
        if blk_e > blk_s:
            time_dim.CumulVar(idx).RemoveInterval(blk_s, blk_e)

    for v in range(n_veh):
        idx = routing.Start(v)
        open_, close_, blk_s, blk_e = tw[0]
        time_dim.CumulVar(idx).SetRange(open_, close_)

    for v in range(n_veh):
        routing.AddVariableMinimizedByFinalizer(time_dim.CumulVar(routing.Start(v)))
        routing.AddVariableMinimizedByFinalizer(time_dim.CumulVar(routing.End(v)))

    # Capacity dimension
    def _demand_cb(fi):
        return int(demands[manager.IndexToNode(fi)])
    dci = routing.RegisterUnaryTransitCallback(_demand_cb)
    routing.AddDimensionWithVehicleCapacity(dci, 0, capacities, True, "Capacity")

    # Pickup-delivery constraints — Python OR-Tools uses var == var2 directly
    solver = routing.solver()
    for (pickup_node, delivery_node) in pd_pairs:
        pi = manager.NodeToIndex(pickup_node)
        di = manager.NodeToIndex(delivery_node)
        routing.AddPickupAndDelivery(pi, di)
        solver.Add(routing.VehicleVar(pi) == routing.VehicleVar(di))
        solver.Add(time_dim.CumulVar(pi) <= time_dim.CumulVar(di))

    # Allow dropping nodes
    for i in range(1, total_n):
        routing.AddDisjunction(
            [manager.NodeToIndex(i)], params.DroppedOrderCost)

    # ── Search parameters ──────────────────────────────────────────────────────
    sp = pywrapcp.DefaultRoutingSearchParameters()
    sp.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
    sp.time_limit.seconds = max(params.SearchTimeLimit, 5)
    sp.log_search = False

    ops = sp.local_search_operators
    ops.use_cross_exchange          = pywrapcp.BOOL_TRUE
    ops.use_extended_swap_active    = pywrapcp.BOOL_TRUE
    ops.use_inactive_lns            = pywrapcp.BOOL_TRUE
    ops.use_make_chain_inactive     = pywrapcp.BOOL_TRUE
    ops.use_relocate_and_make_active = pywrapcp.BOOL_TRUE
    ops.use_relocate_neighbors      = pywrapcp.BOOL_TRUE

    if initial_routes and params.StartFromLastSolution:
        routing.CloseModelWithParameters(sp)
        try:
            init_asgn = routing.ReadAssignmentFromRoutes(initial_routes, True)
            solution  = routing.SolveFromAssignmentWithParameters(init_asgn, sp)
        except Exception:
            sp.first_solution_strategy = (
                routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
            solution = routing.SolveWithParameters(sp)
    else:
        sp.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
        solution = routing.SolveWithParameters(sp)

    if solution is None:
        return {"status": "infeasible", "message": "Solver found no feasible solution."}

    return _extract(solution, routing, manager, time_dim, transit_dim,
                    routing.GetDimensionOrDie("Capacity"),
                    n_veh, lats, lons, svc_times, orig_n)


# ── Solution extraction ────────────────────────────────────────────────────────

def _extract(sol, routing, manager, tdim, trdim, ldim,
             n_veh, lats, lons, svc_times, orig_n):
    dropped = []
    for idx in range(routing.Size()):
        if routing.IsStart(idx) or routing.IsEnd(idx):
            continue
        if sol.Value(routing.NextVar(idx)) == idx:
            node = manager.IndexToNode(idx)
            if node < orig_n:
                dropped.append(node)

    routes    = []
    total_time = 0

    for v in range(n_veh):
        stops   = []
        idx     = routing.Start(v)
        stop_no = 0
        while not routing.IsEnd(idx):
            node  = manager.IndexToNode(idx)
            if node < orig_n:         # skip dummy depot copies
                tv = tdim.CumulVar(idx)
                lv = ldim.CumulVar(idx)
                stops.append({
                    "stop":     stop_no,
                    "node":     node,
                    "lat":      round(lats[node], 6),
                    "lon":      round(lons[node], 6),
                    "early":    sol.Min(tv),
                    "late":     sol.Max(tv),
                    "service":  svc_times[node],
                    "load":     sol.Max(lv),
                    "is_depot": node == 0,
                })
                stop_no += 1
            idx = sol.Value(routing.NextVar(idx))

        # final depot return
        node = manager.IndexToNode(idx)
        tv   = tdim.CumulVar(idx)
        stops.append({
            "stop":     stop_no,
            "node":     0,
            "lat":      round(lats[0], 6),
            "lon":      round(lons[0], 6),
            "early":    sol.Min(tv),
            "late":     sol.Max(tv),
            "service":  0,
            "load":     0,
            "is_depot": True,
        })

        tt   = sol.Min(trdim.CumulVar(idx))
        total_time += tt
        real = [s for s in stops if not s["is_depot"]]
        if real:
            routes.append({"vehicle": v, "stops": stops,
                           "num_stops": len(real), "duration": tt})

    # CSV
    rows = ["route,stop,node,early,late,service,lat,lon"]
    for r in routes:
        for s in r["stops"]:
            rows.append(f"{r['vehicle']},{s['stop']},{s['node']},"
                        f"{s['early']},{s['late']},{s['service']},"
                        f"{s['lat']},{s['lon']}")

    return {
        "status":        "ok",
        "routes":        routes,
        "total_routes":  len(routes),
        "total_stops":   sum(r["num_stops"] for r in routes),
        "total_time":    total_time,
        "dropped_nodes": dropped,
        "csv":           "\n".join(rows),
    }


def _parse_solution(df, n_veh):
    """Build initial_routes list from solution CSV (no header)."""
    rd = {}
    for _, row in df.iterrows():
        try:
            route_id = int(row[0]);  node = int(row[2])
        except (ValueError, TypeError):
            continue
        if node == 0:
            continue
        rd.setdefault(route_id, []).append(node)
    if not rd:
        return None
    max_r = max(rd.keys())
    return [rd.get(v, []) for v in range(max(max_r + 1, n_veh))]


# ── Standalone test ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import math as _math

    def _haversine_matrix(lats, lons, min_time=5):
        """Haversine approximation — used only for offline testing."""
        n = len(lats)
        m = [[0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                la1, lo1 = _math.radians(lats[i]), _math.radians(lons[i])
                la2, lo2 = _math.radians(lats[j]), _math.radians(lons[j])
                a = (_math.sin((la2-la1)/2)**2
                     + _math.cos(la1)*_math.cos(la2)*_math.sin((lo2-lo1)/2)**2)
                km = 6371 * 2 * _math.asin(_math.sqrt(a))
                m[i][j] = max(int(km / 50 * 60), min_time)
        return m

    uploads = Path("/sessions/beautiful-blissful-ritchie/mnt/uploads")
    sheets  = load_from_csvs(
        uploads / "locations.csv",
        uploads / "vehicles.csv",
        uploads / "parameters.csv",
        uploads / "solution.csv",
    )

    # Small test: 10 stops + depot, 3 vehicles, 10-sec limit, fresh solve
    sheets["locations"]  = sheets["locations"].iloc[:11]
    sheets["vehicles"]   = sheets["vehicles"].iloc[:3]
    pf = sheets["parameters"]
    pf.loc[pf[0]=="StartFromLastSolution",       1] = "false"
    pf.loc[pf[0]=="SearchParametersTimeLimit",   1] = "10"
    sheets["solution"] = pd.DataFrame()

    print("Building haversine matrix for 10 stops + 10 dummy depots...")
    loc = sheets["locations"]
    orig_n = len(loc)
    total_n = orig_n * 2 - 1
    lats_all = [float(loc.iloc[i,0]) for i in range(orig_n)]
    lons_all = [float(loc.iloc[i,1]) for i in range(orig_n)]
    depot_la, depot_lo = lats_all[0], lons_all[0]
    for i in range(1, orig_n):
        lats_all.append(depot_la)
        lons_all.append(depot_lo)

    tmat = _haversine_matrix(lats_all, lons_all)
    print(f"Matrix: {len(tmat)}x{len(tmat)}")

    print("Running VRP solver...")
    result = run_vrp(sheets, time_matrix_override=tmat)

    print(f"\nStatus: {result['status']}")
    if result["status"] == "ok":
        print(f"Routes:      {result['total_routes']}")
        print(f"Total stops: {result['total_stops']}")
        print(f"Dropped:     {result['dropped_nodes']}")
        for r in result["routes"]:
            nodes = [s["node"] for s in r["stops"]]
            print(f"  Vehicle {r['vehicle']}: {r['num_stops']} stops, "
                  f"{r['duration']} min | {nodes}")
    else:
        print("Message:", result.get("message"))
