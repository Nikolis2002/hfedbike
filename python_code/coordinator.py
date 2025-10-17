import zmq, json, numpy as np

# ─── Configuration ─────────────────────────────────────
NUM_NODES         = 16
DEALER_SOCKETS    = 5561  # for stats → global μ/σ
LOCAL_UPDATE_PORT = 5560  # for neighborhood model updates

leaders = {
    "region0":"X",
    "region1":"P",
    "region2": "A",
    "region3": "E"
}

# ─── ZMQ Setup ─────────────────────────────────────────
context = zmq.Context()

# ROUTER #1: stats in, μ/σ out
router_stats   = context.socket(zmq.ROUTER)
router_stats.bind(f"tcp://*:{DEALER_SOCKETS}")

# ROUTER #2: neighborhood updates in, dispatch back out
router_updates = context.socket(zmq.ROUTER)
router_updates.bind(f"tcp://*:{LOCAL_UPDATE_PORT}")

poller = zmq.Poller()
poller.register(router_stats,   zmq.POLLIN)
poller.register(router_updates, zmq.POLLIN)

# ─── State ────────────────────────────────────────────
node_values    = {}  # identity → {n, mean, var}
pending_updates= {}  # area → (identity, serialized_weights)

print("Coordinator started. Waiting for stats and neighborhood updates…")


while True:
    events = dict(poller.poll(timeout=1000))

    # 1) Collect stats from each node via router_stats
    if router_stats in events and events[router_stats] == zmq.POLLIN:
        identity, msg = router_stats.recv_multipart()
        payload = json.loads(msg)
        node_values[identity] = payload
        router_stats.send_multipart([identity,b"ACK"])
        print(f"→ got stats from {identity.decode()}")

        # Once we've collected from all nodes, compute and broadcast μ/σ
        if len(node_values) == NUM_NODES:
            N_total = sum(v["number_of_samples"] for v in node_values.values())

            # 2) Reconstruct each node’s sum and sum-of-squares from its mean & variance:
            sum_total = sum(v["mean"] * v["number_of_samples"]
                            for v in node_values.values())

            ss_total  = sum(
                (v["variance"] + v["mean"]**2) * v["number_of_samples"]
                for v in node_values.values()
            )

            mu    = sum_total / N_total
            var   = ss_total / N_total - mu * mu
            sigma = float(np.sqrt(max(var, 0.0)))
            print(f"Global μ={mu:.3f}, σ={sigma:.3f}")

            payload_bytes = json.dumps({"mu": mu, "sigma": sigma}).encode("utf-8")
            for identity in node_values:
                # send exactly two frames: [identity, payload]
                router_stats.send_multipart([identity, payload_bytes])
                print(f"sent μ/σ to {identity.decode()}")
            node_values.clear()


    if router_updates in events and events[router_updates] & zmq.POLLIN:
        # 1) Receive one dealer’s update
        identity, empty, raw = router_updates.recv_multipart()
        data = json.loads(raw.decode("utf-8"))
        area, msg = data["area"], data["msg"]
        pending_updates[area] = (identity, msg)
        print(f"→ stored update for area={area!r} "
            f"({len(pending_updates)}/{len(leaders)})")

        # 2) Once *all* areas have sent, reply to each with the *other* areas’ weights
        if len(pending_updates) == len(leaders):
            print("All neighborhood updates received; dispatching…")
            for recv_area, (recv_id, _) in pending_updates.items():
                # Build a dict of area→msg for everyone except recv_area
                others = {
                    area: msg
                    for area, (sid, msg) in pending_updates.items()
                    if area != recv_area
                }
                reply = json.dumps({
                    "msgs": others
                }).encode("utf-8")

                router_updates.send_multipart([recv_id, b"", reply])
                print(f"   sent updates for {list(others)} → {recv_area!r}")

            # 3) Clear for the next round
            pending_updates.clear()

    
