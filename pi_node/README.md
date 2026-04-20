# Hybrid Deployment — Raspberry Pi runs nodeA

One Raspberry Pi 4 joins the 16-client federation as **nodeA** (Region 2,
subzone 0, IP `192.168.1.208`). The other 15 nodes and the coordinator
run on a host (laptop or desktop) over the existing `my_macvlan`
network. Both sides share the same physical L2 segment, so the Pi's
container talks to the host's containers at L2 as if they were all
on the same switch --- because they are.

This directory is everything that needs to live on the Pi. It pulls
from `git` and runs with three commands.

**This is a template.** The configuration targets a Pi 4 running nodeA,
but everything here (the compose file, the macvlan setup script, the
data layout) generalises to any Linux edge device running Docker.
To run a *different* node on a *different* device:

- edit `NODE_ID` and `NEIGHBORHOOD` in `docker-compose.pi-nodeA.yml`
  (see the header comment there for the valid combinations);
- edit the `ipv4_address` to match the IP expected for that node
  (see `ip_mapping` in `federated/p2p_node/v2_node.py`);
- copy the corresponding `region{N}_subzone{M}_bike_usage.csv` into
  `data/entire_year/`;
- rebuild the container image on the target architecture with
  `docker build -t federated-node ..` from the repo root.

The setup has been validated on Raspberry Pi 4 (2 GB — the smallest
Pi 4 model) and on x86 containers; Jetson Nano, ODROID, BeagleBone, or
similar SBCs with a macvlan-capable Ethernet interface will work
identically. On 2 GB boards the first `docker build` requires swap
(see the main project README for the one-line swap tweak).

## Network topology required

```
    192.168.1.0/24  (macvlan)
    ┌───────────────────────────┐
    │                           │
    │  ┌────────┐   ┌────────┐  │
    │  │ host   │   │  Pi    │  │   host:   eth0 or en0 parent iface
    │  │ 15x    │   │ nodeA  │  │   Pi:     eth0 or end0 parent iface
    │  │ +coord │   │  .208  │  │   both plugged into the same switch
    │  └────────┘   └────────┘  │   or direct crossover cable
    │                           │
    └───────────────────────────┘
```

The physical link between host and Pi must be Ethernet (direct cable or
common switch). Wi-Fi does not support macvlan. `192.168.1.0/24` should
not overlap your home network — use an isolated switch or configure the
subnet elsewhere.

## Setup on the Pi

```bash
# 1. clone and pick the hybrid branch
git clone https://github.com/Nikolis2002/diplwmatikh.git
cd diplwmatikh
git checkout hybrid
cd pi_node

# 2. build the ARM64 container image from the repo's Dockerfile (first run only, ~10 min)
docker build -t federated-node ..

# 3. create the macvlan network (requires sudo)
sudo ./setup_macvlan.sh

# 4. bring up nodeA
docker compose -f docker-compose.pi-nodeA.yml up
```

If your Pi's Ethernet interface is not `eth0` (Raspberry Pi OS Bookworm
uses `end0`, Ubuntu sometimes uses `enp...`), override the parent:

```bash
sudo PARENT_IFACE=end0 ./setup_macvlan.sh
```

Override the subnet/gateway the same way if your lab network is not
`192.168.1.0/24`:

```bash
sudo SUBNET=10.10.1.0/24 GATEWAY=10.10.1.1 ./setup_macvlan.sh
```

## Setup on the host

Back on the host (where the other 15 nodes live):

```bash
cd diplwmatikh
git checkout hybrid
NODE_SCRIPT=v2_node.py docker compose -f docker-compose.hybrid-host.yml up
```

`docker-compose.hybrid-host.yml` is identical to the main
`docker-compose.yml` except `nodeA` is omitted — the Pi owns that slot.

## What you get

- `pi_node/logs/A.log` — the Pi's full weekly protocol log (ring-reduce
  rounds, training epochs, weekly metric writes).
- `pi_node/results/A_results.csv` — weekly `base_mae`, `freeze_mae`,
  `fed_mae` for nodeA, identical schema to the other 15.
- The host's existing `logs/node{B,C,...}/` and `results/current/` fill
  up normally; from the federation's perspective, nothing changed
  except that nodeA's IP is now backed by physical hardware.

## Recording the demo video

On the Pi, run three panes side-by-side:

```bash
# Pane 1 — Pi's container log
docker logs -f nodeA

# Pane 2 — nodeA's weekly results CSV, tailed as it grows
tail -f pi_node/results/A_results.csv

# Pane 3 — watch the ring-reduce traffic leave the Pi
sudo tcpdump -ni <parent_iface> 'host 192.168.1.208 and (port 5555 or port 5562)'
```

Screen-record all three. One weekly cycle is ~90 seconds of footage
and contains every artifact you need to show on the poster: a real
ARM64 device completing local training, sending/receiving ring parcels
with the host's containers, and writing a per-week MAE row.

## Troubleshooting

**`docker network create` fails with "operation not supported on transport endpoint".**
You don't have kernel modules for macvlan. On a minimal OS image: `sudo modprobe macvlan`.

**Pi can't reach the host's containers.**
Both macvlan networks must share the same physical subnet AND the Pi
must be able to ARP the host's IPs. Verify with
`ip neigh` and `ping 192.168.1.203` (coordinator).

**nodeA on the Pi exits with "Address already in use".**
Something on the Pi is already bound to `0.0.0.0:5555` or `:5562`.
Common culprit: an old container. `docker ps -a` and remove any stale
`nodeA` instance.

**nodeA blocks waiting for peers.**
The Pi is the leader for Region 2. It binds a ROUTER socket on
`:5562` and expects nodeB, C, D to connect. If the host's containers
aren't up yet, nodeA waits. Bring the host up first, then the Pi.
