#!/usr/bin/env bash
# Create (or refresh) the macvlan network the Pi container needs.
#
# Requires:
#   - root (for Docker network operations and interface changes)
#   - the Pi's Ethernet interface to be called eth0 (set PARENT_IFACE below
#     if yours is different, e.g. end0 on Raspberry Pi OS Bookworm)
#
# After this succeeds, bring up the node with:
#   docker compose -f docker-compose.pi-nodeA.yml up

set -euo pipefail

NET_NAME=${NET_NAME:-pi_macvlan}
PARENT_IFACE=${PARENT_IFACE:-eth0}
SUBNET=${SUBNET:-192.168.1.0/24}
GATEWAY=${GATEWAY:-192.168.1.1}

if ! command -v docker >/dev/null 2>&1; then
    echo "docker is not installed; install Docker Engine first." >&2
    exit 1
fi

if ! ip link show "$PARENT_IFACE" >/dev/null 2>&1; then
    echo "interface $PARENT_IFACE not found. Set PARENT_IFACE=... to override." >&2
    echo "Available interfaces:" >&2
    ip -br link | awk '{print "  " $1}' >&2
    exit 1
fi

if docker network inspect "$NET_NAME" >/dev/null 2>&1; then
    echo "macvlan '$NET_NAME' already exists — leaving it in place."
    docker network inspect "$NET_NAME" --format '  driver={{.Driver}} parent={{.Options.parent}} subnet={{(index .IPAM.Config 0).Subnet}}'
    exit 0
fi

echo "creating docker macvlan network '$NET_NAME' on parent=$PARENT_IFACE subnet=$SUBNET gateway=$GATEWAY ..."
docker network create -d macvlan \
    --subnet="$SUBNET" \
    --gateway="$GATEWAY" \
    -o parent="$PARENT_IFACE" \
    "$NET_NAME"

echo "done. Bring up nodeA with:"
echo "    docker compose -f docker-compose.pi-nodeA.yml up"
