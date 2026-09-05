#!/usr/bin/env bash
set -euo pipefail

export LC_ALL=C
export PATH="$PATH:/usr/sbin:/sbin"
MODE="${1:-auto}"
case "$MODE" in
    off) exit 0 ;;
    auto|print) ;;
    *) printf 'Unknown firewall mode: %s\n' "$MODE" >&2; exit 2 ;;
esac
if [ "$#" -gt 0 ]; then shift; fi
PORTS=(5353)
for port in "$@"; do
    case "$port" in
        ''|*[!0-9]*) printf 'Invalid UDP port: %s\n' "$port" >&2; exit 2 ;;
    esac
    if [ "${#port}" -gt 5 ] || [ "$((10#$port))" -lt 1 ] || [ "$((10#$port))" -gt 65535 ]; then
        printf 'Invalid UDP port: %s\n' "$port" >&2
        exit 2
    fi
    PORTS+=("$((10#$port))")
done
if [ "$(uname -s)" != Linux ]; then
    printf '%s\n' 'Automatic firewall setup is available on Linux. Allow latticed in your firewall if prompted.'
    exit 0
fi
if ! command -v ip >/dev/null 2>&1; then
    printf '%s\n' 'Cannot find LAN subnets: install iproute2 or configure your firewall manually.' >&2
    exit 1
fi
ROUTES="$(ip -4 route show scope link)"
NETWORKS="$(printf '%s\n' "$ROUTES" | awk '
    $1 != "default" {
        iface = ""
        for (i = 2; i < NF; i++) if ($i == "dev") iface = $(i + 1)
        if (iface == "" || iface == "lo" || iface ~ /^tailscale/) next
        count = split($1, cidr, "/")
        if (count > 2 || (count == 2 && (cidr[2] !~ /^[0-9]+$/ || cidr[2] < 1 || cidr[2] > 32))) next
        if (split(cidr[1], octet, ".") != 4) next
        valid = 1
        for (i = 1; i <= 4; i++) if (octet[i] !~ /^[0-9]+$/ || octet[i] > 255) valid = 0
        if (!valid || octet[1] == 0 || octet[1] == 127 || octet[1] >= 224) next
        print cidr[1] "/" (count == 2 ? cidr[2] : 32), iface
    }
' | sort -u)"
if [ -z "$NETWORKS" ]; then
    printf '%s\n' 'No directly connected IPv4 LAN subnet found. Configure your firewall manually or connect to the LAN.' >&2
    exit 1
fi
PORT_LIST="$(printf '%s\n' "${PORTS[@]}" | sort -nu)"
SUDO=""
privileges() {
    if [ "$EUID" -ne 0 ] && [ -z "$SUDO" ]; then
        if ! command -v sudo >/dev/null 2>&1; then
            printf '%s\n' 'Firewall setup requires sudo. Use --firewall print to see the commands.' >&2
            exit 1
        fi
        SUDO="$(command -v sudo)"
    fi
}
admin() {
    if [ -n "$SUDO" ]; then "$SUDO" -- "$@"; else "$@"; fi
}
show_command() {
    printf 'sudo'
    printf ' %q' "$@"
    printf '\n'
}
ufw_rules() {
    local subnet port
    while read -r subnet; do
        while read -r port; do
            if [ "$MODE" = print ]; then
                show_command "$UFW" allow from "$subnet" to any port "$port" proto udp
            else
                admin "$UFW" allow from "$subnet" to any port "$port" proto udp
            fi
        done <<< "$PORT_LIST"
    done < <(printf '%s\n' "$NETWORKS" | awk '{print $1}' | sort -u)
}
firewalld_rules() {
    local subnet iface zone rule port
    while read -r subnet iface; do
        if ! zone="$(admin "$FIREWALLD" --get-zone-of-interface="$iface" 2>&1)" && [ "$zone" != 'no zone' ]; then
            printf '%s\n' "$zone" >&2
            return 1
        fi
        if [ -z "$zone" ] || [ "$zone" = 'no zone' ]; then
            zone="$(admin "$FIREWALLD" --get-default-zone)"
        fi
        while read -r port; do
            rule="rule family=\"ipv4\" source address=\"$subnet\" port port=\"$port\" protocol=\"udp\" accept"
            if [ "$MODE" = print ]; then
                show_command "$FIREWALLD" --zone="$zone" --add-rich-rule="$rule"
                show_command "$FIREWALLD" --permanent --zone="$zone" --add-rich-rule="$rule"
            else
                if ! admin "$FIREWALLD" --zone="$zone" --query-rich-rule="$rule" >/dev/null; then
                    admin "$FIREWALLD" --zone="$zone" --add-rich-rule="$rule"
                fi
                if ! admin "$FIREWALLD" --permanent --zone="$zone" --query-rich-rule="$rule" >/dev/null; then
                    admin "$FIREWALLD" --permanent --zone="$zone" --add-rich-rule="$rule"
                fi
            fi
        done <<< "$PORT_LIST"
    done <<< "$NETWORKS"
}
UFW="$(command -v ufw || true)"
if [ -n "$UFW" ]; then
    if [ "$MODE" = print ]; then
        printf '%s\n' 'If UFW is active, these commands allow local IPv4 peers:'
        ufw_rules
    else
        privileges
        STATUS="$(admin "$UFW" status)"
        if [[ "$STATUS" == *'Status: active'* ]]; then
            ufw_rules
            exit 0
        fi
    fi
fi
FIREWALLD="$(command -v firewall-cmd || true)"
if [ -n "$FIREWALLD" ] && "$FIREWALLD" --state >/dev/null 2>&1; then
    if [ "$MODE" = auto ]; then privileges; fi
    firewalld_rules
    exit 0
fi
if [ "$MODE" = print ] && [ -n "$UFW" ]; then exit 0; fi
printf '%s\n' 'No active UFW or firewalld detected. If another firewall is active, allow inbound UDP from your LAN for these ports:'
printf '  %s\n' "$PORT_LIST"
