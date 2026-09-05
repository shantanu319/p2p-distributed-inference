#!/usr/bin/env bash
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TEST_DIR="$(mktemp -d)"
trap 'rm -rf "$TEST_DIR"' EXIT
BIN="$TEST_DIR/bin"
mkdir -p "$BIN"
export CALLS="$TEST_DIR/calls" ROUTES="$TEST_DIR/routes" RULES="$TEST_DIR/rules"
export UFW_STATE=active FIREWALLD_STATE=stopped
touch "$CALLS" "$RULES"
cat > "$ROUTES" <<'EOF'
default dev wlan0 scope link
192.168.7.0/24 dev wlan0 proto kernel scope link src 192.168.7.62
192.168.7.0/24 dev wlan0 proto kernel scope link src 192.168.7.63
10.2.0.0/16 dev eth0 proto kernel scope link src 10.2.0.2
127.0.0.0/8 dev lo scope link
100.64.0.0/10 dev tailscale0 scope link
0.0.0.0/0 dev wlan0 scope link
EOF
cat > "$BIN/uname" <<'EOF'
#!/usr/bin/env bash
printf 'Linux\n'
EOF
cat > "$BIN/ip" <<'EOF'
#!/usr/bin/env bash
printf 'ip|%s\n' "$*" >> "$CALLS"
cat "$ROUTES"
EOF
cat > "$BIN/sudo" <<'EOF'
#!/usr/bin/env bash
printf 'sudo|%s\n' "$*" >> "$CALLS"
if [ "$1" = -- ]; then shift; fi
exec "$@"
EOF
cat > "$BIN/ufw" <<'EOF'
#!/usr/bin/env bash
printf 'ufw|%s\n' "$*" >> "$CALLS"
if [ "$1" = status ]; then printf 'Status: %s\n' "$UFW_STATE"; fi
EOF
cat > "$BIN/firewall-cmd" <<'EOF'
#!/usr/bin/env bash
printf 'firewalld|%s\n' "$*" >> "$CALLS"
case "$1" in
    --state) [ "$FIREWALLD_STATE" = running ]; exit ;;
    --get-zone-of-interface=wlan0) printf 'home\n'; exit ;;
    --get-zone-of-interface=eth0) printf 'no zone\n' >&2; exit 2 ;;
    --get-default-zone) printf 'public\n'; exit ;;
esac
KEY="${*//--query-rich-rule=/--add-rich-rule=}"
if [[ "$*" == *--query-rich-rule=* ]]; then
    grep -Fxq -- "$KEY" "$RULES"
else
    printf '%s\n' "$KEY" >> "$RULES"
fi
EOF
chmod +x "$BIN"/*
export PATH="$BIN:/usr/bin:/bin"
fail() { printf 'FAIL: %s\n' "$1" >&2; exit 1; }
contains() { grep -Fq -- "$2" "$1" || fail "missing: $2"; }
absent() { if grep -Eq -- "$2" "$1"; then fail "unexpected: $2"; fi; }
reset_calls() { : > "$CALLS"; }

bash "$REPO/scripts/firewall.sh" auto 47900 47901 > "$TEST_DIR/output"
for subnet in 192.168.7.0/24 10.2.0.0/16; do
    for port in 5353 47900 47901; do
        contains "$CALLS" "ufw|allow from $subnet to any port $port proto udp"
    done
done
[ "$(grep -c '^ufw|allow' "$CALLS")" = 6 ] || fail 'duplicate UFW rules'
absent "$CALLS" 'allow from (127\.|100\.64\.|0\.)|ufw\|allow [0-9]|enable|reload|reset'
if [ "$EUID" -ne 0 ]; then contains "$CALLS" "sudo|-- $BIN/ufw status"; fi

reset_calls
UFW_STATE=inactive bash "$REPO/scripts/firewall.sh" auto 47900 > "$TEST_DIR/output"
absent "$CALLS" 'ufw\|allow|--add-|enable|reload|reset'
contains "$TEST_DIR/output" 'No active UFW or firewalld detected'

reset_calls
bash "$REPO/scripts/firewall.sh" print 47900 47901 > "$TEST_DIR/output"
contains "$TEST_DIR/output" 'allow from 192.168.7.0/24 to any port 47901 proto udp'
absent "$CALLS" 'sudo\||ufw\|'

reset_calls
bash "$REPO/scripts/firewall.sh" off 47900 > "$TEST_DIR/output"
[ ! -s "$CALLS" ] || fail 'off mode executed commands'
if bash "$REPO/scripts/firewall.sh" auto '47900;echo unsafe' > "$TEST_DIR/output" 2>&1; then
    fail 'invalid port accepted'
fi

reset_calls
export UFW_STATE=inactive FIREWALLD_STATE=running
bash "$REPO/scripts/firewall.sh" auto 47900 47901 > "$TEST_DIR/output"
contains "$CALLS" '--zone=home --add-rich-rule=rule family="ipv4" source address="192.168.7.0/24" port port="47900" protocol="udp" accept'
contains "$CALLS" '--permanent --zone=public --add-rich-rule=rule family="ipv4" source address="10.2.0.0/16" port port="47901" protocol="udp" accept'
[ "$(wc -l < "$RULES" | tr -d ' ')" = 12 ] || fail 'missing runtime or permanent firewalld rules'
absent "$CALLS" '--add-port|--add-service|--reload|--enable|source address="(127\.|100\.64\.|0\.)'
reset_calls
bash "$REPO/scripts/firewall.sh" auto 47900 47901 > "$TEST_DIR/output"
absent "$CALLS" '--add-rich-rule='

reset_calls
bash "$REPO/scripts/firewall.sh" print 47900 > "$TEST_DIR/output"
contains "$TEST_DIR/output" '--zone=home'
contains "$TEST_DIR/output" '--zone=public'
absent "$CALLS" 'sudo\||--add-rich-rule=|ufw\|'

reset_calls
: > "$ROUTES"
if bash "$REPO/scripts/firewall.sh" auto 47900 > "$TEST_DIR/output" 2>&1; then
    fail 'missing LAN subnet accepted'
fi
contains "$TEST_DIR/output" 'No directly connected IPv4 LAN subnet found'
absent "$CALLS" 'ufw\|allow|--add-'

INSTALL="$TEST_DIR/install with spaces"
mkdir -p "$INSTALL/scripts" "$INSTALL/target/release"
cp "$REPO/scripts/latticed-master" "$REPO/scripts/latticed-worker" "$INSTALL/scripts/"
cat > "$INSTALL/scripts/latticed" <<'EOF'
#!/usr/bin/env bash
printf 'binary:%s\n' "$0"
printf 'arg:<%s>\n' "$@"
printf 'pid:%s\n' "$$"
exit 23
EOF
chmod +x "$INSTALL/scripts/latticed"
cp "$INSTALL/scripts/latticed" "$INSTALL/target/release/latticed"
cp "$INSTALL/scripts/latticed" "$BIN/latticed"
for role in master worker; do
    result=0
    bash -c 'printf "launcher:%s\n" "$$"; exec "$@"' bash "$INSTALL/scripts/latticed-$role" '--name' 'value with spaces' '' '*' > "$TEST_DIR/output" || result=$?
    [ "$result" = 23 ] || fail 'launcher did not preserve exit status'
    contains "$TEST_DIR/output" "binary:$INSTALL/scripts/latticed"
    contains "$TEST_DIR/output" "arg:<$role>"
    contains "$TEST_DIR/output" 'arg:<value with spaces>'
    contains "$TEST_DIR/output" 'arg:<>'
    contains "$TEST_DIR/output" 'arg:<*>'
    [ "$(sed -n 's/^launcher://p' "$TEST_DIR/output")" = "$(sed -n 's/^pid://p' "$TEST_DIR/output")" ] || fail 'launcher did not exec binary'
done
rm "$INSTALL/scripts/latticed"
"$INSTALL/scripts/latticed-worker" > "$TEST_DIR/output" || [ "$?" = 23 ]
contains "$TEST_DIR/output" "binary:$INSTALL/scripts/../target/release/latticed"
rm "$INSTALL/target/release/latticed"
"$INSTALL/scripts/latticed-worker" > "$TEST_DIR/output" || [ "$?" = 23 ]
contains "$TEST_DIR/output" "binary:$BIN/latticed"
rm "$BIN/latticed"
result=0
"$INSTALL/scripts/latticed-master" > "$TEST_DIR/output" 2>&1 || result=$?
[ "$result" = 127 ] || fail 'missing binary exit code'
contains "$TEST_DIR/output" './scripts/setup.sh'
printf '%s\n' 'Launcher and firewall tests passed.'
