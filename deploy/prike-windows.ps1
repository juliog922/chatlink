# Prike tunnel on Windows (WSL reaches the server through it).
# Paste into PowerShell as Administrator. The keys are read from the
# PRIKE_WG_* lines of the repo's .env inside WSL; nothing else holds them.
# The endpoint stays a name so Windows can use IPv6: the server does not
# answer WireGuard over IPv4.
$Repo = '~/kapalua/chatlink'          # the repo, inside your default WSL distro
$ErrorActionPreference = 'Stop'
$wg = Join-Path $env:ProgramFiles 'WireGuard\wireguard.exe'
if (-not (Test-Path $wg)) {
    winget install --id WireGuard.WireGuard -e --silent --accept-source-agreements --accept-package-agreements | Out-Null
}

# Leftovers of earlier tests: tunnels and key files.
foreach ($t in 'prike', 'prikev4', 'prikev6') {
    if (Get-Service "WireGuardTunnel`$$t" -ErrorAction SilentlyContinue) { & $wg /uninstalltunnelservice $t; Start-Sleep 3 }
}
Remove-Item C:\Temp\prikev4.conf, C:\Temp\prikev6.conf -ErrorAction SilentlyContinue

# The keys, from .env in WSL.
$e = @{}
foreach ($line in (wsl.exe -e sh -lc "cat $Repo/.env")) {
    if ($line -match '^(PRIKE_WG_[A-Z_]+)=(.*)$') { $e[$Matches[1]] = $Matches[2].Trim() }
}
foreach ($k in 'PRIKE_WG_PRIVATE_KEY', 'PRIKE_WG_ADDRESS', 'PRIKE_WG_PEER_PUBLIC_KEY', 'PRIKE_WG_ENDPOINT', 'PRIKE_WG_ALLOWED_IPS') {
    if (-not $e[$k]) { throw "Missing $k in $Repo/.env" }
}

# The config, readable only by SYSTEM and Administrators.
$dir = Join-Path $env:ProgramData 'Prike'
New-Item -ItemType Directory -Force -Path $dir | Out-Null
icacls $dir /inheritance:r /grant:r '*S-1-5-18:(OI)(CI)F' '*S-1-5-32-544:(OI)(CI)F' | Out-Null
$psk = if ($e['PRIKE_WG_PRESHARED_KEY']) { "PresharedKey = $($e['PRIKE_WG_PRESHARED_KEY'])" } else { '' }
$conf = Join-Path $dir 'prike.conf'
@"
[Interface]
PrivateKey = $($e['PRIKE_WG_PRIVATE_KEY'])
Address = $($e['PRIKE_WG_ADDRESS'])

[Peer]
PublicKey = $($e['PRIKE_WG_PEER_PUBLIC_KEY'])
$psk
Endpoint = $($e['PRIKE_WG_ENDPOINT'])
AllowedIPs = $($e['PRIKE_WG_ALLOWED_IPS'])
PersistentKeepalive = 25
"@ | Set-Content -Encoding ascii $conf

# Install as a service (starts with Windows), then wait for the server.
& $wg /installtunnelservice $conf
$server = ($e['PRIKE_WG_ALLOWED_IPS'] -split '/')[0]
for ($i = 0; $i -lt 20; $i++) {
    if (Test-Connection $server -Count 1 -Quiet) { Write-Host "Connected: $server answers." -ForegroundColor Green; break }
    Start-Sleep 2
}
if ($i -eq 20) { Write-Host "Tunnel installed but $server does not answer. Send this screen to Asterio." -ForegroundColor Red }
Write-Host "Disconnect: & '$wg' /uninstalltunnelservice prike"
