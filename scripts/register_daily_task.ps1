<#
.SYNOPSIS
Registers the NHLAnalyzer daily run as a Windows scheduled task.

.DESCRIPTION
Runs `python -m pipeline.scheduled_run` from the repo's .venv every day at
-EasternTime (10:00 by default), converted to this PC's clock.  The task wakes
the PC from sleep to run, and runs whether or not anyone is logged on.  It
cannot start a PC that is shut down; if the PC was off at that time, the task
runs at the next startup, and games already under way by then are not logged.

Run once, from an elevated PowerShell in the repo root:
    powershell -ExecutionPolicy Bypass -File scripts\register_daily_task.ps1

Run it again to change the time; it replaces the old task.
#>
param(
    [string]$EasternTime = "10:00",
    [string]$TaskName = "NHLAnalyzer daily",
    [string]$User = "$env:USERDOMAIN\$env:USERNAME",
    [switch]$Push
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$python = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    throw "No virtualenv at $python. Create it first: py -3.13 -m venv .venv"
}

# Triggers run on the local clock, and the schedule is set in Eastern.  The
# conversion is done once, here: on a PC outside the US daylight-saving rules
# the run drifts by an hour twice a year.
$eastern = [TimeZoneInfo]::FindSystemTimeZoneById("Eastern Standard Time")
$at = [datetime]::ParseExact($EasternTime, "HH:mm", [Globalization.CultureInfo]::InvariantCulture)
$local = [TimeZoneInfo]::ConvertTime($at, $eastern, [TimeZoneInfo]::Local)

$taskArgs = "-m pipeline.scheduled_run"
if ($Push) { $taskArgs += " --push" }

$action = New-ScheduledTaskAction -Execute $python -Argument $taskArgs -WorkingDirectory $root
$trigger = New-ScheduledTaskTrigger -Daily -At $local
$settings = New-ScheduledTaskSettingsSet -WakeToRun -StartWhenAvailable `
    -ExecutionTimeLimit (New-TimeSpan -Hours 3) -MultipleInstances IgnoreNew `
    -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
# S4U runs whether or not anyone is logged on, without storing a password.
# It has no network credentials.  The run does not need any, but it is why
# -Push may fail to authenticate.
$principal = New-ScheduledTaskPrincipal -UserId $User -LogonType S4U -RunLevel Limited

Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger `
    -Settings $settings -Principal $principal -Force | Out-Null

# WakeToRun does nothing unless wake timers are allowed on mains power.
powercfg /SETACVALUEINDEX SCHEME_CURRENT SUB_SLEEP RTCWAKE 1
powercfg /SETACTIVE SCHEME_CURRENT

Write-Host ("Registered '{0}': daily at {1:HH:mm} local ({2} Eastern), as {3}." -f `
    $TaskName, $local, $EasternTime, $User)
Write-Host "Run it now (this posts to Discord):  Start-ScheduledTask -TaskName '$TaskName'"
Write-Host "Logs: $root\logs"
