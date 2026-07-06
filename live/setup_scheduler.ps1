# setup_scheduler.ps1
# Creates a Windows Task Scheduler job that runs s128_live.py daily at 4:30 PM.
#
# WHY DAILY: Task Scheduler has no reliable "last trading day of month" trigger.
# The script itself checks the date and only writes to the CSV on the correct day.
# On all other days it runs for ~30 seconds, prints a preview, and exits.
# This is the most reliable approach on Windows.
#
# Usage: Right-click this file -> "Run with PowerShell"
#   or:  powershell -ExecutionPolicy Bypass -File setup_scheduler.ps1

$pythonPath = (Get-Command python -ErrorAction Stop).Source
$scriptPath = (Resolve-Path "$PSScriptRoot\s128_live.py").Path
$taskName   = "S128_SectorRotation_MonthlySignal"
$logPath    = "$PSScriptRoot\scheduler_log.txt"

Write-Host "Python : $pythonPath"
Write-Host "Script : $scriptPath"
Write-Host "Task   : $taskName"
Write-Host ""

# Remove old task if it exists
schtasks /delete /tn $taskName /f 2>$null
if ($LASTEXITCODE -eq 0) { Write-Host "Removed existing task." }

# Build the command string for schtasks
# Runs: python "C:\...\s128_live.py" >> scheduler_log.txt 2>&1
# Output is appended to scheduler_log.txt so you can see what happened when it ran.
$cmd = "`"$pythonPath`" `"$scriptPath`" >> `"$logPath`" 2>&1"

schtasks /create `
    /tn $taskName `
    /tr "cmd /c $cmd" `
    /sc daily `
    /st 16:30 `
    /f `
    /rl LIMITED

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "Task '$taskName' created successfully."
    Write-Host ""
    Write-Host "Schedule: runs every day at 4:30 PM."
    Write-Host "  -> On the last trading day of the month: logs signal to s128_log.csv"
    Write-Host "  -> All other days: prints a preview and exits (nothing logged)"
    Write-Host "  -> Output saved to: $logPath"
    Write-Host ""
    Write-Host "Useful commands:"
    Write-Host "  View task     : schtasks /query /tn '$taskName' /fo LIST"
    Write-Host "  Run now       : schtasks /run /tn '$taskName'"
    Write-Host "  Delete task   : schtasks /delete /tn '$taskName' /f"
    Write-Host ""
    Write-Host "Note: your PC must be ON and logged in at 4:30 PM on month-end."
    Write-Host "If it's off, run manually: python s128_live.py --force"
} else {
    Write-Host ""
    Write-Host "ERROR: task creation failed (exit code $LASTEXITCODE)."
    Write-Host "Try running PowerShell as Administrator."
}
