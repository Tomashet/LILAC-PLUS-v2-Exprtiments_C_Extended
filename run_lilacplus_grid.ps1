$methods = @(
    "",
    "--lilac --constraint none",
    "--lilac --constraint cpss",
    "--lilac --constraint adjust_speed"
)

$pstay = 1.0, 0.85
$seeds = 0, 1
$total_steps = 3000

foreach ($m in $methods) {
    foreach ($p in $pstay) {
        foreach ($s in $seeds) {

            $cmd = "python -m scripts.train_continuous --env highway-v0 --p_stay $p --seed $s --total_steps $total_steps $m"
            Write-Host ""
            Write-Host "===================================================="
            Write-Host "Running: $cmd"
            Write-Host "===================================================="
            Invoke-Expression $cmd
        }
    }
}